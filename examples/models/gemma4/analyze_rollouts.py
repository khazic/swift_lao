"""Analyze rollout distribution and per-group reward variance in a
completions.jsonl produced by swift's GRPO trainer.

Each line of the jsonl is one training step's full batch
(N = num_generations * prompts_per_step). Within a batch, consecutive
entries with identical prompts form one GRPO group.

Reports:
- Number of steps and groups seen.
- Group-reward statistics (mean / std / spread / zero-variance share).
- Target-language and src_script -> tgt_lang breakdown with per-bucket
  reward statistics, so you can tell which prompt types give learning
  signal vs which collapse to uniform reward.
- Overall reward-value histogram.

Usage:
    python examples/models/gemma4/analyze_rollouts.py \
        --input output/gemma4_31b_grpo_translation_judge/completions.jsonl
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Optional


TARGET_PATTERNS = [
    (re.compile(r'(简体中文|繁体中文|中文)'), 'zh'),
    (re.compile(r'(英语|英文|English)', re.IGNORECASE), 'en'),
    (re.compile(r'(日语|日文|Japanese)', re.IGNORECASE), 'ja'),
    (re.compile(r'(韩语|韩文|Korean)', re.IGNORECASE), 'ko'),
    (re.compile(r'(法语|法文|French)', re.IGNORECASE), 'fr'),
    (re.compile(r'(德语|德文|German)', re.IGNORECASE), 'de'),
    (re.compile(r'(西班牙语|西文|Spanish)', re.IGNORECASE), 'es'),
    (re.compile(r'(葡萄牙语|葡文|Portuguese)', re.IGNORECASE), 'pt'),
    (re.compile(r'(意大利语|意文|Italian)', re.IGNORECASE), 'it'),
    (re.compile(r'(俄语|俄文|Russian)', re.IGNORECASE), 'ru'),
    (re.compile(r'(阿拉伯语|阿文|Arabic)', re.IGNORECASE), 'ar'),
    (re.compile(r'(越南语|Vietnamese)', re.IGNORECASE), 'vi'),
    (re.compile(r'(泰语|Thai)', re.IGNORECASE), 'th'),
    (re.compile(r'(老挝语|寮语|Lao)', re.IGNORECASE), 'lo'),
    (re.compile(r'(印尼语|Indonesian)', re.IGNORECASE), 'id'),
    (re.compile(r'(马来语|Malay)', re.IGNORECASE), 'ms'),
    (re.compile(r'(土耳其语|Turkish)', re.IGNORECASE), 'tr'),
    (re.compile(r'(波兰语|Polish)', re.IGNORECASE), 'pl'),
    (re.compile(r'(丹麦语|Danish)', re.IGNORECASE), 'da'),
    (re.compile(r'(瑞典语|Swedish)', re.IGNORECASE), 'sv'),
    (re.compile(r'(挪威语|Norwegian)', re.IGNORECASE), 'no'),
    (re.compile(r'(芬兰语|Finnish)', re.IGNORECASE), 'fi'),
    (re.compile(r'(荷兰语|Dutch)', re.IGNORECASE), 'nl'),
    (re.compile(r'(捷克语|Czech)', re.IGNORECASE), 'cs'),
    (re.compile(r'(斯洛伐克语|Slovak)', re.IGNORECASE), 'sk'),
    (re.compile(r'(乌克兰语|Ukrainian)', re.IGNORECASE), 'uk'),
    (re.compile(r'(希腊语|Greek)', re.IGNORECASE), 'el'),
    (re.compile(r'(希伯来语|Hebrew)', re.IGNORECASE), 'he'),
    (re.compile(r'(印地语|Hindi)', re.IGNORECASE), 'hi'),
]

TARGET_DIRECTIVE = re.compile(
    r'(翻译为|翻译成|目标语言为[:：]|translate (?:into|to))', re.IGNORECASE)

SOURCE_TAG_RE = re.compile(r'<待翻译文本开始>(.*?)<待翻译文本结束>', re.S)
SOURCE_AFTER_RE = re.compile(
    r'(?:翻译内容如下[:：]|翻译为[一-鿿]+[:：])\s*(.+?)(?:<turn\|>|$)', re.S)


def detect_target_lang(prompt: str) -> Optional[str]:
    for m in TARGET_DIRECTIVE.finditer(prompt):
        window = prompt[m.end():m.end() + 30]
        for pat, code in TARGET_PATTERNS:
            if pat.search(window):
                return code
    return None


def extract_source(prompt: str) -> str:
    m = SOURCE_TAG_RE.search(prompt)
    if m:
        return m.group(1).strip()
    m = SOURCE_AFTER_RE.search(prompt)
    if m:
        return m.group(1).strip()
    return ''


def script_class(c: str) -> Optional[str]:
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF:
        return 'zh'
    if 0x3040 <= o <= 0x30FF:
        return 'ja'
    if 0xAC00 <= o <= 0xD7AF:
        return 'ko'
    if 0x0E80 <= o <= 0x0EFF:
        return 'lo'
    if 0x0E00 <= o <= 0x0E7F:
        return 'th'
    if 0x0600 <= o <= 0x06FF:
        return 'ar'
    if 0x0400 <= o <= 0x04FF:
        return 'ru'
    if 0x0590 <= o <= 0x05FF:
        return 'he'
    if 0x0900 <= o <= 0x097F:
        return 'hi'
    if (0x0041 <= o <= 0x005A) or (0x0061 <= o <= 0x007A):
        return 'lat'
    return None


def detect_source_script(text: str) -> Optional[str]:
    if not text.strip():
        return None
    counts: Counter = Counter()
    for c in text:
        cls = script_class(c)
        if cls:
            counts[cls] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def pick_reward_key(batch: dict) -> str:
    for k in batch:
        if 'judge' in k.lower() or 'reward' in k.lower():
            if isinstance(batch[k], list) and batch[k] and isinstance(batch[k][0], (int, float)):
                return k
    raise RuntimeError(f'Cannot infer reward column. Keys: {list(batch.keys())}')


def group_consecutive(prompts, rewards):
    i = 0
    while i < len(prompts):
        j = i + 1
        while j < len(prompts) and prompts[j] == prompts[i]:
            j += 1
        yield prompts[i], rewards[i:j]
        i = j


def fmt_stats(rewards):
    if not rewards:
        return 'mean=  -    std=  -   spread=  -'
    mean = sum(rewards) / len(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    spread = max(rewards) - min(rewards)
    return f'mean={mean:5.3f}  std={std:5.3f}  spread={spread:5.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--reward-key', default=None,
                    help='Reward column name in the jsonl (auto-detect if omitted).')
    ap.add_argument('--top-pairs', type=int, default=15)
    args = ap.parse_args()

    n_steps = 0
    n_groups = 0
    n_dead_groups = 0   # std == 0
    n_weak_groups = 0   # std < 0.02

    by_tgt = defaultdict(list)         # tgt_lang -> list[group_std]
    by_pair = defaultdict(list)        # 'src->tgt' -> list[group_std]
    by_tgt_mean = defaultdict(list)    # tgt_lang -> list[group_mean]
    by_pair_mean = defaultdict(list)   # 'src->tgt' -> list[group_mean]

    reward_hist: Counter = Counter()   # per-completion (rounded to 0.05)
    reward_key = args.reward_key

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            try:
                batch = json.loads(line)
            except json.JSONDecodeError:
                continue
            if reward_key is None:
                reward_key = pick_reward_key(batch)
                print(f'[info] auto-detected reward column: {reward_key!r}\n')

            prompts = batch.get('prompt') or []
            rewards = batch.get(reward_key) or []
            if not prompts or len(prompts) != len(rewards):
                continue
            n_steps += 1

            for prompt, grp_rewards in group_consecutive(prompts, rewards):
                n_groups += 1
                if len(grp_rewards) < 2:
                    continue
                std = statistics.pstdev(grp_rewards)
                mean = sum(grp_rewards) / len(grp_rewards)
                if std == 0:
                    n_dead_groups += 1
                if std < 0.02:
                    n_weak_groups += 1

                tgt = detect_target_lang(prompt) or '<unknown>'
                src_text = extract_source(prompt)
                src_script = detect_source_script(src_text) or '?'
                pair = f'{src_script}->{tgt}'

                by_tgt[tgt].append(std)
                by_pair[pair].append(std)
                by_tgt_mean[tgt].append(mean)
                by_pair_mean[pair].append(mean)

                for r in grp_rewards:
                    bucket = round(round(r / 0.05) * 0.05, 2)
                    reward_hist[bucket] += 1

    if n_steps == 0:
        print('No data parsed.')
        return

    print(f'steps logged : {n_steps:,}')
    print(f'total groups : {n_groups:,}')
    print(f'dead groups  : {n_dead_groups:,}  ({n_dead_groups / n_groups:.1%})  '
          f'(std == 0, advantage = 0 → no signal)')
    print(f'weak groups  : {n_weak_groups:,}  ({n_weak_groups / n_groups:.1%})  '
          f'(std < 0.02, weak signal)')

    print('\n=== Per target language ===')
    print(f'{"tgt":>6}  {"groups":>7}  {"dead%":>6}  {"mean(rew)":>9}  {"mean(std)":>9}')
    for tgt, stds in sorted(by_tgt.items(), key=lambda kv: -len(kv[1])):
        means = by_tgt_mean[tgt]
        dead = sum(1 for s in stds if s == 0) / len(stds)
        print(f'  {tgt:>4}  {len(stds):>7,}  {dead:>5.1%}  '
              f'{sum(means) / len(means):>8.3f}  {sum(stds) / len(stds):>8.3f}')

    print(f'\n=== Top {args.top_pairs} src_script -> tgt pairs ===')
    print(f'{"pair":>12}  {"groups":>7}  {"dead%":>6}  {"mean(rew)":>9}  {"mean(std)":>9}')
    for pair, stds in sorted(by_pair.items(), key=lambda kv: -len(kv[1]))[:args.top_pairs]:
        means = by_pair_mean[pair]
        dead = sum(1 for s in stds if s == 0) / len(stds)
        print(f'  {pair:>10}  {len(stds):>7,}  {dead:>5.1%}  '
              f'{sum(means) / len(means):>8.3f}  {sum(stds) / len(stds):>8.3f}')

    print('\n=== Reward histogram (per completion, 0.05 buckets) ===')
    total_completions = sum(reward_hist.values())
    for bucket in sorted(reward_hist):
        n = reward_hist[bucket]
        bar = '#' * int(60 * n / max(reward_hist.values()))
        print(f'  {bucket:4.2f}  {n:>7,}  ({n / total_completions:5.1%})  {bar}')


if __name__ == '__main__':
    main()
