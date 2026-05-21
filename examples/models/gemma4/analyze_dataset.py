"""Quick analyzer for the translation RL dataset.

Reports:
- Total sample count.
- Target-language distribution (parsed from translation directives).
- Source script detection by Unicode range majority vote.
- "source==target" trivial fraction (likely to collapse GRPO group std to 0).
- Source-text and full-prompt length percentiles.
- Top src->tgt language pairs.

Usage:
    python examples/models/gemma4/analyze_dataset.py \
        --input /llm-align/liuchonghan/data/translation_rl_train.jsonl
"""

import argparse
import json
import re
from collections import Counter
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

# Match the directive header, then look at the short window after it for the
# language keyword. Searching the whole prompt risks matching mentions of
# languages in the source text itself.
TARGET_DIRECTIVE = re.compile(
    r'(翻译为|翻译成|目标语言为[:：]|translate (?:into|to))', re.IGNORECASE)

SOURCE_TAG_RE = re.compile(r'<待翻译文本开始>(.*?)<待翻译文本结束>', re.S)
SOURCE_AFTER_RE = re.compile(
    r'(?:翻译内容如下[:：]|翻译为[一-鿿]+[:：])\s*(.+)$', re.S)

SAME_SCRIPT_TARGETS = {'zh', 'ja', 'ko', 'lo', 'th', 'ar', 'ru', 'he', 'hi'}


def detect_target_lang(prompt: str) -> Optional[str]:
    for m in TARGET_DIRECTIVE.finditer(prompt):
        window = prompt[m.end():m.end() + 30]
        for pat, code in TARGET_PATTERNS:
            if pat.search(window):
                return code
    return None


def extract_source(user_content: str) -> str:
    m = SOURCE_TAG_RE.search(user_content)
    if m:
        return m.group(1).strip()
    m = SOURCE_AFTER_RE.search(user_content)
    if m:
        return m.group(1).strip()
    parts = [p for p in user_content.split('\n') if p.strip()]
    return parts[-1].strip() if parts else ''


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


def percentile(sorted_arr, p: float):
    return sorted_arr[min(int(len(sorted_arr) * p), len(sorted_arr) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--show-pairs', type=int, default=20)
    args = ap.parse_args()

    total = 0
    target_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    source_lens = []
    prompt_lens = []
    trivial = 0
    no_target = 0
    short_source = 0

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get('messages') or []
            user = next((m.get('content', '') for m in msgs if m.get('role') == 'user'), '')
            system = next((m.get('content', '') for m in msgs if m.get('role') == 'system'), '')
            total += 1

            tgt = detect_target_lang(user) or detect_target_lang(system)
            if tgt is None:
                no_target += 1
                target_counts['<unknown>'] += 1
                continue
            target_counts[tgt] += 1

            src = extract_source(user)
            source_lens.append(len(src))
            prompt_lens.append(len(user) + len(system))

            if len(src) < 10:
                short_source += 1

            src_script = detect_source_script(src) or '?'
            pair_counts[f'{src_script}->{tgt}'] += 1
            if src_script == tgt and tgt in SAME_SCRIPT_TARGETS:
                trivial += 1

    print(f'\nTotal samples: {total:,}\n')

    print('=== Target language distribution ===')
    for lang, n in target_counts.most_common():
        print(f'  {lang:10s} {n:>7,}  ({n / total:5.1%})')

    print('\n=== Likely zero-variance ("trivial") samples ===')
    print(f'  source-script == target (e.g. zh->zh, en-letters->en):')
    print(f'      {trivial:,} / {total:,}  ({trivial / total:5.1%})')
    print(f'  target directive not parseable:')
    print(f'      {no_target:,} / {total:,}  ({no_target / total:5.1%})')
    print(f'  source text < 10 chars (very short, likely trivial):')
    print(f'      {short_source:,} / {total:,}  ({short_source / total:5.1%})')

    print(f'\n=== Top {args.show_pairs} src_script->tgt pairs ===')
    for pair, n in pair_counts.most_common(args.show_pairs):
        marker = '  TRIVIAL' if pair.split('->')[0] == pair.split('->')[1] else ''
        print(f'  {pair:18s} {n:>7,}  ({n / total:5.1%}){marker}')

    if source_lens:
        source_lens.sort()
        prompt_lens.sort()
        print('\n=== Source text length (chars) ===')
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            print(f'  P{int(p * 100):2d}: {percentile(source_lens, p):>6,}')
        print(f'  max: {source_lens[-1]:,}')

        print('\n=== Full prompt length (user+system, chars) ===')
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            print(f'  P{int(p * 100):2d}: {percentile(prompt_lens, p):>6,}')
        print(f'  max: {prompt_lens[-1]:,}')


if __name__ == '__main__':
    main()
