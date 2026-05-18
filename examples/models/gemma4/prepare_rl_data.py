"""Sample the SFT translation parquet files into a GRPO-ready JSONL.

Each output row is a single ms-swift-style messages list containing only the
system (when present) and final user turn. Any assistant turns from the SFT
data are dropped because GRPO needs the model to roll out its own candidates.

Defaults match the dayuzhong layout under /llm-align/liuchonghan; override via
CLI flags. Sampling is proportional to file size so the file mix is preserved.
Run example:

    python3 examples/models/gemma4/prepare_rl_data.py \
        --src-dir /llm-align/liuchonghan/ins_dataset/ins_dataset/dayuzhong \
        --out /llm-align/liuchonghan/data/translation_rl_train.jsonl \
        --total 50000
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DEFAULT_FILES = [
    'dayuzhong_instruct_gemini3.1_5w.parquet',
    'dyz_trans_10w_gemini3.1.parquet',
    'filtered_chat.parquet',
    'lowest_metricx_train_selected.cleaned.messages.parquet',
    'trans_terms_for_train_data.parquet',
]


def normalize_messages(raw, max_user_chars: int) -> Optional[List[Dict[str, str]]]:
    """Return system+user messages only, or None if the row is unusable."""
    if raw is None:
        return None
    try:
        msgs = list(raw)
    except TypeError:
        return None

    out: List[Dict[str, str]] = []
    seen_user = False
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get('role') or 'user'
        content = m.get('content')
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        if role == 'system' and not seen_user:
            out.append({'role': 'system', 'content': content})
        elif role == 'user':
            out.append({'role': 'user', 'content': content})
            seen_user = True
        # assistant turns are intentionally discarded

    if not seen_user:
        return None
    # Trim back-to-back system messages: keep the first system + the LAST user.
    last_user_idx = max(i for i, m in enumerate(out) if m['role'] == 'user')
    sys_msgs = [m for m in out[:last_user_idx] if m['role'] == 'system'][:1]
    final = sys_msgs + [out[last_user_idx]]
    total_chars = sum(len(m['content']) for m in final)
    if total_chars > max_user_chars:
        return None
    return final


def proportional_quotas(sizes: List[int], total: int) -> List[int]:
    """Largest-remainder method so the sum exactly equals `total`."""
    grand = sum(sizes)
    if grand == 0:
        return [0] * len(sizes)
    raw = [s * total / grand for s in sizes]
    floors = [int(math.floor(x)) for x in raw]
    remainder = total - sum(floors)
    fracs = sorted(enumerate(raw), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True)
    for i, _ in fracs[:remainder]:
        floors[i] += 1
    return [min(floors[i], sizes[i]) for i in range(len(sizes))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='/llm-align/liuchonghan/ins_dataset/ins_dataset/dayuzhong')
    parser.add_argument('--out', default='/llm-align/liuchonghan/data/translation_rl_train.jsonl')
    parser.add_argument('--total', type=int, default=50000)
    parser.add_argument('--files', nargs='+', default=DEFAULT_FILES,
                        help='Parquet filenames under --src-dir.')
    parser.add_argument('--max-user-chars', type=int, default=6000,
                        help='Drop rows whose system+user content exceeds this many chars.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dedup', action='store_true', default=True,
                        help='Drop exact-duplicate user prompts across all files.')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src_dir = Path(args.src_dir)
    paths = [src_dir / f for f in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f'Missing input files: {missing}')

    print(f'[prep] loading {len(paths)} parquet files from {src_dir}', flush=True)
    frames = []
    sizes = []
    for p in paths:
        df = pd.read_parquet(p, columns=['messages'])
        frames.append(df)
        sizes.append(len(df))
        print(f'[prep]   {p.name}: {len(df):,} rows', flush=True)

    quotas = proportional_quotas(sizes, args.total)
    print(f'[prep] proportional quotas: '
          + ', '.join(f'{p.name}={q:,}' for p, q in zip(paths, quotas)),
          flush=True)

    seen_user_prompts = set()
    written = 0
    dropped_malformed = 0
    dropped_overlong = 0
    dropped_dup = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')

    with tmp_path.open('w', encoding='utf-8') as fout:
        for path, df, quota in zip(paths, frames, quotas):
            if quota == 0:
                continue
            indices = list(range(len(df)))
            rng.shuffle(indices)
            taken_from_file = 0
            for idx in indices:
                if taken_from_file >= quota:
                    break
                row_msgs = normalize_messages(df.iloc[idx]['messages'], args.max_user_chars)
                if row_msgs is None:
                    # Distinguish malformed vs over-length isn't critical; lump together.
                    dropped_malformed += 1
                    continue
                user_text = row_msgs[-1]['content']
                if args.dedup:
                    if user_text in seen_user_prompts:
                        dropped_dup += 1
                        continue
                    seen_user_prompts.add(user_text)
                fout.write(json.dumps({'messages': row_msgs}, ensure_ascii=False) + '\n')
                taken_from_file += 1
                written += 1
            print(f'[prep]   sampled {taken_from_file:,} from {path.name}', flush=True)

    os.replace(tmp_path, out_path)
    print(f'[prep] wrote {written:,} rows -> {out_path}', flush=True)
    print(f'[prep]   dropped malformed/overlong: {dropped_malformed:,}', flush=True)
    if args.dedup:
        print(f'[prep]   dropped duplicates: {dropped_dup:,}', flush=True)


if __name__ == '__main__':
    main()
