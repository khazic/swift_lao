"""Translation-quality LLM-judge reward for GRPO.

The judge is an OpenAI-compatible chat endpoint that takes one prompt plus one
candidate translation and returns a JSON document with a score in [0, 100].
Each completion is scored independently in its own judge call, which keeps the
input short and lets the judge focus on a single candidate at a time. The
judge prompt asks only for the score field (no reason / ranking text), which
keeps output tokens small and JSON parsing reliable.

Required env vars:
    TRANSLATION_JUDGE_API_BASE        OpenAI-compatible base URL.
    TRANSLATION_JUDGE_API_KEY         Bearer token for the judge.
    TRANSLATION_JUDGE_MODEL           Model name to send in the request body.
    TRANSLATION_JUDGE_NUM_GENERATIONS Must equal --num_generations on the
                                      training command line.

Optional env vars:
    TRANSLATION_JUDGE_TIMEOUT         Seconds per call, default 180.
    TRANSLATION_JUDGE_TEMPERATURE     Sampling temperature, default 0.0.
    TRANSLATION_JUDGE_MAX_RETRIES     Retry budget per group, default 2.
    TRANSLATION_JUDGE_FALLBACK_REWARD Reward when the judge fails for a single
                                      completion, default 0.0.
    TRANSLATION_JUDGE_MAX_CONCURRENCY Max simultaneous in-flight judge calls
                                      from this rank, default 32.

Register name: ``translation_judge``. Use via
    --external_plugins examples/train/grpo/plugin/translation_judge.py \
    --reward_funcs translation_judge
"""

import asyncio
import json
import os
import random
import re
import uuid
from typing import List, Optional

import aiohttp

from swift.rewards import AsyncORM, orms
from swift.utils import get_logger

logger = get_logger()


JUDGE_PROMPT_TEMPLATE = """\
## 任务：你是一名翻译质量诊断专家。对一条翻译结果进行打分。

## 评判规则（仅用于内部思考，不要输出）：
1、input 中 prompt 是原文及翻译要求，correct_answer 是模型的翻译结果。
2、按以下维度评估翻译质量：
   - 准确性（误添加、误译、遗漏、未翻译文本）
   - 流畅性（语法、字符编码、标点、语域、拼写）
   - 风格（自然度、是否怪异 / 尴尬）
   - 术语（一致性、上下文匹配）
3、评分参考：
   - 0   = 翻译完全错误，没有价值
   - 33  = 翻译部分有价值或较多语法错误
   - 66  = 翻译大部分有价值或少量语法错误
   - 100 = 完美的翻译和语法
4、综合参照翻译要求、错误类别及严重性、专业翻译标准，给出 0-100 分。
5、即使原文语义模糊，也要按模型的实际表现打分，不要拒绝评分。
6、拿到的样本必须是QA的直接翻译，不应出现模型的其他见解，不然给低分处理

## 输出格式（严格 JSON，仅输出以下结构；不要任何额外文字、不要代码块包裹）：
{"models": [{"model_name": "model_0", "score": 85}]}

## 输出要求：
1、严格按上述 JSON 输出，不输出任何解释、说明、思考过程或代码块。
2、models 数组只包含一个元素，model_name 固定为 "model_0"。
3、score 必须是 0-100 之间的数字（整数或浮点）。

## input：__JUDGE_INPUT_PAYLOAD__
"""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f'TranslationJudgeReward requires env var {name}. '
            'See examples/train/grpo/plugin/translation_judge.py docstring.')
    return value


class TranslationJudgeReward(AsyncORM):

    def __init__(self, args, **kwargs):
        super().__init__(args)
        base = _require_env('TRANSLATION_JUDGE_API_BASE').rstrip('/')
        if not base.endswith('/v1'):
            base = base + '/v1'
        self.api_base = base
        self.api_key = _require_env('TRANSLATION_JUDGE_API_KEY')
        self.model_name = _require_env('TRANSLATION_JUDGE_MODEL')
        self.num_generations = int(_require_env('TRANSLATION_JUDGE_NUM_GENERATIONS'))
        self.timeout = float(os.environ.get('TRANSLATION_JUDGE_TIMEOUT', '180'))
        self.temperature = float(os.environ.get('TRANSLATION_JUDGE_TEMPERATURE', '0.0'))
        self.max_retries = int(os.environ.get('TRANSLATION_JUDGE_MAX_RETRIES', '2'))
        self.fallback_reward = float(os.environ.get('TRANSLATION_JUDGE_FALLBACK_REWARD', '0.0'))
        self.max_concurrency = int(os.environ.get('TRANSLATION_JUDGE_MAX_CONCURRENCY', '32'))
        self._semaphore: Optional[asyncio.Semaphore] = None
        logger.info(
            f'TranslationJudgeReward initialized | base={self.api_base} | '
            f'model={self.model_name} | num_generations={self.num_generations} | '
            f'timeout={self.timeout}s | max_concurrency={self.max_concurrency}')

    @staticmethod
    def _last_user_content(msg_list) -> str:
        for msg in reversed(msg_list):
            if msg.get('role') == 'user':
                return msg.get('content', '') or ''
        return ''

    @staticmethod
    def _build_input_payload(prompt: str, completions: List[str]) -> str:
        models = [{
            'answer_id': f'a{i}',
            'model_name': f'model_{i}',
            'answer': c,
        } for i, c in enumerate(completions)]
        payload = {
            'case_id': uuid.uuid4().hex[:12],
            'prompt': prompt,
            'correct_answer': models,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _parse_scores(raw: str, expected_n: int) -> Optional[List[Optional[float]]]:
        text = raw.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r'\{.*\}', text, re.S)
            if not match:
                return None
            try:
                obj = json.loads(match.group(0))
            except Exception:
                return None
        if not isinstance(obj, dict):
            return None
        models = obj.get('models')
        if not isinstance(models, list) or not models:
            return None

        out: List[Optional[float]] = [None] * expected_n
        unassigned: List[Optional[float]] = []
        for entry in models:
            try:
                score = float(entry.get('score'))
            except (TypeError, ValueError):
                score = None
            name = str(entry.get('model_name', ''))
            m = re.match(r'model_(\d+)$', name)
            if m and score is not None:
                idx = int(m.group(1))
                if 0 <= idx < expected_n:
                    out[idx] = score
                    continue
            unassigned.append(score)

        if all(s is None for s in out) and unassigned:
            for i, s in enumerate(unassigned[:expected_n]):
                out[i] = s
        return out

    async def _score_group(self, session: aiohttp.ClientSession, prompt: str,
                           completions: List[str], allow_split: bool = True) -> List[float]:
        n = len(completions)
        # Shuffle candidate order so judge position-bias does not always pin
        # the same training sample to e.g. model_0. `indices[j]` is the
        # original index of whatever ends up at shuffled position j.
        indices = list(range(n))
        if n > 1:
            random.shuffle(indices)
        shuffled = [completions[indices[j]] for j in range(n)]

        user_input = self._build_input_payload(prompt, shuffled)
        user_message = JUDGE_PROMPT_TEMPLATE.replace('__JUDGE_INPUT_PAYLOAD__', user_input)
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': user_message}],
            'temperature': self.temperature,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        url = f'{self.api_base}/chat/completions'

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._semaphore:
                    async with session.post(
                            url, json=payload, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                        if resp.status != 200:
                            last_err = f'HTTP {resp.status}: {(await resp.text())[:300]}'
                            logger.warning(f'judge attempt {attempt} failed: {last_err}')
                            continue
                        data = await resp.json()
                content = data['choices'][0]['message']['content']
                scores = self._parse_scores(content, n)
                if scores is None or all(s is None for s in scores):
                    last_err = f'unparseable judge response: {content[:200]!r}'
                    logger.warning(last_err)
                    continue
                # scores[j] is the score for model_j == shuffled position j;
                # map back to original completion order.
                unshuffled: List[float] = [self.fallback_reward] * n
                for j, orig in enumerate(indices):
                    s = scores[j]
                    if s is not None:
                        unshuffled[orig] = max(0.0, min(1.0, s / 100.0))
                return unshuffled
            except asyncio.TimeoutError:
                last_err = 'timeout'
                logger.warning('judge call timed out')
            except Exception as e:
                last_err = str(e)
                logger.warning(f'judge call error: {e}')

        # Comparative call failed. Fall back to per-completion scoring before
        # giving up: one bad API call would otherwise zero an entire GRPO
        # group and produce zero advantage variance.
        if allow_split and n > 1:
            logger.warning(
                f'judge group failed after {self.max_retries + 1} attempts ({last_err}); '
                f'falling back to {n} per-completion judge calls')
            per_calls = await asyncio.gather(*[
                self._score_group(session, prompt, [c], allow_split=False) for c in completions
            ])
            return [r[0] for r in per_calls]

        logger.warning(f'judge group failed after {self.max_retries + 1} attempts: {last_err}')
        return [self.fallback_reward] * n

    async def __call__(self, completions: List[str], messages, **kwargs) -> List[float]:
        n = len(completions)
        if n == 0:
            return []
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

        # Per-completion scoring: one judge call per completion.
        # Why: with large num_generations the comparative call carries N long
        # candidates, risks output truncation, and tends to compress scores
        # across the group, hurting advantage variance.
        groups = []
        for i in range(n):
            prompt = self._last_user_content(messages[i])
            groups.append((prompt, [completions[i]]))

        connector = aiohttp.TCPConnector(limit=self.max_concurrency)
        # trust_env=True so aiohttp honors HTTP_PROXY / HTTPS_PROXY / NO_PROXY
        # from the environment (the cluster may force outbound through a proxy
        # and we want NO_PROXY=api.360.cn to bypass it).
        async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
            tasks = [self._score_group(session, p, c) for p, c in groups]
            results = await asyncio.gather(*tasks)

        rewards: List[float] = []
        for r in results:
            rewards.extend(r)
        if len(rewards) != n:
            logger.error(f'reward count mismatch: expected {n}, got {len(rewards)}')
            rewards = (rewards + [self.fallback_reward] * n)[:n]
        return rewards


orms['translation_judge'] = TranslationJudgeReward
