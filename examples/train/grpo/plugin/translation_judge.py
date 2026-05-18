"""Translation-quality LLM-judge reward for GRPO.

The judge is an OpenAI-compatible chat endpoint that takes one prompt plus N
candidate translations and returns a JSON document containing a per-model score
in [0, 100], an error analysis, and a ranking. We exploit GRPO's group
structure: the N completions sampled for one prompt are sent in a single judge
call as N separate models. This is roughly N times cheaper than scoring each
completion alone and lets the judge score comparatively.

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
import re
import uuid
from typing import List, Optional

import aiohttp

from swift.rewards import AsyncORM, orms
from swift.utils import get_logger

logger = get_logger()


JUDGE_PROMPT_TEMPLATE = """\
## 任务：你是一名翻译质量诊断专家，你的任务是评判以下几个model翻译后的answer结果质量，需要你输出每个model的翻译评分、翻译评分的原因(reason_a)、指出翻译错误(reason_b)、对几个模型的结果进行综合的评判。
## 评判说明：
1、以下input中的prompt是翻译内容及要求，correct_answer是多个模型的翻译结果。
2、标注几个模型翻译结果中的错误类型，并进行分类。
2.1 错误类别包括：准确性（误添加、误译、遗漏、未翻译文本）、流畅性（字符编码、语法、不一致、标点、语域、拼写）、风格（尴尬、怪异）、术语（不适合上下文、使用不一致）、未翻译、其他或无错误。
2.2 每个错误分为两档：重大错误或轻微错误。重大错误会扰乱文本的流畅性，并使文本的可理解性变得困难或不可能。轻微错误是不会显著扰乱流畅性的错误，并且文本试图表达的内容仍然可以理解。
3、对模型的翻译结果评分：
3.1 评分规则：参照prompt中的翻译要求以及其他指令要求检查翻译结果是否符合prompt要求、参照翻译结果的错误类别及严重性、以及你作为专业翻译的补充、结合标记出的错误，综合以上对模型的翻译结果进行评分。
3.2 评分范围：0到100分。评分时需要参考：0="翻译完全错误，没有价值"，33="翻译部分有价值或较多语法错误"，66="翻译大部分有价值或少量语法错误"，最高100="完美的翻译和语法"。
4、为几个模型的翻译结果进行排名：排名规则：根据评分结果和综合判断，输出多个翻译的排名。
## 参考人工评测步骤：xxx
## 输出格式：
{
    "case_id": "xxx",
    "models": [
        {
            "answer_id": "xxx",
            "model_name": "xxx",
            "score": "xxx",
            "skill": "xxx",
            "reason_a": "xxx",
            "reason_b": "xxx",
            "extra": [
                {
                    "a": "xxx",
                    "b": "xxx"
                }
            ]
        }
    ],
    "notes": [
        {
            "model_rank": "xxx",
            "reason": "xxx"
        }
    ]
}
### 格式说明：
1、score：是指裁判给模型结果打分；
2、reason_a：是指裁判打分的理由
3、reason_b：是指裁判对于模型回答标记错误的详细说明，需要详细指出具体的错误是哪几个。如果原语言语义不明放弃打分，在此处备注"无效query"
4、notes：字段model_rank是裁判对多个模型的排名，以及排名的原因（如模型a>模型b>xxx，因为xxx。）。字段reason是裁判对排名第一的模型给出改进的建议。
## 输出要求：
1、严格按输出格式进行输出，不要在输出json格式以外解释任何信息，也不要输出其他与格式无关的任何解释、说明等内容。代码块也不要出现"```json"。
2、input当中有几个model_name的answer，在输出中就要有对应数量的模型平分。
3、当原语言表述不清、语义模糊时，不再进行翻译质量判断。
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
                           completions: List[str]) -> List[float]:
        n = len(completions)
        user_input = self._build_input_payload(prompt, completions)
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
                return [self.fallback_reward if s is None else max(0.0, min(1.0, s / 100.0))
                        for s in scores]
            except asyncio.TimeoutError:
                last_err = 'timeout'
                logger.warning('judge call timed out')
            except Exception as e:
                last_err = str(e)
                logger.warning(f'judge call error: {e}')
        logger.warning(f'judge group failed after {self.max_retries + 1} attempts: {last_err}')
        return [self.fallback_reward] * n

    async def __call__(self, completions: List[str], messages, **kwargs) -> List[float]:
        n = len(completions)
        if n == 0:
            return []
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

        group_size = self.num_generations
        if n % group_size != 0:
            logger.warning(
                f'completions length {n} is not divisible by num_generations '
                f'{group_size}; falling back to per-completion scoring.')
            group_size = 1

        groups = []
        for start in range(0, n, group_size):
            grp_messages = messages[start]
            prompt = self._last_user_content(grp_messages)
            grp_completions = completions[start:start + group_size]
            groups.append((prompt, grp_completions))

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
