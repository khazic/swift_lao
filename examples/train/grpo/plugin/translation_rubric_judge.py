"""Translation rubric LLM-judge reward for GRPO.

Scores each completion against a fixed rubric with several discrete criteria,
sums the per-criterion scores, and normalizes to [0, 1]. Compared with a
single 0-100 score, the rubric gives the judge crisper per-criterion
decisions and a more discriminative aggregate reward, which helps avoid the
"judge regresses everything to 80-90" failure mode of scalar scoring.

Aggregate range: 0..MAX_SCORE, normalized to [0, 1]. With the default rubric
(2+2+2+2+1+1=10) the reward lives on an 11-level discrete grid.

Required env vars:
    TRANSLATION_RUBRIC_JUDGE_API_BASE        OpenAI-compatible base URL.
    TRANSLATION_RUBRIC_JUDGE_API_KEY         Bearer token for the judge.
    TRANSLATION_RUBRIC_JUDGE_MODEL           Model name to send.

Optional env vars:
    TRANSLATION_RUBRIC_JUDGE_TIMEOUT         Seconds per call, default 180.
    TRANSLATION_RUBRIC_JUDGE_TEMPERATURE     Sampling temperature, default 0.0.
    TRANSLATION_RUBRIC_JUDGE_MAX_RETRIES     Retry budget per call, default 2.
    TRANSLATION_RUBRIC_JUDGE_FALLBACK_REWARD Reward when the judge fails,
                                              default 0.0.
    TRANSLATION_RUBRIC_JUDGE_MAX_CONCURRENCY Max in-flight calls, default 32.

Register name: ``translation_rubric_judge``. Use via
    --external_plugins examples/train/grpo/plugin/translation_rubric_judge.py \
    --reward_funcs translation_rubric_judge
"""

import asyncio
import json
import os
import re
import uuid
from typing import List, Optional, Tuple

import aiohttp

from swift.rewards import AsyncORM, orms
from swift.utils import get_logger

logger = get_logger()


# Rubric definition: (criterion_key, max_score). Adjust max values to
# re-weight criteria; the aggregate is sum(scores) / sum(max_scores).
RUBRIC_ITEMS: List[Tuple[str, int]] = [
    ('completeness', 2),
    ('accuracy', 2),
    ('grammar', 2),
    ('fluency', 2),
    ('terminology', 1),
    ('pure_translation', 1),
]
MAX_SCORE: int = sum(max_v for _, max_v in RUBRIC_ITEMS)


JUDGE_PROMPT_TEMPLATE = """\
## 任务：你是一名翻译质量诊断专家。按 rubric 给一条翻译结果逐项打分。

## input 结构：
- input.prompt：原文及翻译要求
- input.answer：模型给出的翻译结果

## 特殊规则：源语言与目标语言一致时
1. 如果 prompt 要求翻译为某语言（例如要求翻译为中文），而待翻译的源文本本身就是该语言（例如源文本就是中文），那么模型逐字原样输出源文本即视为完美翻译。
2. 这种情况下六个维度全部按满分计：completeness=2, accuracy=2, grammar=2, fluency=2, terminology=1, pure_translation=1（总分 10/10）。
3. 若模型在此场景下没有逐字输出而是做了改写、翻译或添加解释，按改写后的实际质量打分，不享受满分待遇。

## 通用判官指令
1. 即使原文语义模糊、含糊不清或难以判断，也必须按模型输出的实际表现给出分数，禁止拒绝评分或返回空字段。
2. 评分时综合考虑错误的类别和严重程度：重大语义错误显著扣分，细微瑕疵小幅扣分。

## 评分项（每项独立判断，相互不影响）：

1. completeness（完整性）：翻译是否覆盖原文全部信息？
   - 0 = 大量遗漏，或大量无关添加
   - 1 = 少量遗漏或添加，不影响核心意思
   - 2 = 完整覆盖，无多余内容

2. accuracy（准确性）：核心语义是否被正确传达？
   - 0 = 关键内容误译，整体意思偏差大
   - 1 = 整体正确，细节有错或歧义
   - 2 = 语义完全忠实

3. grammar（语法）：目标语言的语法、拼写、字符编码是否正确？
   - 0 = 多处语法错误，或出现乱码
   - 1 = 个别小瑕疵
   - 2 = 语法正确，无乱码

4. fluency（流畅性与风格）：读起来是否自然、地道，标点和语域是否得当？
   - 0 = 别扭、生硬、机翻味重，或标点/语域明显不合适
   - 1 = 基本通顺，但有局部不自然或语域偏差
   - 2 = 自然流畅，标点和语域贴合目标语言习惯

5. terminology（术语）：专有名词、习语、行业术语是否准确一致？
   - 0 = 术语错译或前后不一致
   - 1 = 术语正确

6. pure_translation（纯翻译输出）：输出是否只是翻译本身？
   - 0 = 含有模型解释、评论、Markdown 标题、拒绝、反问等额外内容
   - 1 = 仅翻译内容

## 输出格式（严格 JSON，仅输出以下结构；不要任何额外文字、不要代码块包裹）：
{"rubric": {"completeness": 2, "accuracy": 2, "grammar": 2, "fluency": 2, "terminology": 1, "pure_translation": 1}}

## 输出要求：
1. 严格按上述 JSON 输出，不输出任何解释或思考过程。
2. 每项必须出现且为整数，取值范围如上。
3. 不输出 rubric 以外的字段。

## input：__JUDGE_INPUT_PAYLOAD__
"""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f'TranslationRubricJudgeReward requires env var {name}. '
            'See examples/train/grpo/plugin/translation_rubric_judge.py docstring.')
    return value


class TranslationRubricJudgeReward(AsyncORM):

    def __init__(self, args, **kwargs):
        super().__init__(args)
        base = _require_env('TRANSLATION_RUBRIC_JUDGE_API_BASE').rstrip('/')
        if not base.endswith('/v1'):
            base = base + '/v1'
        self.api_base = base
        self.api_key = _require_env('TRANSLATION_RUBRIC_JUDGE_API_KEY')
        self.model_name = _require_env('TRANSLATION_RUBRIC_JUDGE_MODEL')
        self.timeout = float(os.environ.get('TRANSLATION_RUBRIC_JUDGE_TIMEOUT', '180'))
        self.temperature = float(os.environ.get('TRANSLATION_RUBRIC_JUDGE_TEMPERATURE', '0.0'))
        self.max_retries = int(os.environ.get('TRANSLATION_RUBRIC_JUDGE_MAX_RETRIES', '2'))
        self.fallback_reward = float(os.environ.get('TRANSLATION_RUBRIC_JUDGE_FALLBACK_REWARD', '0.0'))
        self.max_concurrency = int(os.environ.get('TRANSLATION_RUBRIC_JUDGE_MAX_CONCURRENCY', '32'))
        self._semaphore: Optional[asyncio.Semaphore] = None
        logger.info(
            f'TranslationRubricJudgeReward initialized | base={self.api_base} | '
            f'model={self.model_name} | rubric_max={MAX_SCORE} | '
            f'timeout={self.timeout}s | max_concurrency={self.max_concurrency}')

    @staticmethod
    def _last_user_content(msg_list) -> str:
        for msg in reversed(msg_list):
            if msg.get('role') == 'user':
                return msg.get('content', '') or ''
        return ''

    @staticmethod
    def _strip_chat_artifacts(text: str) -> str:
        # The SFT base uses a harmony-style chat template whose end-of-turn
        # token decodes to the literal string "<turn|>". Without stripping
        # it (and any trailing channel/turn fragments) the judge sees the
        # tokens as non-translation content and dings pure_translation.
        # Truncate at the first end-of-turn marker, then drop any stray
        # template markers that leak through.
        cut = re.split(r'<turn\|>|<eos>', text, maxsplit=1)[0]
        cut = re.sub(
            r'<\|(?:turn|channel|think|tool_call|tool|tool_response|audio|image)>'
            r'|<(?:channel|tool_call|tool|tool_response|audio|image)\|>'
            r'|<(?:bos|eos|pad|unk|mask)>',
            '',
            cut,
        )
        return cut.strip()

    @staticmethod
    def _build_input_payload(prompt: str, completion: str) -> str:
        payload = {
            'case_id': uuid.uuid4().hex[:12],
            'prompt': prompt,
            'answer': completion,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _parse_rubric(raw: str) -> Optional[float]:
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
        # Accept either {"rubric": {...}} or the flat dict, since some judges
        # drop the outer key when only one field is required.
        rubric = obj.get('rubric') if isinstance(obj.get('rubric'), dict) else obj

        total = 0.0
        present = False
        for key, max_v in RUBRIC_ITEMS:
            v = rubric.get(key)
            try:
                s = float(v)
            except (TypeError, ValueError):
                # Missing or unparseable criterion is treated as 0; this
                # incentivizes the judge to fill in every field.
                continue
            present = True
            s = max(0.0, min(float(max_v), s))
            total += s
        if not present:
            return None
        return total / float(MAX_SCORE)

    async def _score_one(self, session: aiohttp.ClientSession, prompt: str,
                         completion: str) -> float:
        completion = self._strip_chat_artifacts(completion)
        user_input = self._build_input_payload(prompt, completion)
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
                            logger.warning(f'rubric judge attempt {attempt} failed: {last_err}')
                            continue
                        data = await resp.json()
                content = data['choices'][0]['message']['content']
                reward = self._parse_rubric(content)
                if reward is None:
                    last_err = f'unparseable rubric response: {content[:300]!r}'
                    logger.warning(last_err)
                    continue
                return reward
            except asyncio.TimeoutError:
                last_err = 'timeout'
                logger.warning('rubric judge call timed out')
            except Exception as e:
                last_err = str(e)
                logger.warning(f'rubric judge call error: {e}')

        logger.warning(
            f'rubric judge failed after {self.max_retries + 1} attempts: {last_err}')
        return self.fallback_reward

    async def __call__(self, completions: List[str], messages, **kwargs) -> List[float]:
        n = len(completions)
        if n == 0:
            return []
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

        prompts = [self._last_user_content(messages[i]) for i in range(n)]

        connector = aiohttp.TCPConnector(limit=self.max_concurrency)
        # trust_env=True so aiohttp honors HTTP_PROXY / HTTPS_PROXY / NO_PROXY.
        async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
            tasks = [self._score_one(session, p, c) for p, c in zip(prompts, completions)]
            rewards = await asyncio.gather(*tasks)

        if len(rewards) != n:
            logger.error(f'rubric reward count mismatch: expected {n}, got {len(rewards)}')
            rewards = (list(rewards) + [self.fallback_reward] * n)[:n]
        return list(rewards)


orms['translation_rubric_judge'] = TranslationRubricJudgeReward
