import re

import numpy as np
from lighteval.metrics.metrics import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


def ifeval_prompt_fn(line, task_name: str = None):
    prompt = line.get("prompt", "")
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[""],
        gold_index=0,
        instruction=prompt,
        specific={"prompt": prompt},
    )


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])


def ifeval_proxy_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    # Proxy metric for IF-Eval when official instruction checker is not wired.
    # It handles several common explicit constraints from natural language prompts.
    pred = (predictions[0] if predictions else "").strip()
    prompt = (formatted_doc.specific or {}).get("prompt", "")
    prompt_l = prompt.lower()

    checks = []

    # At least / at most N words
    for m in re.finditer(r"at least (\d+) words?", prompt_l):
        checks.append(_count_words(pred) >= int(m.group(1)))
    for m in re.finditer(r"at most (\d+) words?", prompt_l):
        checks.append(_count_words(pred) <= int(m.group(1)))

    # Exactly N sentences
    for m in re.finditer(r"exactly (\d+) sentences?", prompt_l):
        checks.append(_count_sentences(pred) == int(m.group(1)))

    # Must include keyword/phrase
    for m in re.finditer(r"(?:include|contain) (?:the )?(?:word|phrase) ['\"]([^'\"]+)['\"]", prompt_l):
        checks.append(m.group(1).lower() in pred.lower())

    # Do not use
    if "do not use commas" in prompt_l or "without commas" in prompt_l:
        checks.append("," not in pred)
    if "do not use the letter e" in prompt_l:
        checks.append("e" not in pred.lower())

    # If no explicit rule matched, use non-empty response as minimal validity.
    if not checks:
        return 1.0 if pred else 0.0
    return 1.0 if all(checks) else 0.0


ifeval_metric = SampleLevelMetric(
    metric_name="ifeval_proxy_strict",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.INSTRUCTION_FOLLOWING,
    higher_is_better=True,
    sample_level_fn=ifeval_proxy_metric,
    corpus_level_fn=np.mean,
)


ifeval = LightevalTaskConfig(
    name="ifeval",
    suite=["custom"],
    prompt_function=ifeval_prompt_fn,
    hf_repo="google/IFEval",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metric=[ifeval_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=1,
)


TASKS_TABLE = [ifeval]
