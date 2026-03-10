import json
import re

import numpy as np
from lighteval.metrics.metrics import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


def _normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return re.sub(r"\s+", " ", x).strip().lower()
    try:
        return re.sub(r"\s+", " ", json.dumps(x, sort_keys=True)).strip().lower()
    except Exception:
        return str(x).strip().lower()


def _extract_turn_text(row):
    if "prompt" in row and isinstance(row["prompt"], str):
        return row["prompt"]
    turns = row.get("turns", None)
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, dict):
            for k in ["content", "text", "utterance", "user"]:
                if k in first and isinstance(first[k], str):
                    return first[k]
        if isinstance(first, str):
            return first
    for k in ["question", "query", "input"]:
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def _extract_tools(row):
    for k in ["tools", "functions", "function", "tool_defs", "available_functions"]:
        if k in row:
            return row[k]
    return None


def _extract_gold(row):
    for k in ["ground_truth", "answer", "target", "gold"]:
        if k in row:
            return row[k]
    return ""


def bfcl_v3_prompt_fn(line, task_name: str = None):
    user_text = _extract_turn_text(line)
    tools = _extract_tools(line)
    gold = _extract_gold(line)

    tool_block = ""
    if tools is not None:
        tool_block = "\n\nAvailable tools/functions:\n" + json.dumps(
            tools, ensure_ascii=False
        )

    query = (
        "You are an assistant that must return a single function call in JSON only. "
        "Do not add explanations.\n\nUser request:\n"
        + user_text
        + tool_block
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=[_normalize_text(gold)],
        gold_index=0,
        specific={"gold": gold},
    )


def bfcl_v3_exec_proxy_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    pred = predictions[0] if predictions else ""
    gold = (formatted_doc.specific or {}).get("gold", "")
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


bfcl_v3_metric = SampleLevelMetric(
    metric_name="bfcl_v3_exec_proxy_em",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=bfcl_v3_exec_proxy_metric,
    corpus_level_fn=np.mean,
)


bfcl_v3 = LightevalTaskConfig(
    name="bfcl_v3",
    suite=["custom"],
    prompt_function=bfcl_v3_prompt_fn,
    hf_repo="llamastack/bfcl_v3",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=2048,
    metric=[bfcl_v3_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=1,
)


TASKS_TABLE = [bfcl_v3]
