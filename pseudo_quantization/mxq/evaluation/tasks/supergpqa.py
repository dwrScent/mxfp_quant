import random

from lighteval.metrics.dynamic_metrics import (
    IndicesExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


supergpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[
        IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
    ],
    pred_extraction_target=[
        IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
    ],
    precision=5,
)


def supergpqa_prompt_fn(line, task_name: str = None):
    options = line["options"]
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("SuperGPQA options should be a list with at least 2 choices")

    labels = [chr(ord("A") + i) for i in range(len(options))]

    if "answer_letter" in line and line["answer_letter"] in labels:
        gold_index = labels.index(line["answer_letter"])
    elif "answer" in line and line["answer"] in options:
        gold_index = options.index(line["answer"])
    else:
        # Fallback for noisy rows
        gold_index = random.randint(0, len(options) - 1)

    options_str = "\n".join(
        f"{label}) {choice}" for label, choice in zip(labels, options)
    )
    query = (
        "Answer the following multiple choice question. Think step by step before answering. "
        "The last line of your response must be exactly: 'Answer: $LETTER' "
        "(without quotes), where LETTER is one of "
        + "".join(labels)
        + ".\n\n"
        + line["question"]
        + "\n\n"
        + options_str
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=labels,
        gold_index=gold_index,
        instruction=query,
    )


supergpqa = LightevalTaskConfig(
    name="supergpqa",
    suite=["custom"],
    prompt_function=supergpqa_prompt_fn,
    hf_repo="m-a-p/SuperGPQA",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[supergpqa_metric],
    stop_sequence=[],
    trust_dataset=True,
    version=1,
)


TASKS_TABLE = [supergpqa]
