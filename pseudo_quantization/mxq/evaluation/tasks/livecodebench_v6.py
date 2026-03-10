import json
from typing import Any

import numpy as np
from lighteval.metrics.metrics import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.extended.lcb.codegen_metrics import (
    codegen_metrics,
    extract_code,
    translate_private_test_cases,
)
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig


def prepare_prompt(line: dict[str, Any]) -> str:
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    return query


def lcb_codegeneration_prompt_fn(line, task_name: str = "lcb:codegeneration_v6") -> Doc:
    query = prepare_prompt(line)
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(line["metadata"]).get("func_name", None),
        },
    )


def codegen_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    generated_code_snippets = [[extract_code(pred) for pred in predictions]]
    evaluation_sample = {
        "inputs": formatted_doc.specific["inputs"],
        "outputs": formatted_doc.specific["outputs"],
        "fn_name": formatted_doc.specific["fn_name"],
    }
    evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

    metrics, _ = codegen_metrics(
        evaluation_sample,
        generated_code_snippets,
        k_list=[1],
        num_process_evaluate=8,
    )
    return metrics["pass@1"]


lcb_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:1",
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    higher_is_better=True,
    sample_level_fn=codegen_metric,
    corpus_level_fn=np.mean,
)


TASKS_TABLE = [
    LightevalTaskConfig(
        name="lcb:codegeneration_v6",
        suite=["custom"],
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="livecodebench/code_generation_lite",
        hf_subset="v6",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=32768,
        metric=[lcb_codegen_metric],
        stop_sequence=[],
        trust_dataset=True,
        version=0,
    )
]
