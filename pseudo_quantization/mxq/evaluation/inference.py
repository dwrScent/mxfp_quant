from mxq.evaluation import patch, quant_method
import os
import json
import random
import argparse
import glob
import shutil

import torch

from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.endpoints.openai_model import OpenAIModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
from mxq.evaluation.main_vllm import vllm


def parser_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="whether to re-evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to save inference results."
    )
    # model
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model to load.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="dtype to use")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow execution of custom model code from Hub repos.",
    )
    # quantization
    parser.add_argument(
        "--quant_method", type=str, choices=["mxfp", "m2xfp"], default="m2xfp"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers", "openai"],
        default="vllm",
        help="Inference backend. Use openai for omni-infer OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL (used when backend=openai).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for OpenAI-compatible endpoint (used when backend=openai).",
    )
    parser.add_argument(
        "--served_model_name",
        type=str,
        default=os.getenv("OPENAI_MODEL_NAME", "openpangu_r_72b_2512"),
        help="Served model name for OpenAI-compatible endpoint (used when backend=openai).",
    )
    # dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="AIME-2024",
        choices=[
            "AIME-2024",
            "AIME-2025",
            "AIME-90",
            "MATH-500",
            "NuminaMath-1.5",
            "GSM8K",
            "GPQA-Diamond",
            "MMLU-PRO",
            "LiveCodeBench",
            "LiveCodeBench-V6",
            "SuperGPQA",
            "IF-Eval",
            "BFCL-V3",
        ],
        help="Dataset to load.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max #samples (for debug)"
    )
    # generation
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Generation temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Generation top_p")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens to generate per output sequence.",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=None,
        help="Maximum model input length. If unset, infer from model config.",
    )
    args = parser.parse_args()

    # force float16 for gptqmodel inference
    if "gptqmodel" in args.model:
        args.dtype = "float16"

    # output path
    args.model_name = args.model.strip("/").replace("/", "_")
    output_dir = (
        os.path.join(
            "./outputs", args.quant_method, f"{args.model_name}-seed{args.seed}"
        )
        if args.output_dir is None
        else args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)
    args.output_path = os.path.join(output_dir, f"{args.dataset}.jsonl")

    # Distributed settings
    args.tensor_parallel_size = torch.cuda.device_count()

    return args


def _ensure_pangu_modeling(model_name_or_path: str) -> None:
    """Inject missing modeling_pangu_moe.py for openPangu-R repos."""
    if "openPangu-R-72B-2512" not in model_name_or_path:
        return

    src = os.path.join(os.path.dirname(__file__), "modeling_pangu_moe.py")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing local helper file: {src}")

    if "/" not in model_name_or_path:
        return

    repo_id = model_name_or_path.strip("/")
    repo_fs = repo_id.replace("/", "--")
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_root = os.path.join(hf_home, "hub", f"models--{repo_fs}")
    modules_root = os.path.join(hf_home, "modules", "transformers_modules")

    copied_paths = []
    # 1) Inject into HF snapshots cache.
    for snap_dir in glob.glob(os.path.join(hub_root, "snapshots", "*")):
        if not os.path.isdir(snap_dir):
            continue
        dst = os.path.join(snap_dir, "modeling_pangu_moe.py")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        for pycache in glob.glob(os.path.join(snap_dir, "__pycache__", "modeling_pangu_moe*.pyc")):
            os.remove(pycache)
        copied_paths.append(dst)

    # 2) Inject into transformers dynamic modules cache.
    repo_path_nested = os.path.join(modules_root, *repo_id.split("/"))
    repo_path_flat = os.path.join(modules_root, repo_fs)
    for base in [repo_path_nested, repo_path_flat]:
        for mod_dir in glob.glob(os.path.join(base, "*")):
            if not os.path.isdir(mod_dir):
                continue
            dst = os.path.join(mod_dir, "modeling_pangu_moe.py")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            for pycache in glob.glob(os.path.join(mod_dir, "__pycache__", "modeling_pangu_moe*.pyc")):
                os.remove(pycache)
            copied_paths.append(dst)

    if copied_paths:
        print(f"[pangu] injected modeling code into {len(copied_paths)} cache path(s).")
    else:
        print(
            "[pangu] no local cache snapshot found yet; once model files are cached, "
            "re-running this command will inject modeling code automatically."
        )


def transformers_eval(
    model_config,
    tasks,
    custom_tasks,
    use_chat_template,
    max_samples,
):
    token = os.getenv("HF_TOKEN")
    cache_dir = os.getenv("HF_HOME", "/scratch")
    env_config = EnvConfig(token=token, cache_dir=cache_dir)

    evaluation_tracker = EvaluationTracker(
        output_dir="results",
        save_details=False,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        env_config=env_config,
        dataset_loading_processes=1,
        custom_tasks_directory=custom_tasks,
        override_batch_size=-1,
        num_fewshot_seeds=1,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=None,
        load_responses_from_details_date_id=None,
    )
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options={},
    )
    pipeline.evaluate()
    pipeline.show_results()
    return pipeline.get_results(), evaluation_tracker.details


def openai_eval(
    model_config,
    tasks,
    custom_tasks,
    use_chat_template,
    max_samples,
):
    token = os.getenv("HF_TOKEN")
    cache_dir = os.getenv("HF_HOME", "/scratch")
    env_config = EnvConfig(token=token, cache_dir=cache_dir)

    evaluation_tracker = EvaluationTracker(
        output_dir="results",
        save_details=False,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.OPENAI,
        env_config=env_config,
        dataset_loading_processes=1,
        custom_tasks_directory=custom_tasks,
        override_batch_size=-1,
        num_fewshot_seeds=1,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=None,
        load_responses_from_details_date_id=None,
    )
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options={},
    )
    pipeline.evaluate()
    pipeline.show_results()
    return pipeline.get_results(), evaluation_tracker.details


def main(args):
    if not args.debug and not args.overwrite and os.path.exists(args.output_path):
        print(f"Evaluation results found at {args.output_path}. Skip evaluation")
        return

    random.seed(args.seed)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["QUANT_METHOD"] = args.quant_method

    generation_parameters = GenerationParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=30 if "QwQ" in args.model else None,  # TODO. enable top_k only for QwQ?
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    task_map = {
        "AIME-2024": ("custom|aime24|0|0", "mxq.evaluation.tasks.reasoning"),
        "AIME-2025": ("custom|aime25|0|0", "mxq.evaluation.tasks.reasoning"),
        "AIME-90": ("custom|aime90|0|0", "mxq.evaluation.tasks.reasoning"),
        "MATH-500": ("custom|math_500|0|0", "mxq.evaluation.tasks.reasoning"),
        "NuminaMath-1.5": ("custom|numina_math|0|0", "mxq.evaluation.tasks.reasoning"),
        "GSM8K": ("custom|gsm8k|0|0", "mxq.evaluation.tasks.reasoning"),
        "GPQA-Diamond": ("custom|gpqa:diamond|0|0", "mxq.evaluation.tasks.reasoning"),
        "MMLU-PRO": ("custom|mmlu_pro|0|0", "mxq.evaluation.tasks.reasoning"),
        # Keep legacy name for compatibility, but evaluate on v6 by default.
        "LiveCodeBench": ("custom|lcb:codegeneration_v6|0|0", "mxq.evaluation.tasks.livecodebench_v6"),
        "LiveCodeBench-V6": ("custom|lcb:codegeneration_v6|0|0", "mxq.evaluation.tasks.livecodebench_v6"),
        "SuperGPQA": ("custom|supergpqa|0|0", "mxq.evaluation.tasks.supergpqa"),
        "IF-Eval": ("custom|ifeval|0|0", "mxq.evaluation.tasks.ifeval"),
        "BFCL-V3": ("custom|bfcl_v3|0|0", "mxq.evaluation.tasks.bfcl_v3"),
    }
    tasks, custom_tasks = task_map[args.dataset]
    task_kwargs = {"tasks": tasks, "custom_tasks": custom_tasks}
    if args.backend == "vllm":
        model_config = VLLMModelConfig(
            pretrained=args.model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_model_length=args.max_model_length,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            generation_parameters=generation_parameters,
        )
        try:
            results, details = vllm(
                model_config=model_config,
                use_chat_template=True,
                # output_dir="./outputs/lighteval_outputs",
                max_samples=args.max_samples,
                **task_kwargs,
            )
        except ValueError as e:
            if "are not supported for now" in str(e):
                raise ValueError(
                    f"{e}\nModel is unsupported by vLLM. Re-run with `--backend transformers`."
                ) from e
            raise
    elif args.backend == "transformers":
        if args.trust_remote_code:
            _ensure_pangu_modeling(args.model)
        model_config = TransformersModelConfig(
            pretrained=args.model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_length=args.max_model_length,
            generation_parameters=generation_parameters,
            use_chat_template=True,
        )
        try:
            results, details = transformers_eval(
                model_config=model_config,
                use_chat_template=True,
                max_samples=args.max_samples,
                **task_kwargs,
            )
        except OSError as e:
            if "modeling_pangu_moe.py" in str(e):
                raise OSError(
                    f"{e}\n`{args.model}` repo is missing required modeling code for HF AutoModel."
                    " This model currently needs its official omni-infer deployment path "
                    "(see repo docs) or a different HF-compatible model."
                ) from e
            raise
    else:
        model_config = OpenAIModelConfig(
            model=args.served_model_name,
            base_url=args.api_base_url,
            api_key=args.api_key,
            generation_parameters=generation_parameters,
        )
        results, details = openai_eval(
            model_config=model_config,
            use_chat_template=False,
            max_samples=args.max_samples,
            **task_kwargs,
        )

    # save evaluation results
    eval_results = []
    task_name = list(details.keys())[0]
    for detail in details[task_name]:
        eval_results.append(
            {
                "full_prompt": detail["full_prompt"],
                "generated_text": detail["predictions"][0],
                "gold": detail["gold"],
                "metrics": detail["metrics"],
            }
        )
    with open(args.output_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved at {args.output_path}.")


if __name__ == "__main__":
    args = parser_gen()
    main(args)
