from mxq.evaluation import patch, quant_method
import os
import json
import random
import argparse

import torch

from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
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
    # quantization
    parser.add_argument("--quant_method", type=str, choices=["mxfp", "m2xfp"])
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
        default=32768,
        help="Maximum model input length.",
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
    model_config = VLLMModelConfig(
        pretrained=args.model,
        dtype=args.dtype,
        max_model_length=args.max_model_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        generation_parameters=generation_parameters,
    )

    if args.dataset == "AIME-2024":
        task_kwargs = {
            "tasks": "custom|aime24|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "AIME-2025":
        task_kwargs = {
            "tasks": "custom|aime25|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "AIME-90":
        task_kwargs = {
            "tasks": "custom|aime90|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "MATH-500":
        task_kwargs = {
            "tasks": "custom|math_500|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "NuminaMath-1.5":
        task_kwargs = {
            "tasks": "custom|numina_math|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "GSM8K":
        task_kwargs = {
            "tasks": "custom|gsm8k|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "GPQA-Diamond":
        task_kwargs = {
            "tasks": "custom|gpqa:diamond|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "MMLU-PRO":
        task_kwargs = {
            "tasks": "custom|mmlu_pro|0|0",
            "custom_tasks": "mxq.evaluation.tasks.reasoning",
        }
    elif args.dataset == "LiveCodeBench":
        task_kwargs = {
            "tasks": "custom|lcb:codegeneration|0|0",
            "custom_tasks": "mxq.evaluation.tasks.livecodebench",
        }

    results, details = vllm(
        model_config=model_config,
        use_chat_template=True,
        # output_dir="./outputs/lighteval_outputs",
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
#
# import os
# import json
# import random
# import argparse
# import shutil
# import re
#
# import torch
#
# from lighteval.models.model_input import GenerationParameters
# from lighteval.models.vllm.vllm_model import VLLMModelConfig
# from lighteval.models.transformers.transformers_model import TransformersModelConfig
# from lighteval.models.endpoints.openai_model import OpenAIModelConfig
# from lighteval.logging.evaluation_tracker import EvaluationTracker
# from lighteval.pipeline import EnvConfig, ParallelismManager, Pipeline, PipelineParameters
# from mxq.evaluation.main_vllm import vllm
#
#
# def _load_local_config_json(model_name_or_path: str) -> dict:
#     """Best-effort read of local config.json; returns empty dict when unavailable."""
#     if not model_name_or_path or not os.path.isdir(model_name_or_path):
#         return {}
#     config_path = os.path.join(model_name_or_path, "config.json")
#     if not os.path.isfile(config_path):
#         return {}
#     try:
#         with open(config_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return {}
#
#
# def _is_openpangu_model(model_name_or_path: str) -> bool:
#     """Detect openPangu models for compatibility workarounds.
#
#     Supports both hub repo ids and local snapshot paths.
#     """
#     model_str = str(model_name_or_path or "")
#     model_lower = model_str.lower()
#
#     # Fast path for common hub ids / local path names.
#     if "openpangu-r-72b-2512" in model_lower or "openpangu" in model_lower:
#         return True
#
#     # Local snapshot path may not contain model name. Inspect config metadata.
#     config = _load_local_config_json(model_str)
#     if not config:
#         return False
#
#     model_type = str(config.get("model_type", "")).lower()
#     if "pangu" in model_type:
#         return True
#
#     architectures = [str(x).lower() for x in config.get("architectures", [])]
#     if any("pangu" in arch for arch in architectures):
#         return True
#
#     auto_map = config.get("auto_map", {})
#     if isinstance(auto_map, dict) and any("pangu" in str(v).lower() for v in auto_map.values()):
#         return True
#
#     return False
#
#
# def parser_gen():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument(
#         "--overwrite", action="store_true", help="whether to re-evaluate"
#     )
#     parser.add_argument(
#         "--output_dir", type=str, default=None, help="Path to save inference results."
#     )
#     # model
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#         help="Model to load.",
#     )
#     parser.add_argument("--dtype", type=str, default="bfloat16", help="dtype to use")
#     parser.add_argument(
#         "--trust_remote_code",
#         action="store_true",
#         help="Allow execution of custom model code from Hub repos.",
#     )
#     # quantization
#     parser.add_argument(
#         "--quant_method", type=str, choices=["mxfp", "m2xfp"], default="m2xfp"
#     )
#     parser.add_argument(
#         "--backend",
#         type=str,
#         choices=["vllm", "transformers", "openai"],
#         default="vllm",
#         help="Inference backend. Use openai for omni-infer OpenAI-compatible API.",
#     )
#     parser.add_argument(
#         "--api_base_url",
#         type=str,
#         default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
#         help="OpenAI-compatible API base URL (used when backend=openai).",
#     )
#     parser.add_argument(
#         "--api_key",
#         type=str,
#         default=os.getenv("OPENAI_API_KEY", "EMPTY"),
#         help="API key for OpenAI-compatible endpoint (used when backend=openai).",
#     )
#     parser.add_argument(
#         "--served_model_name",
#         type=str,
#         default=os.getenv("OPENAI_MODEL_NAME", "openpangu_r_72b_2512"),
#         help="Served model name for OpenAI-compatible endpoint (used when backend=openai).",
#     )
#     # dataset
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default="AIME-2024",
#         choices=[
#             "AIME-2024",
#             "AIME-2025",
#             "AIME-90",
#             "MATH-500",
#             "NuminaMath-1.5",
#             "GSM8K",
#             "GPQA-Diamond",
#             "MMLU-PRO",
#             "LiveCodeBench",
#             "LiveCodeBench-V6",
#             "SuperGPQA",
#             "IF-Eval",
#             "BFCL-V3",
#         ],
#         help="Dataset to load.",
#     )
#     parser.add_argument(
#         "--max_samples", type=int, default=None, help="Max #samples (for debug)"
#     )
#     # generation
#     parser.add_argument(
#         "--temperature", type=float, default=0.6, help="Generation temperature"
#     )
#     parser.add_argument("--top_p", type=float, default=0.95, help="Generation top_p")
#     parser.add_argument("--seed", type=int, default=42, help="Generation seed")
#     parser.add_argument(
#         "--max_new_tokens",
#         type=int,
#         default=1024,
#         help="Maximum number of tokens to generate per output sequence.",
#     )
#     parser.add_argument(
#         "--max_model_length",
#         type=int,
#         default=None,
#         help="Maximum model input length. If unset, infer from model config.",
#     )
#     parser.add_argument(
#         "--model_parallel",
#         action="store_true",
#         help="Enable transformers model-parallel (requires accelerate multi-process launch).",
#     )
#     args = parser.parse_args()
#
#     # force float16 for gptqmodel inference
#     if "gptqmodel" in args.model:
#         args.dtype = "float16"
#
#     # output path
#     args.model_name = args.model.strip("/").replace("/", "_")
#     output_dir = (
#         os.path.join(
#             "./outputs", args.quant_method, f"{args.model_name}-seed{args.seed}"
#         )
#         if args.output_dir is None
#         else args.output_dir
#     )
#     os.makedirs(output_dir, exist_ok=True)
#     args.output_path = os.path.join(output_dir, f"{args.dataset}.jsonl")
#
#     # Distributed settings
#     args.tensor_parallel_size = torch.cuda.device_count()
#
#     return args
#
#
# def _find_local_hf_snapshot(repo_id: str) -> str | None:
#     """Find local HF snapshot path only; never trigger network download."""
#     repo_fs = repo_id.replace("/", "--")
#     hf_homes = [
#         os.getenv("HF_HOME"),
#         "/cephfs/shared/xyli/hf_cache",
#         os.path.expanduser("~/.cache/huggingface"),
#     ]
#
#     seen = set()
#     for hf_home in hf_homes:
#         if not hf_home or hf_home in seen:
#             continue
#         seen.add(hf_home)
#         hub_root = os.path.join(hf_home, "hub", f"models--{repo_fs}")
#         snapshots_root = os.path.join(hub_root, "snapshots")
#         if not os.path.isdir(snapshots_root):
#             continue
#
#         ref_main = os.path.join(hub_root, "refs", "main")
#         if os.path.isfile(ref_main):
#             with open(ref_main, "r") as f:
#                 sha = f.read().strip()
#             snapshot = os.path.join(snapshots_root, sha)
#             if sha and os.path.isdir(snapshot):
#                 return snapshot
#
#         snapshot_dirs = [
#             os.path.join(snapshots_root, d)
#             for d in os.listdir(snapshots_root)
#             if os.path.isdir(os.path.join(snapshots_root, d))
#         ]
#         if snapshot_dirs:
#             snapshot_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
#             return snapshot_dirs[0]
#     return None
#
#
# def _sync_pangu_dynamic_module(src: str, snapshot_dir: str) -> None:
#     """Keep HF dynamic module cache in sync with patched modeling file."""
#     sha = os.path.basename(os.path.normpath(snapshot_dir))
#     hf_homes = [
#         os.getenv("HF_HOME"),
#         "/cephfs/shared/xyli/hf_cache",
#         os.path.expanduser("~/.cache/huggingface"),
#     ]
#     seen = set()
#     for hf_home in hf_homes:
#         if not hf_home or hf_home in seen:
#             continue
#         seen.add(hf_home)
#         module_dir = os.path.join(
#             hf_home, "modules", "transformers_modules", sha
#         )
#         if not os.path.isdir(module_dir):
#             continue
#         dst = os.path.join(module_dir, "modeling_pangu_moe.py")
#         shutil.copy2(src, dst)
#         print(f"[pangu] synced dynamic module cache: {dst}")
#
#
# def _ensure_pangu_modeling(model_name_or_path: str) -> str:
#     """Inject missing modeling_pangu_moe.py for openPangu-R and return local load path."""
#     if not _is_openpangu_model(model_name_or_path):
#         return model_name_or_path
#
#     src = os.path.join(os.path.dirname(__file__), "modeling_pangu_moe.py")
#     if not os.path.exists(src):
#         raise FileNotFoundError(f"Missing local helper file: {src}")
#
#     if os.path.isdir(model_name_or_path):
#         dst = os.path.join(model_name_or_path, "modeling_pangu_moe.py")
#         shutil.copy2(src, dst)
#         print(f"[pangu] using local model dir and injected modeling code: {dst}")
#         _sync_pangu_dynamic_module(src, model_name_or_path)
#         return model_name_or_path
#
#     repo_id = model_name_or_path.strip("/")
#     if "/" not in repo_id:
#         return model_name_or_path
#
#     snapshot_dir = _find_local_hf_snapshot(repo_id)
#     if snapshot_dir is None:
#         raise FileNotFoundError(
#             "[pangu] local snapshot not found. "
#             "Expected cache under /cephfs/shared/xyli/hf_cache (or HF_HOME). "
#             "Please download weights first, then rerun."
#         )
#
#     dst = os.path.join(snapshot_dir, "modeling_pangu_moe.py")
#     shutil.copy2(src, dst)
#     print(f"[pangu] using local snapshot and injected modeling code: {dst}")
#     _sync_pangu_dynamic_module(src, snapshot_dir)
#     return snapshot_dir
#
#
# def _should_use_chat_template(model_name_or_path: str) -> bool:
#     """Return whether lighteval should apply tokenizer chat template.
#
#     For openPangu instruction models, default to chat template unless explicitly
#     disabled by env var PANGU_USE_CHAT_TEMPLATE=0.
#     """
#     if _is_openpangu_model(model_name_or_path):
#         flag = os.getenv("PANGU_USE_CHAT_TEMPLATE", "1").strip().lower()
#         return flag not in {"0", "false", "no", "off"}
#     return True
#
#
# def _patch_decode_keep_special_for_pangu(model_name_or_path: str) -> None:
#     """Avoid decoding empty strings when model emits special-only tokens."""
#     if not _is_openpangu_model(model_name_or_path):
#         return
#     from lighteval.models.abstract_model import LightevalModel
#
#     def _tok_decode_keep_special(self, tokens):
#         primary = self.tokenizer.batch_decode(
#             tokens, skip_special_tokens=True, errors="ignore"
#         )
#         fallback = self.tokenizer.batch_decode(
#             tokens, skip_special_tokens=False, errors="ignore"
#         )
#         decoded = []
#         for p, f in zip(primary, fallback):
#             text = (p or "").strip()
#             if text and text != "�":
#                 decoded.append(text)
#                 continue
#
#             # Fallback path: keep content but strip common control/sentinel markers.
#             alt = (f or "").strip()
#             alt = re.sub(r"\[unused\d+\]", "", alt)
#             alt = alt.replace("<unk>", "").replace("�", "").strip()
#             decoded.append(alt)
#         return decoded
#
#     LightevalModel.tok_decode = _tok_decode_keep_special
#     print("[pangu] patched tok_decode(primary skip_special_tokens=True + fallback)")
#
#
# def _patch_transformers_sampling_for_pangu(model_name_or_path: str) -> None:
#     """Force sampling for pangu on lighteval transformers backend.
#
#     Some openPangu checkpoints tend to emit EOS immediately with greedy decoding,
#     resulting in empty generations for evaluation prompts.
#     """
#     if not _is_openpangu_model(model_name_or_path):
#         return
#     from lighteval.models.transformers import transformers_model as tm
#
#     original_greedy_until = tm.TransformersModel.greedy_until
#     if getattr(original_greedy_until, "_pangu_sampling_patched", False):
#         return
#
#     def _greedy_until_with_sampling(self, requests, override_bs=None):
#         for req in requests:
#             req.do_sample = True
#         return original_greedy_until(self, requests, override_bs=override_bs)
#
#     _greedy_until_with_sampling._pangu_sampling_patched = True
#     tm.TransformersModel.greedy_until = _greedy_until_with_sampling
#     print("[pangu] patched transformers greedy_until(do_sample=True)")
#
#
# def transformers_eval(
#     model_config,
#     tasks,
#     custom_tasks,
#     use_chat_template,
#     max_samples,
# ):
#     token = os.getenv("HF_TOKEN")
#     cache_dir = os.getenv("HF_HOME", "/scratch")
#     env_config = EnvConfig(token=token, cache_dir=cache_dir)
#
#     evaluation_tracker = EvaluationTracker(
#         output_dir="results",
#         save_details=False,
#         push_to_hub=False,
#         push_to_tensorboard=False,
#         public=False,
#         hub_results_org=None,
#     )
#     pipeline_params = PipelineParameters(
#         launcher_type=ParallelismManager.NONE,
#         env_config=env_config,
#         dataset_loading_processes=1,
#         custom_tasks_directory=custom_tasks,
#         override_batch_size=-1,
#         num_fewshot_seeds=1,
#         max_samples=max_samples,
#         use_chat_template=use_chat_template,
#         system_prompt=None,
#         load_responses_from_details_date_id=None,
#     )
#     pipeline = Pipeline(
#         tasks=tasks,
#         pipeline_parameters=pipeline_params,
#         evaluation_tracker=evaluation_tracker,
#         model_config=model_config,
#         metric_options={},
#     )
#     pipeline.evaluate()
#     pipeline.show_results()
#     return pipeline.get_results(), evaluation_tracker.details
#
#
# def openai_eval(
#     model_config,
#     tasks,
#     custom_tasks,
#     use_chat_template,
#     max_samples,
# ):
#     token = os.getenv("HF_TOKEN")
#     cache_dir = os.getenv("HF_HOME", "/scratch")
#     env_config = EnvConfig(token=token, cache_dir=cache_dir)
#
#     evaluation_tracker = EvaluationTracker(
#         output_dir="results",
#         save_details=False,
#         push_to_hub=False,
#         push_to_tensorboard=False,
#         public=False,
#         hub_results_org=None,
#     )
#     pipeline_params = PipelineParameters(
#         launcher_type=ParallelismManager.OPENAI,
#         env_config=env_config,
#         dataset_loading_processes=1,
#         custom_tasks_directory=custom_tasks,
#         override_batch_size=-1,
#         num_fewshot_seeds=1,
#         max_samples=max_samples,
#         use_chat_template=use_chat_template,
#         system_prompt=None,
#         load_responses_from_details_date_id=None,
#     )
#     pipeline = Pipeline(
#         tasks=tasks,
#         pipeline_parameters=pipeline_params,
#         evaluation_tracker=evaluation_tracker,
#         model_config=model_config,
#         metric_options={},
#     )
#     pipeline.evaluate()
#     pipeline.show_results()
#     return pipeline.get_results(), evaluation_tracker.details
#
#
# def main(args):
#     if not args.debug and not args.overwrite and os.path.exists(args.output_path):
#         print(f"Evaluation results found at {args.output_path}. Skip evaluation")
#         return
#
#     random.seed(args.seed)
#     os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
#     os.environ["QUANT_METHOD"] = args.quant_method
#
#     pangu_model = _is_openpangu_model(args.model)
#     generation_parameters = GenerationParameters(
#         temperature=args.temperature,
#         top_p=args.top_p,
#         # Avoid invalid top_k defaults (e.g. -1 in some custom generation configs).
#         top_k=30 if ("QwQ" in args.model or pangu_model) else None,
#         max_new_tokens=args.max_new_tokens,
#         seed=args.seed,
#     )
#     _patch_decode_keep_special_for_pangu(args.model)
#     _patch_transformers_sampling_for_pangu(args.model)
#     use_chat_template = _should_use_chat_template(args.model)
#     print(f"[eval] use_chat_template={use_chat_template}")
#     task_map = {
#         "AIME-2024": ("custom|aime24|0|0", "mxq.evaluation.tasks.reasoning"),
#         "AIME-2025": ("custom|aime25|0|0", "mxq.evaluation.tasks.reasoning"),
#         "AIME-90": ("custom|aime90|0|0", "mxq.evaluation.tasks.reasoning"),
#         "MATH-500": ("custom|math_500|0|0", "mxq.evaluation.tasks.reasoning"),
#         "NuminaMath-1.5": ("custom|numina_math|0|0", "mxq.evaluation.tasks.reasoning"),
#         "GSM8K": ("custom|gsm8k|0|0", "mxq.evaluation.tasks.reasoning"),
#         "GPQA-Diamond": ("custom|gpqa:diamond|0|0", "mxq.evaluation.tasks.reasoning"),
#         "MMLU-PRO": ("custom|mmlu_pro|0|0", "mxq.evaluation.tasks.reasoning"),
#         # Keep legacy name for compatibility, but evaluate on v6 by default.
#         "LiveCodeBench": ("custom|lcb:codegeneration_v6|0|0", "mxq.evaluation.tasks.livecodebench_v6"),
#         "LiveCodeBench-V6": ("custom|lcb:codegeneration_v6|0|0", "mxq.evaluation.tasks.livecodebench_v6"),
#         "SuperGPQA": ("custom|supergpqa|0|0", "mxq.evaluation.tasks.supergpqa"),
#         "IF-Eval": ("custom|ifeval|0|0", "mxq.evaluation.tasks.ifeval"),
#         "BFCL-V3": ("custom|bfcl_v3|0|0", "mxq.evaluation.tasks.bfcl_v3"),
#     }
#     tasks, custom_tasks = task_map[args.dataset]
#     task_kwargs = {"tasks": tasks, "custom_tasks": custom_tasks}
#     if args.backend == "vllm":
#         from mxq.evaluation import patch, quant_method  # noqa: F401
#
#         model_config = VLLMModelConfig(
#             pretrained=args.model,
#             dtype=args.dtype,
#             trust_remote_code=args.trust_remote_code,
#             max_model_length=args.max_model_length,
#             tensor_parallel_size=args.tensor_parallel_size,
#             gpu_memory_utilization=0.9,
#             enforce_eager=True,
#             enable_prefix_caching=False,
#             enable_chunked_prefill=False,
#             generation_parameters=generation_parameters,
#         )
#         try:
#             results, details = vllm(
#                 model_config=model_config,
#                 use_chat_template=use_chat_template,
#                 # output_dir="./outputs/lighteval_outputs",
#                 max_samples=args.max_samples,
#                 **task_kwargs,
#             )
#         except ValueError as e:
#             if "are not supported for now" in str(e):
#                 raise ValueError(
#                     f"{e}\nModel is unsupported by vLLM. Re-run with `--backend transformers`."
#                 ) from e
#             raise
#     elif args.backend == "transformers":
#         if args.trust_remote_code:
#             args.model = _ensure_pangu_modeling(args.model)
#         model_parallel = args.model_parallel
#         local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
#         if model_parallel and local_world_size > 1:
#             print(
#                 "[warn] --model_parallel is only supported in single-process mode for this lighteval version. "
#                 f"Detected LOCAL_WORLD_SIZE={local_world_size}, falling back to model_parallel=False."
#             )
#             model_parallel = False
#         accelerator = None
#         if model_parallel:
#             try:
#                 from accelerate import Accelerator
#             except Exception as e:
#                 raise ImportError(
#                     "--model_parallel requires `accelerate` package. Install via `pip install accelerate`."
#                 ) from e
#             accelerator = Accelerator()
#         model_config = TransformersModelConfig(
#             pretrained=args.model,
#             dtype=args.dtype,
#             trust_remote_code=args.trust_remote_code,
#             max_length=args.max_model_length,
#             model_parallel=model_parallel,
#             accelerator=accelerator,
#             generation_parameters=generation_parameters,
#             use_chat_template=use_chat_template,
#         )
#         try:
#             results, details = transformers_eval(
#                 model_config=model_config,
#                 use_chat_template=use_chat_template,
#                 max_samples=args.max_samples,
#                 **task_kwargs,
#             )
#         except OSError as e:
#             if "modeling_pangu_moe.py" in str(e):
#                 raise OSError(
#                     f"{e}\n`{args.model}` repo is missing required modeling code for HF AutoModel."
#                     " This model currently needs its official omni-infer deployment path "
#                     "(see repo docs) or a different HF-compatible model."
#                 ) from e
#             raise
#     else:
#         model_config = OpenAIModelConfig(
#             model=args.served_model_name,
#             base_url=args.api_base_url,
#             api_key=args.api_key,
#             generation_parameters=generation_parameters,
#         )
#         results, details = openai_eval(
#             model_config=model_config,
#             use_chat_template=False,
#             max_samples=args.max_samples,
#             **task_kwargs,
#         )
#
#     # save evaluation results
#     eval_results = []
#     task_name = list(details.keys())[0]
#     for detail in details[task_name]:
#         eval_results.append(
#             {
#                 "full_prompt": detail["full_prompt"],
#                 "generated_text": detail["predictions"][0],
#                 "gold": detail["gold"],
#                 "metrics": detail["metrics"],
#             }
#         )
#     total = len(eval_results)
#     non_empty = sum(1 for x in eval_results if str(x.get("generated_text", "")).strip())
#     print(f"[eval] non_empty_generations={non_empty}/{total}")
#     with open(args.output_path, "w") as f:
#         json.dump(eval_results, f, indent=4)
#     print(f"Evaluation results saved at {args.output_path}.")
#
#
# if __name__ == "__main__":
#     args = parser_gen()
#     main(args)
