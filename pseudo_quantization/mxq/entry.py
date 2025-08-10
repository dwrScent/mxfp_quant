from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import torch
import argparse
import os, sys
import json
import numpy as np
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from mxq.utils.parallel import auto_parallel
from mxq.quantize.quantizer import pseudo_quant_output_mse, make_quant_linear,pseudo_quantize_model_weight
# from mxq.utils.lm_eval_adaptor import LMEvalAdaptor
from mxq.utils.utils import simple_dispatch_model

import datetime
import re
import tqdm
from torch import nn

from mxq.models.opt_giant import OPTForCausalLM_giant
from mxq.models.bloom_giant import BloomForCausalLM_giant
from mxq.models.llama_mxfp import LlamaForCausalLM_mxfp
from mxq.models.mistal_mxfp import MistralForCausalLM_mxfp
from mxq.models.qwen_vllm.qwen_mxfp import Qwen2ForCausalLM_mxfp
from mxq.eval import inference

from transformers import OPTConfig, BloomConfig, LlamaConfig, MistralConfig
from copy import deepcopy

from vllm import ModelRegistry
ModelRegistry.register_model("Qwen2ForCausalLM_mxfp", Qwen2ForCausalLM_mxfp)


def print_time(print_str):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} - {print_str}')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of the hf model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
# model config
parser.add_argument('--parallel', action='store_true',
                    help="enable model parallelism")
parser.add_argument('--auto_parallel', action='store_true',
                    help="automatically set parallel and batch_size")
# quantization config
parser.add_argument('--quant_bit_width', type=str, default='w16a16k16v16')
parser.add_argument('--w_bit', type=int, default=None)
parser.add_argument('--a_bit', type=int, default=16)
parser.add_argument('--q_bit', type=int, default=16)
parser.add_argument('--k_bit', type=int, default=16)
parser.add_argument('--v_bit', type=int, default=16)
parser.add_argument('--q_group_size', type=int, default=-1)
parser.add_argument('--no_zero_point', action='store_true',
                    help="disable zero_point")
parser.add_argument('--q_backend', type=str,
                    default="fake", choices=["fake", "real"])
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--em_bit", type=int, default=2)
parser.add_argument("--es_bit", type=int, default=2)

# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument('--quant_mode', type=str, default="compute_encode")
parser.add_argument('--ant_mode', type=str, default="int")
parser.add_argument('--mse_type', type=str, default="weight")
parser.add_argument('--ant_asym', type=int, default=0)
parser.add_argument('--w_low', type=int, default=75)
parser.add_argument('--w_high', type=int, default=150)
parser.add_argument('--mxfp_mode', type=str, default="w-base-a-base")

parser.add_argument('--outlier_type', type=str, default="none")
parser.add_argument('--outlier_ratio', type=float, default=-1.0)
parser.add_argument('--a_stride', type=int, default=5)

# save/load real quantized weights
parser.add_argument('--dump_quant', type=str, default=None,
                    help='save quantized model')
parser.add_argument('--load_quant', type=str, default=None,
                    help='load quantized model')

parser.add_argument("--max_iter", type=int, default=600,
                    help="kmeans max iterations")
args = parser.parse_args()

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization

}
quant_mode_config = {
    "quant_method": args.quant_mode,
    "quant_kv": False,
}
def parse_mxfp_modes(mode_str):
    """
    兼容两种输入格式：
    1. w-base-a-base
    2. w-sub_group+4+max-a-sub_group+4+max
    """
    # 详细格式匹配
    detailed_pattern = r'w-(\w+)\+(\d+)\+(\w+)-a-(\w+)\+(\d+)\+(\w+)'
    # 简单格式匹配
    simple_pattern = r'w-(\w+)-a-(\w+)'

    if re.match(detailed_pattern, mode_str):
        match = re.match(detailed_pattern, mode_str)
        return {
            "weight_mxfp_mode": match.group(1),
            "weight_sub_group_size": int(match.group(2)),
            "weight_sub_group_mode": match.group(3),
            "input_mxfp_mode": match.group(4),
            "input_sub_group_size": int(match.group(5)),
            "input_sub_group_mode": match.group(6),
        }
    elif re.match(simple_pattern, mode_str):
        match = re.match(simple_pattern, mode_str)
        return {
            "weight_mxfp_mode": match.group(1),
            "weight_sub_group_size": None,
            "weight_sub_group_mode": None,
            "input_mxfp_mode": match.group(2),
            "input_sub_group_size": None,
            "input_sub_group_mode": None,
        }
    else:
        raise ValueError(f"Invalid mode string: {mode_str}")

mxfp_config = parse_mxfp_modes(args.mxfp_mode)
ant_config = {
    "ant_mode": args.ant_mode,  
    "ant_search_granularity": 1,  
    "w_low": args.w_low,
    "w_high": args.w_high,
    "ant_asym": args.ant_asym,
    "weight_mxfp_mode": mxfp_config["weight_mxfp_mode"],
    "input_mxfp_mode": mxfp_config["input_mxfp_mode"],
    "weight_sub_group_size": mxfp_config.get("weight_sub_group_size"),
    "weight_sub_group_mode": mxfp_config.get("weight_sub_group_mode"),
    "input_sub_group_size": mxfp_config.get("input_sub_group_size"),
    "input_sub_group_mode": mxfp_config.get("input_sub_group_mode"),
    "es_bit": args.es_bit,
    "em_bit": args.em_bit,
    "topk": args.topk,
}

outlier_config = {
    "method": args.outlier_type,  
    "keep_ratio": args.outlier_ratio,  
    "keep_num": 1, 
}

def extract_bitwidths(quantization_string):
    w_bits = int(re.search(r'w(-?\d+)', quantization_string).group(1))
    a_bits = int(re.search(r'a(-?\d+)', quantization_string).group(1))
    q_bits = int(re.search(r'q(-?\d+)', quantization_string).group(1))
    k_bits = int(re.search(r'k(-?\d+)', quantization_string).group(1))
    v_bits = int(re.search(r'v(-?\d+)', quantization_string).group(1))
    return w_bits, a_bits, q_bits, k_bits, v_bits

args.w_bit, args.a_bit, args.q_bit, args.k_bit, args.v_bit = extract_bitwidths(args.quant_bit_width)
if args.k_bit < 16 or args.v_bit < 16:
    quant_mode_config['quant_kv'] = True
print("Quantization config:", q_config)
max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

# build model and tokenizer

def build_model_and_enc(model_path, need_og=False):
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path)
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    origin_model = None

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():

            if quant_mode_config['quant_kv']:
                kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}
                config.a_bit = args.a_bit
                config.w_bit = args.w_bit
                config.q_bit = args.q_bit
                config.k_bit = args.k_bit
                config.v_bit = args.v_bit
                config.group_size = args.q_group_size
                config.quant_kv = quant_mode_config['quant_kv']
                if isinstance(config, OPTConfig):
                    model = OPTForCausalLM_giant.from_pretrained(
                        model_path, config=config, **kwargs)
                elif isinstance(config, BloomConfig):
                    model = BloomForCausalLM_giant.from_pretrained(
                        model_path, config=config, **kwargs)
                elif isinstance(config, LlamaConfig):
                    model = LlamaForCausalLM_mxfp.from_pretrained(
                        model_path, config=config, **kwargs)
                elif isinstance(config, MistralConfig):
                    model = MistralForCausalLM_mxfp.from_pretrained(
                    model_path, config=config, **kwargs)
                else:
                    raise NotImplementedError('not support yet')
            else:
                model = AutoModelForCausalLM.from_config(
                        config=config, torch_dtype=torch.float16, trust_remote_code=True
                    )

        model.tie_weights()

        # Infer device map
        max_memory = {0: '38GiB', 1:'38GiB', 2: '38GiB', 3:'38GiB', 'cpu':'30GiB'}
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "OPTDecoderLayer_giant",
                "LlamaDecoderLayer",
                "LlamaDecoderLayer_giant",
                "BloomBlock",
                "BloomBlock_giant",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)
        print(model, device_map, max_memory)

        if quant_mode_config['quant_method'] in ['ant', 'olive', 'int', 'mokey', 'giant', 'mxfp', 'nvfp', 'smxfp']:
            make_quant_linear(
                model, args.w_bit, args.a_bit, q_config, ant_config=ant_config, quant_mode_config=quant_mode_config
            )

        model.eval()
    else:  # fp16 to quantized
        kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}

        # modify the attention layer
        if quant_mode_config['quant_kv']:
            config.a_bit = args.a_bit
            config.w_bit = args.w_bit
            config.q_bit = args.q_bit
            config.k_bit = args.k_bit
            config.v_bit = args.v_bit
            config.group_size = args.q_group_size
            config.quant_kv = quant_mode_config['quant_kv']
            if isinstance(config, OPTConfig):
                model = OPTForCausalLM_giant.from_pretrained(
                    model_path, config=config, **kwargs)
            elif isinstance(config, BloomConfig):
                model = BloomForCausalLM_giant.from_pretrained(
                    model_path, config=config, **kwargs)
            elif isinstance(config, LlamaConfig):
                model = LlamaForCausalLM_mxfp.from_pretrained(
                    model_path, config=config, **kwargs)
            elif isinstance(config, MistralConfig):
                model = MistralForCausalLM_mxfp.from_pretrained(
                    model_path, config=config, **kwargs)
            else:
                raise NotImplementedError('not support yet')
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, **kwargs)
        if need_og:
            origin_model = deepcopy(model)
        # weight quantization
        if args.w_bit and args.w_bit != -1:
            if args.q_backend == "fake":
                # assert args.dump_quant is None, \
                #     "Need to use real quantization to dump quantized weights"
                quant_mode = quant_mode_config['quant_method']
                print_time('Start pseudo quantize')

                if quant_mode in ['ant', 'olive', 'mxfp', 'nvfp', 'smxfp']:
                    make_quant_linear(
                        model, args.w_bit, args.a_bit, q_config, ant_config=ant_config, quant_mode_config=quant_mode_config
                    )
                elif quant_mode =='giant':
                    # weight quantization
                    if args.w_bit == 8:
                        pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                    elif args.w_bit == 4:
                        pseudo_quant_output_mse(
                            model, enc, w_bit=args.w_bit, q_config=q_config, ant_config=ant_config, n_samples=512, seqlen=512, max_iter=args.max_iter, a_stride=args.a_stride
                        )
                    else:
                        print('not supported yet')
                        exit(0)
                    make_quant_linear(
                        model, args.w_bit, args.a_bit, q_config, ant_config=ant_config, quant_mode_config=quant_mode_config
                    )
                elif quant_mode == 'int':
                    if quant_mode_config['quant_kv']:
                        print('quant KV Cache')
                    pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                    make_quant_linear(
                        model, args.w_bit, args.a_bit, q_config, ant_config=ant_config, quant_mode_config=quant_mode_config
                    )
                elif quant_mode == 'mokey':
                    make_quant_linear(model, 8, args.a_bit, q_config=q_config, quant_mode_config=quant_mode_config, init_only=False)
                else:
                    raise NotImplementedError(f"{args.mse_type} not supported yet!")
                print_time('Finish pseudo quantize')
                if args.dump_quant:
                    model.save_pretrained(f'quant_cache/{args.dump_quant}')
                    enc.save_pretrained(f'quant_cache/{args.dump_quant}')
                    print(f"Saving the quantized model at {args.dump_quant}...")
                    # torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError
    
    return model, enc, origin_model

def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    # if args.dump_awq and os.path.exists(args.dump_awq):
    #     print(f"Found existing AWQ results {args.dump_awq}, exit.")
    #     exit()

    print("\nargs:", args, "\n")
    # a hack here to auto set model group

    if args.tasks in ["AIME-2025", "AIME-90","MATH-500", "GSM8K", "GPQA-Diamond", "LiveCodeBench"]:
        config = AutoConfig.from_pretrained(args.model_path)
        name = config.architectures[0]
        if not name.endswith("_mxfp"):
            name += "_mxfp"
        config.architectures[0] = name
        config.save_pretrained(args.model_path)

        if name != "Qwen2ForCausalLM_mxfp":
                        raise Exception(f"{name} is not supported yet")
        inference.main(args.model_path, args.tasks)
        return

    if "mse" not in args.tasks:
        model, enc, origin_model = build_model_and_enc(args.model_path)
    else:
        model, enc, origin_model = build_model_and_enc(args.model_path, need_og=True)

    # lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
    lm_eval_model = HFLM(pretrained=model, batch_size=args.batch_size)


    
    if args.tasks is not None:
        # TODO: lm-eval 0.4.0 does not need the prefix hendrycksTest. This part can be updated.
        if args.tasks == "mmlu":
            # do evaluation on the Massive Multitask Language Understanding dataset
            task_dict = {"STEM":[], "humanities":[], "social sciences":[], "other (business, health, misc.)":[]}
            with open(os.getcwd() + '/mxq/utils/mmlu_data/categories.json', 'r') as f:
                categories = json.loads(f.read())
            with open(os.getcwd() + '/mxq/utils/mmlu_data/subcategories.json', 'r') as f:
                subcategories = json.loads(f.read())
            

            for key, value in subcategories.items():
                task_name = "".join(["hendrycksTest-", key])
                task_category = categories[value[0]]
                task_dict[task_category].append(task_name)
                
                
            total_acc_list = []
            total_acc_norm_list = []
            task_num_list = []
            
            docs_num_dict = {
                'STEM': [100, 152, 144, 100, 100, 100, 102, 100, 235, 145, 378, 310, 203, 100, 270, 151, 216, 112], 
                'humanities': [126, 165, 204, 237, 121, 108, 163, 346, 895, 311, 324, 1534, 171], 
                'social sciences': [114, 198, 193, 390, 238, 545, 131, 612, 110, 245, 201, 100], 
                'other (business, health, misc.)': [135, 100, 265, 173, 100, 223, 103, 234, 100, 783, 306, 282, 272, 166]
            }
            task_acc_dict = {
                "STEM":[],
                "humanities":[], 
                "social sciences":[], 
                "other (business, health, misc.)":[]
            }
            task_acc_norm_dict = {
                "STEM":[],
                "humanities":[], 
                "social sciences":[], 
                "other (business, health, misc.)":[]
            }
            print_time('Start a task')
            for key, value in task_dict.items():
                #task_names = ["hendrycksTest-anatomy", "hendrycksTest-astronomy"]
                task_names = value
                if task_names != []:
                    results = evaluator.simple_evaluate(
                        model=lm_eval_model,
                        tasks=task_names,
                        batch_size=args.batch_size,
                        # no_cache=True,
                        num_fewshot=args.num_fewshot,
                    )
                    # print(results)
                    acc_list = []
                    acc_norm_list = []
                    for task in task_names:
                        acc_list.append(results["results"][task]["acc"])
                        acc_norm_list.append(results["results"][task]["acc_norm"])
                    task_acc_dict[key] = acc_list
                    task_acc_norm_dict[key] = acc_norm_list
                    
                    print(evaluator.make_table(results))
                    print("Category " + key +": acc = {}".format(float(sum( np.multiply(docs_num_dict[key], acc_list).astype(int) )) / sum(docs_num_dict[key])))
                    print("Category " + key +": acc_norm = {}".format(float(sum( np.multiply(docs_num_dict[key], acc_norm_list).astype(int) )) / sum(docs_num_dict[key])))
                    total_acc_list.append(sum( np.multiply(docs_num_dict[key], acc_list).astype(int) ))
                    total_acc_norm_list.append(sum( np.multiply(docs_num_dict[key], acc_norm_list).astype(int) ))
                    task_num_list.append(sum(docs_num_dict[key]))                       
                    print(acc_list)
                    print(acc_norm_list)
                print_time('Task finish!')

            print("Category Average: acc = {}\n".format(float(sum(total_acc_list))/sum(task_num_list)))
            print("Category Average: acc_norm = {}\n".format(float(sum(total_acc_norm_list))/sum(task_num_list)))

        elif args.tasks in ['wikitext', 'c4', 'ptb']:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
            from .utils.dataload_utils import get_loaders
            model.seqlen = 2048
            _, testenc = get_loaders(args.tasks, model=args.model_path, seqlen=model.seqlen)
            
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            # nsamples = 10
            # nsamples = 30
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(1), desc="Data Type Search..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits_tmp = model(batch).logits
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())    
        elif args.tasks == 'wikitext-mse':
            from .utils.dataload_utils import get_loaders
            model.seqlen = 2048
            origin_model.seqlen = 2048
            _, testenc = get_loaders(args.tasks, model=args.model_path, seqlen=model.seqlen)
            
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            # nsamples = 10
            # nsamples = 30
            model = model.eval()
            nlls = []
            device       = next(model.parameters()).device  
            batch_mses   = []
            
            for i in tqdm.tqdm(range(nsamples),  desc="evaluating"):
                start_idx     = i * model.seqlen  
                end_idx       = start_idx + model.seqlen  
                batch         = testenc[:, start_idx:end_idx].to(device)
            
                with torch.no_grad(): 
                    logits_qt   = model(batch).logits         # [B=1,S,V]
                    logits_og   = origin_model(batch).logits 
            
                    diff_sq     =(logits_qt - logits_og).pow(2)
                    mse_this_batch= diff_sq.mean().item()       # ← one scalar per sample 
            
                    batch_mses.append(mse_this_batch) 
            
            global_mse=np.mean(batch_mses)                      # float64 safe for millions of samples 
            print("MSE:", global_mse)

        else:
            # do other evaluations
            print_time('Start a task')
            task_names = args.tasks.split(",")

            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                # no_cache=True,
                num_fewshot=args.num_fewshot,
                # limit=2,
            )
            print_time('Task finish!')
            # print(evaluator.make_table(results))
            print(make_table(results))

if __name__ == '__main__':
    main()
