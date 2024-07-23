from lm_eval import evaluator
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
from awq.utils.parallel import auto_parallel
from awq.quantize.quantizer import pseudo_quant_output_mse, make_quant_linear,pseudo_quantize_model_weight
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model

import datetime

from awq.models.opt_giant import OPTForCausalLM_giant
from awq.models.bloom_giant import BloomForCausalLM_giant
from awq.models.llama_giant import LlamaForCausalLM_giant

from transformers import OPTConfig, BloomConfig, LlamaConfig

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
parser.add_argument('--w_bit', type=int, default=None)
parser.add_argument('--a_bit', type=int, default=16)
parser.add_argument('--q_group_size', type=int, default=-1)
parser.add_argument('--no_zero_point', action='store_true',
                    help="disable zero_point")
parser.add_argument('--q_backend', type=str,
                    default="fake", choices=["fake", "real"])
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
parser.add_argument('--quant_kv', type=int, default=0)
parser.add_argument('--ant_mode', type=str, default="int")
parser.add_argument('--mse_type', type=str, default="weight")
parser.add_argument('--ant_search_granularity', type=int, default=1)
parser.add_argument('--ant_asym', type=int, default=0)
parser.add_argument('--w_low', type=int, default=75)
parser.add_argument('--w_high', type=int, default=150)

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
    "quant_kv": args.quant_kv,
}
ant_config = {
    "ant_mode": args.ant_mode,  
    "ant_search_granularity": args.ant_search_granularity,  
    "w_low": args.w_low,
    "w_high": args.w_high,
    "ant_asym": args.ant_asym,
}

outlier_config = {
    "method": args.outlier_type,  
    "keep_ratio": args.outlier_ratio,  
    "keep_num": 1, 
}

print("Quantization config:", q_config)
max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

# build model and tokenizer

def build_model_and_enc(model_path):
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path)
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            kwargs_init = {"device_map": "balanced", "torch_dtype": torch.float16}
            if quant_mode_config['quant_method'] =='giant' and quant_mode_config['quant_kv'] and isinstance(config, OPTConfig):
                model = OPTForCausalLM_giant.from_pretrained(
                    model_path, config=config, **kwargs_init)
            elif quant_mode_config['quant_method'] =='giant' and quant_mode_config['quant_kv'] and isinstance(config, BloomConfig):
                model = BloomForCausalLM_giant.from_pretrained(
                    model_path, config=config, **kwargs_init)
            elif quant_mode_config['quant_method'] =='giant' and quant_mode_config['quant_kv'] and isinstance(config, LlamaConfig):
                model = LlamaForCausalLM_giant.from_pretrained(
                    model_path, config=config, **kwargs_init)
            else:
                model = AutoModelForCausalLM.from_config(
                        config=config, torch_dtype=torch.float16, trust_remote_code=True
                    )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
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

        if quant_mode_config['quant_method'] in ['ant', 'olive', 'int', 'mokey', 'giant']:
            make_quant_linear(
                model, args.w_bit, args.a_bit, q_config, ant_config=ant_config, quant_mode_config=quant_mode_config
            )

        model.eval()
    else:  # fp16 to quantized
        kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}

        # modify the attention layer
        if quant_mode_config['quant_method'] =='giant' and quant_mode_config['quant_kv'] and isinstance(config, OPTConfig):
            model = OPTForCausalLM_giant.from_pretrained(
                model_path, config=config, **kwargs)
        elif quant_mode_config['quant_method'] =='giant' and quant_mode_config['quant_kv'] and isinstance(config, BloomConfig):
            model = BloomForCausalLM_giant.from_pretrained(
                model_path, config=config, **kwargs)
        elif quant_mode_config['quant_method'] =='giant' and quant_mode_config['quant_kv'] and isinstance(config, LlamaConfig):
            model = LlamaForCausalLM_giant.from_pretrained(
                model_path, config=config, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, **kwargs)

        # weight quantization
        if args.w_bit and args.w_bit != -1 and args.w_bit < 16:
            if args.q_backend == "fake":
                # assert args.dump_quant is None, \
                #     "Need to use real quantization to dump quantized weights"
                quant_mode = quant_mode_config['quant_method']
                print_time('Start pseudo quantize')

                if quant_mode in ['ant', 'olive', 'mokey']:
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

    return model, enc


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
    model, enc = build_model_and_enc(args.model_path)

    lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
    
    if args.tasks is not None:
        if args.tasks == "mmlu" :
            # do evaluation on the Massive Multitask Language Understanding dataset
            task_dict = {"STEM":[], "humanities":[], "social sciences":[], "other (business, health, misc.)":[]}
            with open(os.getcwd() + '/awq/mmlu_data/categories.json', 'r') as f:
                categories = json.loads(f.read())
            with open(os.getcwd() + '/awq/mmlu_data/subcategories.json', 'r') as f:
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
                        no_cache=True,
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

                    
        else:
            # do other evaluations
            print_time('Start a task')
            task_names = args.tasks.split(",")

            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )
            print_time('Task finish!')
            print(evaluator.make_table(results))

if __name__ == '__main__':
    main()
