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
from awq.quantize.quantizer import pseudo_quant_output_mse, make_quant_linear, pseudo_quantize_model_weight
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
import torch.nn as nn
from tqdm import tqdm

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
parser.add_argument(
    '--seed',
    type=int, default=0, help='Seed for sampling the calibration data.'
)

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
        # with init_empty_weights():
            # if quant_mode_config['quant_method'] =='giant' and quant_mode_config['quant_kv']:
            #     kwargs_init = {"device_map": "balanced", "torch_dtype": torch.float16}
            #     model = OPTForCausalLM_giant.from_pretrained(
            #         model_path, config=config, **kwargs_init)

            # else:
            #     model = AutoModelForCausalLM.from_config(
            #         config=config, torch_dtype=torch.float16, trust_remote_code=True
            #     )

        kwargs_init = {"device_map": "balanced", "torch_dtype": torch.float16}
        model = OPTForCausalLM_giant.from_pretrained(
            args.load_quant, config=config, **kwargs_init)

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

        if quant_mode_config['quant_method'] in ['ant', 'olive', 'int', 'mokey', 'giant']:
            make_quant_linear(
                model, args.w_bit, args.a_bit, q_config, ant_config=ant_config, quant_mode_config=quant_mode_config
            )

        model.eval()
    else:
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
        if args.w_bit is not None and args.w_bit != -1:
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
                    # exit(0)
            # elif args.q_backend == "real":
            #     pass
            else:
                raise NotImplementedError

    return model, enc
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext' in name:
        print("wikekwkw")
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        # if 'new' in name:
        #     return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        # if 'new' in name:
        #     return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    # traindata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    # model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def main():

    print("\nargs:", args, "\n")
    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)
    model.seqlen = 2048
    # model = get_llama(args.model)

    model.eval()

    if args.tasks is not None:          
        # do other evaluations
        print_time('Start a task')
        task_names = args.tasks.split(",")
        dev = torch.device('cuda:0')
        dataset = args.tasks
        dataloader, testenc = get_loaders(
            dataset, seed=args.seed, model=args.model_path, seqlen=model.seqlen
        )
        testenc = testenc.input_ids
        nsamples = testenc.numel() // model.seqlen
        
        # nsamples = 1

        use_cache = model.config.use_cache
        model.config.use_cache = False


        layers = model.model.decoder.layers

 

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {'i': 0, 'attention_mask': None}



        if 'opt' in args.model_path:
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    cache['attention_mask'] = kwargs['attention_mask']
                    raise ValueError
            layers[0] = Catcher(layers[0])
            for i in range(nsamples):
                batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
                try:
                    model(batch)
                except ValueError:
                    pass
            layers[0] = layers[0].module
            
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.cpu()
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.cpu()
            torch.cuda.empty_cache()
            outs = torch.zeros_like(inps)
            attention_mask = cache['attention_mask']

            # layers[0] = layers[0].cpu()
        # 无需搬移到 CPU
        # layers[0] = layers[0].cpu()
        


        inps_for_search = inps.clone().detach()
        outs_for_search = outs.clone().detach()

        for i in tqdm(range(len(layers)), desc="data type search..."):
            # if i == 26:
            #     continue
            layer = layers[i]
            # if i==1:
            #     print(inps_for_search[0])
            inps_for_search = inps_for_search.to(layer.self_attn.q_proj.weight.data.device)
            # 用第一个 sample 进行 search，确定 data type
            # outs_for_search[0] = layer(inps_for_search[0].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs_for_search[0] = layer(inps_for_search[0].unsqueeze(0), attention_mask=attention_mask)[0]
            inps_for_search, outs_for_search = outs_for_search, inps_for_search
            dev = inps_for_search.device
        

        for i in tqdm(range(len(layers)), desc="forwarding..."):
            # print(i)
            # layer = layers[i].to(dev)
            layer = layers[i]
            # print(layer.self_attn.q_proj.weight.data)

            inps = inps.to(layer.self_attn.q_proj.weight.data.device)

            # print(inps.device)
            
            for j in range(nsamples):
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

            # 为了后面的操作，因为这一步完成之后数据都在最后一张 device 上
            dev = inps.device

        print("for loop done")
        
        
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        model.lm_head = model.lm_head.to(dev)

        testenc = testenc.to(dev)
        # testenc = testenc.to(torch.device('cuda:3'))
        nlls = []

        # 回到 dev
        inps = inps.to(dev)
        # inps = inps.to(torch.device('cuda:3'))
        outs = outs.to(dev)
        # outs = outs.to(torch.device('cuda:3'))
        # print(inps.device, model.device)
        for i in range(nsamples):
            hidden_states = inps[i].unsqueeze(0)
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc[
                :, (i * model.seqlen):((i + 1) * model.seqlen)
            ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print("ppl: ",ppl.item())

        model.config.use_cache = use_cache
        print_time('Task finish!')

if __name__ == '__main__':
    main()
