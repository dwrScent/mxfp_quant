import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock
import os
from .ant_quant import ant_quantization, ant_quantization_search, meta_flint_set
from ..utils.make_distribution import group_dist_outlier, group_dist, make_heat_map, outlier_ratio_stat, outlier_count, outlier_judge_mask
import math
import kmeans_parallel
from .kmeans import ant_kmeans_quant

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import functools
from collections import defaultdict
import numpy as np
EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]

def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif 'mptblock' in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif 'falcon' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'bigcode' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif 'neox' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, 
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=8,
                           zero_point=True, q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False
                           ):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        # assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)

        # max_val.shape is 4096
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

def print_stats(mode_list, overall_stats, ant_config, mse_stats):
    print("\n OVERALL STATS \n")
    overall_select = 0
    for mode in mode_list:
        overall_select = overall_select + overall_stats[mode]
    for mode in mode_list:
        ratio = overall_stats[mode] / overall_select
        print(f"{mode} ratio: {ratio * 100:.3f}%")
    if "kmeans" == ant_config["ant_mode"]:
        print("USE KMEANS ONLY")
    elif "kmeans" in ant_config["ant_mode"] and "flint" in ant_config["ant_mode"]:
        print(f"mse kmeans: {mse_stats['kmeans']:.9f} mse ant: {mse_stats['ant']:.9f} mse overall: {mse_stats['overall']:.9f}")
        print(f"ant num: {mse_stats['ant_num']} * 1e5 kmeans num: {mse_stats['kmeans_num']} * 1e5")
        print(f"ant ratio: {mse_stats['ant_num'] / (mse_stats['ant_num'] + mse_stats['kmeans_num'] ) * 100:.3f}% kmeans ratio: {mse_stats['kmeans_num'] / (mse_stats['ant_num'] + mse_stats['kmeans_num']) * 100:.3f}% ")
    else:
        print(f"mse overall: {mse_stats['overall']:.9f}")

@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_config, model_path, ant_config=None, outlier_config=None
):   
    from .pre_quant import get_blocks, get_named_linears
    layers = get_blocks(model)
    mse = nn.MSELoss()

    mode_list = ant_config['ant_mode'].split('-')
    if 'meta_flint' in mode_list:
        mode_list.remove('meta_flint')
        mode_list.extend(meta_flint_set.keys())
    overall_stats = {}
    for mode in mode_list:
        overall_stats[mode] = torch.tensor(0.)
    mse_stats = {key: torch.tensor(0.) for key in ['kmeans', 'ant', 'overall', 'ant_num', 'kmeans_num']}
    total_entropy = torch.tensor(0.)
    total_tensor = torch.tensor(0.)

    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])

        
        for n, m in named_linears.items():
            w_init_data = m.weight.data.clone().cpu()
            
            if ant_config['ant_mode'] == "int":
                m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, **q_config)
                mse_stats['overall'] += mse(w_init_data.float(), m.weight.data.cpu().float())  
            elif "kmeans" in ant_config["ant_mode"]:
                m.weight.data = ant_kmeans_quant(m.weight.data, w_bit, q_config, ant_config, outlier_config, mse_stats, overall_stats, i, n)
            else:
                m.weight.data = ant_quantization_search(m.weight.data, w_bit, q_config, ant_config, outlier_config, overall_stats=overall_stats)
            
            mse_stats['overall'] += mse(w_init_data.float(), m.weight.data.cpu().float())  
     
            # _, counts = np.unique(m.weight.data.flatten(), return_counts=True)
            # probabilities = counts / counts.sum()
            # entropy = -np.sum(probabilities * np.log2(probabilities))

            # print(f'Entropy of the distribution: {entropy} bits')
            # total_entropy += entropy
            # total_tensor += 1.0

    print_stats(mode_list, overall_stats, ant_config, mse_stats)

    del mse_stats, overall_stats
    torch.cuda.empty_cache()

@torch.no_grad()
def pseudo_quant_output_mse(
    model, enc,
    w_bit, q_config,
    ant_config=None,
    n_samples=512, seqlen=512,
    # some configs for ablation study
    calib_data="pileval",
    max_iter=600
):
    from ..utils.calib_data import get_calib_dataset
    from .pre_quant import get_blocks, get_named_linears
    from .kmeans import use_kmeans_quantization
    from .ant_quant import generate_quant_grid, get_quant_weight
    from .codebook_gen import codebook_gen

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    gc.collect()
    torch.cuda.empty_cache()
    overall_mse = torch.tensor(0.).to(next(layers[0].parameters()).device)
    total_params = torch.tensor(0.).to(next(layers[0].parameters()).device)
    total_bits = torch.tensor(0.).to(next(layers[0].parameters()).device)
    mode_list = ant_config['ant_mode'].split('-')
    d_type_stats = {}
    tensor_stats = {}
    for mode in mode_list:
        d_type_stats[mode] = torch.tensor(0.)
        tensor_stats[mode] = torch.tensor(0.)

    # solve layer by layer
    for i in tqdm(range(len(layers)), desc="Psuedo quantizatoion with output MSE..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}


        tensor_dict = {'self_attn.q_proj':'q', 'self_attn.k_proj':'k', 'self_attn.v_proj':'v', 'self_attn.o_proj':'o_proj', 'mlp.up_proj':'up', 'mlp.down_proj':'down', 'mlp.gate_proj':'gate'}
        for name, m in named_linears.items():
            # if i == ant_config['layer_id'] and name == ant_config['tensor_name']:
            if tensor_dict[name] in ant_config['tensor_name']:
                tensor_bit = ant_config['tensor_bit']
            else:
                tensor_bit = w_bit

            input_x = input_feat[name] # 65 * 512 * k ,运算前先转换为二维的 m * k
            group_size = q_config["q_group_size"]
            if group_size == -1:
                group_size = m.weight.data.shape[1]
            group_num = m.weight.data.shape[1] // group_size
             
            input_x = input_x.to(m.weight.data.device)
            input_x = input_x.reshape(-1, input_x.shape[-1])
            
            tensor_mse = torch.tensor(0.).to(m.weight.data.device)

            # reserve 0.2
            # reserve_weight_mask = torch.where(m.weight.data.abs() > 0.2, 1, 0)
            # w_init = m.weight.data.clone()
            # m.weight.data = m.weight.data * (1 - reserve_weight_mask)
            # print(reserve_weight_mask, reserve_weight_mask.sum())

            for group_id in range(0, group_num):
                x = input_x[ : , group_id * group_size: (group_id + 1) * group_size ] 
                
                org_group_w = m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ]     
                org_group_output = torch.mm(x, org_group_w.T)
                
                def weight_quant(tensor_bit):
                    # support output MSE search for ant and kmeans
                    mask_list = [] # 0-int, 1-flint, 2-pot, 3-fp4/flint_0, 4-nf4, 5-kmeans
                    deq_w = torch.zeros_like(org_group_w, dtype=torch.half).to(m.weight.data.device)
                    min_mse = torch.full([1, m.weight.data.shape[0]], 10000.0).to(m.weight.data.device)  # 1 x N
                    quant_grid_set = generate_quant_grid(n_bit=tensor_bit, signed=True, ant_mode=ant_config['ant_mode'])
                    if 'meta_flint' in mode_list:
                        mode_list.remove('meta_flint')
                        mode_list.extend(meta_flint_set.keys())
                    # x_feature = x.abs().mean(0, keepdim=False)
                    
                    # for stats
                    data_type_identify = torch.zeros_like(min_mse, dtype=torch.int32)
                    mapping_list = {}

                    x_feature = x.mean(0, keepdim=False)
                    # for mode in ["weighted_kmeans"]:

                    for idx, mode in enumerate(mode_list):
                        if mode != "weighted_kmeans":
                            quant_grid = quant_grid_set[mode]
                            w_group_deq, _ = get_quant_weight(org_group_w, quant_grid, mode=mode, q_group_size=group_size)
                            w_group_deq = w_group_deq.half()
                        else:
                            w_group_deq = use_kmeans_quantization(org_group_w, w_bit=tensor_bit, x_feature=x_feature, zero_point=False, q_group_size=group_size, outlier_config=None, max_iter=max_iter)

                        deq_group_output = torch.mm(x, w_group_deq.T)
                        mse = (deq_group_output - org_group_output).pow(2).mean(0, keepdim=True)

                        sig = (mse <= min_mse).to(torch.half) # 1 x N
                        mask = sig.repeat(group_size, 1).T # N x group_size
                        org_mask = 1.0 - mask
                        deq_w = torch.mul(deq_w, org_mask) + torch.mul(w_group_deq, mask)
                        # deq_w = torch.mul(w_group_deq, mask)

                        # for stats
                        mapping_list[mode] = idx
                        data_type_identify = torch.where(mse < min_mse, idx, data_type_identify)
                        
                        # update min MSE
                        min_mse = torch.where(mse <= min_mse, mse, min_mse)

                    data_type_mask = {}
                    for mode in mode_list:
                        data_type_mask[mode] = (data_type_identify == mapping_list[mode])
                
                        d_type_stats[mode] = d_type_stats[mode].to(data_type_identify.device)
                        tensor_stats[mode] = tensor_stats[mode].to(data_type_identify.device)
                        d_type_stats[mode] = d_type_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) / 1e5
                        tensor_stats[mode] = tensor_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) 
                        # print(f"{mode}: {torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode])}")

                    return deq_w, min_mse
            
                use_fusion = False
                if use_fusion:
                    deq_w_5bit, mse_5bit = weight_quant(5)
                    deq_w_4bit, mse_4bit = weight_quant(4)
                    deq_w_3bit, mse_3bit = weight_quant(3)
                    mask_3bit = (mse_3bit < 1e-5)
                    mask_4bit = (mse_4bit < 1e-5)
                    # 3-bit 量化时 MSE > 1e-4，并且 4-bit 量化时 MSE < 1e-4 的 group 使用 4-bit
                    mask_4bit = (~mask_3bit & mask_4bit)
                    # 3-bit, 4-bit 都解决不了的用 5-bit
                    mask_5bit = (~mask_3bit & ~mask_4bit)
                    assert torch.sum(mask_3bit & mask_4bit & mask_5bit) == 0
                    assert torch.sum(mask_3bit | mask_4bit | mask_5bit) == m.weight.data.shape[0]

                    deq_w = torch.mul(deq_w_5bit, mask_5bit.repeat(group_size, 1).T) + torch.mul(deq_w_4bit, mask_4bit.repeat(group_size, 1).T) + torch.mul(deq_w_3bit, mask_3bit.repeat(group_size, 1).T)
                    min_mse = torch.mul(mse_5bit, mask_5bit) + torch.mul(mse_4bit, mask_4bit) + torch.mul(mse_3bit, mask_3bit)
                    total_bits += (mask_5bit.sum() * 5 * group_size + mask_4bit.sum() * 4 * group_size + mask_3bit.sum() * 3 * group_size) / 10000.0

                else: 
                    deq_w, min_mse = weight_quant(tensor_bit)
                    # total_bits += (tensor_bit * group_size * m.weight.data.shape[0]) / 10000.0
                # print(min_mse.mean(), torch.sum(min_mse < 1e-4), min_mse.numel())
                tensor_mse += min_mse.mean()
                # print(min_mse, min_mse.mean())
                m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ] = deq_w

            # reserve 0.2
            # print(f"max: {m.weight.data.max()} min: {m.weight.data.min()}")
            # m.weight.data = m.weight.data * (1 - reserve_weight_mask) + w_init * reserve_weight_mask
                
            # m.weight.data = w_init
            # 计算每个 tensor 的等效 bit width
            total_params += (m.weight.data.numel() / 10000.0) 
            # 16 是 scaling factor，3 用于选择 codebook，假设有 8 种 codebook
            if ant_config['ant_mode'] == "weighted_kmeans":
                total_bits += ((m.weight.data.numel() * tensor_bit + m.weight.data.shape[0] * (2 ** tensor_bit * 16.) * group_num ) / 10000.0)
                scaling_bits = 0.
            else:
                scaling_bits = (16. + 2.) / group_size if group_size != 1 else 0.00312
                total_bits += ((m.weight.data.numel() * tensor_bit) / 10000.0)
            
            # scaling_bits = (16. + 2.) / group_size if group_size != 1 else 0.00312   
            for mode in mode_list:
                print(f"{mode} num: {tensor_stats[mode]}")    
                tensor_stats[mode] = torch.tensor(0.)
            print(f"layer: {i}, {name}, tensor_mse: {tensor_mse / group_num} w_bit: {tensor_bit} {m.weight.data.shape} eq_bit: {total_bits/total_params + scaling_bits}")
            overall_mse = overall_mse.to(tensor_mse.device)
            overall_mse += tensor_mse

        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    overall_select = 0
    for mode in mode_list:
        overall_select = overall_select + d_type_stats[mode]
    for mode in mode_list:
        ratio = d_type_stats[mode] / overall_select
        print(f"{mode} ratio: {ratio * 100:.3f}%")
    print(f"overall_mse: {overall_mse} equivalent bit width: {total_bits / total_params + scaling_bits}")
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def real_quantize_model_weight(
    model, w_bit, q_config, ant_config=None, outlier_config=None,
    init_only=False
):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears
    # assert q_config["zero_point"], "We only support zero_point quantization now."
    
    layers = get_blocks(model)
    mode_list = ant_config['ant_mode'].split('-')

    # Generate quantization grid values for different data types

    if 'meta_flint' in mode_list:
        mode_list.remove('meta_flint')
        mode_list.extend(meta_flint_set.keys())
    overall_stats = {}
    for mode in mode_list:
        overall_stats[mode] = torch.tensor(0.)
    mse_stats = {key: torch.tensor(0.) for key in ['kmeans', 'ant', 'overall', 'ant_num', 'kmeans_num']}
    for i in tqdm(range(len(layers)), desc="real weight quantization..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)
        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config['q_group_size'], True)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.weight.data, labels, codebook = ant_kmeans_quant(module.weight.data, w_bit, q_config, ant_config, outlier_config, mse_stats, overall_stats, i, name, get_labels=True)
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config['q_group_size'], False, labels, codebook)
                # module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()
    # 统计 ant mode 选择情况
    print_stats(mode_list, overall_stats, ant_config, mse_stats)
    

    torch.cuda.empty_cache()
    gc.collect()