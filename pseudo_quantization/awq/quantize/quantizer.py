import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock
import os
from .ant_quant import ant_quantization, ant_quantization_search, meta_flint_set
import math

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import functools
from collections import defaultdict
import numpy as np
import copy

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    # Padding if the last dimension is not divisible by q_group_size
    if q_group_size > 0:
        # assert org_w_shape[-1] % q_group_size == 0
        if org_w_shape[-1] % q_group_size != 0:
            # return tensor
            padding = (q_group_size - org_w_shape[-1] % q_group_size) % q_group_size
            # Padding is only added to the last dimension
            w = torch.nn.functional.pad(w, (0, padding), "constant", 0)
            # torch.set_printoptions(profile="full")
            print(f'padding last dimension from {org_w_shape[-1]} to {w.shape[-1]}')
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        # assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    # Remove padding to restore original shape
    if org_w_shape[-1] % q_group_size != 0:
        w = w.reshape(-1, org_w_shape[-1] + padding)
        w = w[:, :org_w_shape[-1]]  
    else:
        w = w.reshape(org_w_shape)

    # w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            # m.cpu()

@torch.no_grad()
def pseudo_quant_stats(
    model, enc,
    w_bit, q_config,
    ant_config=None,
    n_samples=512, seqlen=512,
    # some configs for ablation study
    calib_data="pileval",
    max_iter=600,
    a_stride=5
):
    from ..utils.calib_data import get_calib_dataset
    from .pre_quant import get_blocks, get_named_linears
    from .ant_quant import generate_quant_grid, get_quant_weight
    from .qmodule_giant import encode_gen, encode_gen_no_zero

    
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
    # mode_list = ant_config['ant_mode'].split('-')
    # if 'meta_flint' in mode_list:
    #     mode_list.remove('meta_flint')
    #     mode_list.extend(meta_flint_set.keys())

    # quant_grid_set = encode_gen(w_bit)
    quant_grid_set = encode_gen_no_zero(w_bit, a_stride=a_stride)

    int_grid_set = generate_quant_grid(n_bit=w_bit, signed=True, ant_mode='int')
    mode_list = []
    # mode_list = ant_config['ant_mode'].split('-')
    mode_list.extend(quant_grid_set.keys())
    mode_list.append('int')
    quant_grid_set['int'] = int_grid_set['int']    

    # ant_config['ant_mode'] = 'int-flint-pot'
    # mode_list = ant_config['ant_mode'].split('-')
    # quant_grid_set = generate_quant_grid(n_bit=w_bit, signed=True, ant_mode=ant_config['ant_mode'])
    

    d_type_stats = {}
    tensor_stats = {}
    for mode in mode_list:
        d_type_stats[mode] = torch.tensor(0.)
        tensor_stats[mode] = torch.tensor(0.)

    # solve layer by layer
    for i in tqdm(range(len(layers)), desc="Psuedo weight quantizatoion with output MSE..."):
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
        layer_kwargs_copy = copy.deepcopy(layer_kwargs)
        
        # new copy
        inps = layer(inps, **layer_kwargs)[0]
        # layer(inps, **layer_kwargs)
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        for name, m in named_linears.items():
            # if name == 'self_attn.q_proj' or name == 'self_attn.k_proj' or name == 'self_attn.v_proj':
            # if name == 'self_attn.o_proj':
            # if 'mlp' in name:
            input_x = input_feat[name] # 65 * 512 * k ,运算前先转换为二维的 m * k
            group_size = q_config["q_group_size"]
            if group_size == -1:
                group_size = m.weight.data.shape[1]
            group_num = m.weight.data.shape[1] // group_size
            
            input_x = input_x.to(m.weight.data.device)
            input_x = input_x.reshape(-1, input_x.shape[-1])

            # m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=6, **q_config)
            org_weight = m.weight.data.clone()

            total_labels = torch.zeros_like(m.weight.data, dtype=torch.int).to(m.weight.data.device)
            
            tensor_mse = torch.tensor(0.).to(m.weight.data.device)
            for group_id in range(0, group_num):
                x = input_x[ : , group_id * group_size: (group_id + 1) * group_size ] 
                
                org_group_w = m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ]     
                org_group_output = torch.mm(x, org_group_w.T)
                
                def weight_quant():
                    # support output MSE search for ant and kmeans
                    mask_list = [] # 0-int, 1-flint, 2-pot, 3-fp4/flint_0, 4-nf4, 5-kmeans
                    deq_w = torch.zeros_like(org_group_w, dtype=torch.half).to(m.weight.data.device)

                    deq_labels = torch.zeros_like(org_group_w, dtype=torch.int).to(m.weight.data.device) - 1

                    # min_mse = torch.full([1, m.weight.data.shape[0]], 10000.0).to(m.weight.data.device)  # 1 x N

                    min_mse = torch.full([m.weight.data.shape[0], 1], 10000.0).to(m.weight.data.device)  # N x 1

                    
                    # x_feature = x.abs().mean(0, keepdim=False)
                    # for stats
                    data_type_identify = torch.zeros_like(min_mse, dtype=torch.int32)
                    mapping_list = {}

                    x_feature = x.mean(0, keepdim=False)
                    # for mode in ["weighted_kmeans"]:
    
                    for idx, mode in enumerate(mode_list):
                        quant_grid = quant_grid_set[mode]
                        quant_grid, _ = quant_grid.sort()
                        print(quant_grid)
                        w_group_deq, _, labels, _ = get_quant_weight(org_group_w, quant_grid, mode=mode, q_group_size=group_size, get_labels=True)
                        # w_group_deq, _ = get_quant_weight(org_group_w, quant_grid, mode=mode, q_group_size=group_size)
                        w_group_deq = w_group_deq.half()

                        deq_group_output = torch.mm(x, w_group_deq.T)

                        mse = (deq_group_output - org_group_output).pow(2).mean(0, keepdim=True)

                        weight_mse = (org_group_w - w_group_deq).pow(2).mean(1, keepdim=True)

                        sig = (weight_mse <= min_mse).to(torch.half) # N x 1

                        # sig = (mse <= min_mse).to(torch.half) # 1 x N
                        # mask = sig.repeat(group_size, 1).T # N x group_size
                        mask = sig.repeat(1, group_size) # N x group_size
                        org_mask = 1.0 - mask

                        # print(org_mask.shape, mask.shape, deq_w.shape, w_group_deq.shape)
                        

                        deq_w = torch.mul(deq_w, org_mask) + torch.mul(w_group_deq, mask)
                        deq_labels = torch.mul(deq_labels, org_mask) + torch.mul(labels, mask)

                        # deq_w = torch.mul(w_group_deq, mask)
                        # for stats
                        mapping_list[mode] = idx
                        # data_type_identify = torch.where(mse < min_mse, idx, data_type_identify)
                        data_type_identify = torch.where(weight_mse < min_mse, idx, data_type_identify)
                        
                        # update min MSE
                        # min_mse = torch.where(mse <= min_mse, mse, min_mse)
                        min_mse = torch.where(weight_mse <= min_mse, weight_mse, min_mse)
                    data_type_mask = {}
                    for mode in mode_list:
                        data_type_mask[mode] = (data_type_identify == mapping_list[mode])
                
                        d_type_stats[mode] = d_type_stats[mode].to(data_type_identify.device)
                        tensor_stats[mode] = tensor_stats[mode].to(data_type_identify.device)
                        d_type_stats[mode] = d_type_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) / 1e5
                        tensor_stats[mode] = tensor_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) 
                    
                    assert not torch.any(deq_labels == -1), "Tensor contains -1"
                    return deq_w, min_mse, deq_labels
                
                deq_w, min_mse, deq_labels = weight_quant()


                tensor_mse += min_mse.mean()
                # print(min_mse, min_mse.mean())
                m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ] = deq_w
                total_labels[ : ,group_id * group_size: (group_id + 1) * group_size ] = deq_labels

            def compute_row_frequencies(tensor):
                row_frequencies = []
                for row in tensor:
                    values, counts = torch.unique(row, return_counts=True)
                    total_elements = row.numel()
                    row_freq = {value.item(): count.item() / total_elements for value, count in zip(values, counts)}
                    row_frequencies.append(row_freq)
                return row_frequencies

            def compute_entropy(frequencies):
                entropy = -sum(p * np.log2(p) for p in frequencies.values() if p > 0)
                return entropy

            def compute_average_row_entropy(tensor):
                row_frequencies = compute_row_frequencies(tensor)
                row_entropies = [compute_entropy(freq) for freq in row_frequencies]
                average_entropy = np.mean(row_entropies)
                return average_entropy
            
            # print(total_labels, total_labels.shape, m.weight.data)
            # compute frequency

            values, counts = torch.unique(total_labels, return_counts=True)
            total_elements = total_labels.numel()
            frequencies = {value.item(): count.item() / total_elements for value, count in zip(values, counts)}
            # frequencies[8] = 0
            print(frequencies)
            entropy = -sum(p * np.log2(p) for p in frequencies.values() if p > 0)

            average_entropy = compute_average_row_entropy(total_labels)

            for value, frequency in sorted(frequencies.items()):
                print(f'{value}: {frequency:.4f}')
            for value, frequency in sorted(frequencies.items()):
                print(f'{frequency:.4f}')
            print(name, entropy)
            print(average_entropy)

            for mode in mode_list:
                print(f"{mode} num: {tensor_stats[mode]}")    
                tensor_stats[mode] = torch.tensor(0.)
            
            # print(m.weight.data.shape, m.weight.data)
            mse = nn.MSELoss()
            print(mse(org_weight, m.weight.data))

            input('contuine')


            print(f"layer: {i}, {name}, tensor_mse: {tensor_mse}")
            overall_mse = overall_mse.to(tensor_mse.device)
            overall_mse += tensor_mse
            # exit(0)
        input('contuine')

        # new copy
        inps = layer(inps, **layer_kwargs_copy)[0]
        del layer_kwargs_copy

        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    overall_select = 0
    for mode in mode_list:
        overall_select = overall_select + d_type_stats[mode]
    for mode in mode_list:
        ratio = d_type_stats[mode] / overall_select
        print(f"{mode} ratio: {ratio * 100:.3f}%")
    print(f"overall_mse: {overall_mse}")

    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def pseudo_quant_output_mse(
    model, enc,
    w_bit, q_config,
    ant_config=None,
    n_samples=512, seqlen=512,
    # some configs for ablation study
    calib_data="pileval",
    max_iter=600,
    a_stride=5
):
    from ..utils.calib_data import get_calib_dataset
    from .pre_quant import get_blocks, get_named_linears
    from .ant_quant import generate_quant_grid, get_quant_weight
    from .qmodule_giant import encode_gen, encode_gen_no_zero

    
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
    # mode_list = ant_config['ant_mode'].split('-')
    # if 'meta_flint' in mode_list:
    #     mode_list.remove('meta_flint')
    #     mode_list.extend(meta_flint_set.keys())

    # quant_grid_set = encode_gen(w_bit)
    quant_grid_set = encode_gen_no_zero(w_bit, a_stride=a_stride)

    int_grid_set = generate_quant_grid(n_bit=w_bit, signed=True, ant_mode='int')
    mode_list = []
    # mode_list = ant_config['ant_mode'].split('-')
    mode_list.extend(quant_grid_set.keys())
    mode_list.append('int')
    quant_grid_set['int'] = int_grid_set['int']    

    # quant_grid_set = generate_quant_grid(n_bit=w_bit, signed=True, ant_mode='int-flint-pot')
    # mode_list = ['int', 'flint', 'pot']



    d_type_stats = {}
    tensor_stats = {}
    for mode in mode_list:
        d_type_stats[mode] = torch.tensor(0.)
        tensor_stats[mode] = torch.tensor(0.)

    # solve layer by layer
    for i in tqdm(range(len(layers)), desc="Psuedo weight quantizatoion with output MSE..."):
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
        layer_kwargs_copy = copy.deepcopy(layer_kwargs)
        
        # new copy
        inps = layer(inps, **layer_kwargs)[0]
        # layer(inps, **layer_kwargs)
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        for name, m in named_linears.items():
            # if name == 'self_attn.q_proj' or name == 'self_attn.k_proj' or name == 'self_attn.v_proj':
            # if name == 'self_attn.o_proj':
            # if 'mlp' in name:
            input_x = input_feat[name] # 65 * 512 * k ,运算前先转换为二维的 m * k
            group_size = q_config["q_group_size"]
            if group_size == -1:
                group_size = m.weight.data.shape[1]
            group_num = m.weight.data.shape[1] // group_size
            
            input_x = input_x.to(m.weight.data.device)
            input_x = input_x.reshape(-1, input_x.shape[-1])

            # m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=6, **q_config)    
            tensor_mse = torch.tensor(0.).to(m.weight.data.device)
            for group_id in range(0, group_num):
                x = input_x[ : , group_id * group_size: (group_id + 1) * group_size ] 
                
                org_group_w = m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ]     
                org_group_output = torch.mm(x, org_group_w.T)
                
                def weight_quant():
                    # support output MSE search for ant and kmeans
                    mask_list = [] # 0-int, 1-flint, 2-pot, 3-fp4/flint_0, 4-nf4, 5-kmeans
                    deq_w = torch.zeros_like(org_group_w, dtype=torch.half).to(m.weight.data.device)
                    min_mse = torch.full([1, m.weight.data.shape[0]], 10000.0).to(m.weight.data.device)  # 1 x N
                    
                    # x_feature = x.abs().mean(0, keepdim=False)
                    # for stats
                    data_type_identify = torch.zeros_like(min_mse, dtype=torch.int32)
                    mapping_list = {}

                    x_feature = x.mean(0, keepdim=False)
                    # for mode in ["weighted_kmeans"]:
    
                    for idx, mode in enumerate(mode_list):
                        quant_grid = quant_grid_set[mode]
                        w_group_deq, _ = get_quant_weight(org_group_w, quant_grid, mode=mode, q_group_size=group_size)
                        w_group_deq = w_group_deq.half()

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
                    return deq_w, min_mse
                
                deq_w, min_mse = weight_quant()
                tensor_mse += min_mse.mean()
                # print(min_mse, min_mse.mean())
                m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ] = deq_w


            for mode in mode_list:
                print(f"{mode} num: {tensor_stats[mode]}")    
                tensor_stats[mode] = torch.tensor(0.)
            
            print(f"layer: {i}, {name}, tensor_mse: {tensor_mse}")
            overall_mse = overall_mse.to(tensor_mse.device)
            overall_mse += tensor_mse
            # exit(0)

        # new copy
        inps = layer(inps, **layer_kwargs_copy)[0]
        del layer_kwargs_copy

        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    overall_select = 0
    for mode in mode_list:
        overall_select = overall_select + d_type_stats[mode]
    for mode in mode_list:
        ratio = d_type_stats[mode] / overall_select
        print(f"{mode} ratio: {ratio * 100:.3f}%")
    print(f"overall_mse: {overall_mse}")

    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def make_quant_linear(
    model, w_bit, a_bit, q_config, ant_config=None, outlier_config=None, quant_mode_config=None,
    init_only=False
):
    from .pre_quant import get_blocks, get_named_linears
    # assert q_config["zero_point"], "We only support zero_point quantization now."
    
    layers = get_blocks(model)

    # layers = layers.cpu()
    for i in tqdm(range(len(layers)), desc=" make quant linear..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        for name, module in named_linears.items():
            # module.cuda()
            if quant_mode_config['quant_method'] == 'ant':
                from .qmodule_ant import ANT_Linear
                q_linear = ANT_Linear.from_linear(
                    module, w_bit, a_bit, q_config['q_group_size'], i, name, init_only=False, ant_config=ant_config)
            elif quant_mode_config['quant_method'] == 'olive':
                from .qmodule_olive import OliVe_Linear
                q_linear = OliVe_Linear.from_linear(
                    module, w_bit, a_bit, q_config['q_group_size'], i, name, init_only=False, ant_config=ant_config)
            elif quant_mode_config['quant_method'] == 'giant':
                from .qmodule_giant import GIANT_Linear
                q_linear = GIANT_Linear.from_linear(
                    module, w_bit, a_bit, q_config['q_group_size'], i, name, quant_mode_config['quant_kv'], init_only=False, ant_config=ant_config)
            elif quant_mode_config['quant_method'] == 'int':
                from .qmodule_giant import GIANT_Linear
                q_linear = GIANT_Linear.from_linear(
                    module, w_bit, a_bit, q_config['q_group_size'], i, name, quant_mode_config['quant_kv'], init_only=False, ant_config=ant_config)
            elif quant_mode_config['quant_method'] == 'mokey':
                from .qmodule_mokey import Mokey_Linear
                q_linear = Mokey_Linear.from_linear(module, layer_id=i, layer_name=name)
            elif quant_mode_config['quant_method'] == 'mxfp':
                from .qmodule_mxfp import MXFP_Linear
                q_linear = MXFP_Linear.from_linear(
                    module, w_bit, a_bit, q_config['q_group_size'], i, name, init_only=False, ant_config=ant_config, quant_mode="fp4_e2m1")
            else:
                pass
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)

        #     module.cpu()
        #     del module
        #     torch.cuda.empty_cache()
        #     gc.collect()
        # del layer
        # torch.cuda.empty_cache()
        # gc.collect()


    torch.cuda.empty_cache()
    gc.collect()