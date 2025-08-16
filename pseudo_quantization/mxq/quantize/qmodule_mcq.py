import torch
import torch.nn as nn
# from .ant_quant import generate_quant_grid
# from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

import logging
import time

import numpy as np

# from .permute import permutation_zigzag
# from .count import count_exist



# def pseudo_quantize_int(tensor, n_bit=8, zero_point=True, q_group_size=-1, alpha=1.0, is_input=False):
#     org_shape = tensor.shape
#     assert q_group_size == -1
#     if q_group_size > 0:
#         assert org_shape[-1] % q_group_size == 0
#         tensor = tensor.reshape(-1, q_group_size)
#     # assert tensor.dim() == 2

#     if is_input:
#         max_val = tensor.abs().amax()
#         max_val = max_val.clamp(min=1e-5)
#     else:
#         max_val = tensor.abs().amax(dim=1, keepdim=True)
#         max_val = max_val.clamp(min=1e-5)

#     max_int = 2 ** (n_bit - 1) - 1
#     min_int = - 2 ** (n_bit - 1)
#     scales = (max_val * alpha) / max_int
#     zeros = 0

#     assert torch.isnan(scales).sum() == 0
#     assert torch.isnan(tensor).sum() == 0

#     tensor = (torch.clamp(torch.round(tensor / scales) +
#                          zeros, min_int, max_int) - zeros) * scales
#     assert torch.isnan(tensor).sum() == 0
#     tensor = tensor.reshape(org_shape)
#     return tensor

# @torch.no_grad()
# def get_quant(tensor_value, quant_grid, alpha=1.0, is_input=False):

#     org_shape = tensor_value.shape
#     quant_grid = quant_grid.to(tensor_value.device)

#     # tensor-wise for activation quantization
#     if is_input:
#         max_val = tensor_value.abs().amax()
#     # channel-wise for weight quantization
#     else:
#         max_val = tensor_value.abs().amax(dim=1, keepdim=True)

#     max_quant_val = max(quant_grid)
#     scales = (max_val * alpha) / max_quant_val
#     zeros = 0

#     # labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
#     # tensor_deq = quant_grid[labels] * scales - zeros
#     # argmin, find to index
#     if is_input:
#         labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
#         tensor_deq = quant_grid[labels] * scales - zeros
#     else:
#         # Batch processing to avoid OOM
#         batch_num = 4
#         assert org_shape[0] % batch_num == 0
#         batch_size = org_shape[0] // batch_num
#         tensor_deq = torch.zeros_like(tensor_value)
#         for idx in range(batch_num):
#             tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
#             labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
#             tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
#             tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

#     # quant_mse = (tensor_deq - tensor_value).abs().pow(2)
#     tensor_deq = tensor_deq.to(tensor_value.device).half()

#     return tensor_deq

# def ant_quant(self, n_bit, weight, input, ant_config, group_size, layer_id, layer_name, is_input=False):
#     # channel-wise quantization for weight, tensor-wise for activation
#     assert group_size == -1, "ANT use per-channel quantization for weight and per-tensor for activation"
#     mode_list = ant_config['ant_mode'].split('-')
#     quant_grid_set = generate_quant_grid(n_bit=n_bit, signed=True, ant_mode=ant_config['ant_mode'])

#     mx_emax = {'fp4_e2m1': 2, 'fp6_e3m2': 4, 'fp6_e2m3': 2, 'fp8_e4m3': 8}

#     # transform to float to avoid overflow in MSE compute
#     org_output = torch.mm(input, weight.T).to(torch.float)

#     min_mse = float('inf')
#     best_mode = 'null'
#     best_alpha = -1
#     # lower bound and upper bound of alpha
#     lb = ant_config['w_low']
#     ub = ant_config['w_high']

#     # tensor-wise for activation quantization
#     if is_input:
#         org_shape = input.shape
#         final_tensor = torch.zeros_like(input, dtype=torch.half).to(input.device)
#         input = input.reshape(-1)
#     else:
#         final_tensor = torch.zeros_like(weight, dtype=torch.half).to(weight.device)

#     for idx, mode in enumerate(mode_list):
#         quant_grid = quant_grid_set[mode]
#         for i in range(lb, ub, 10):
#             search_alpha = i * 0.01

#             if is_input:
#                 if n_bit > 6:
#                     tensor_deq = pseudo_quantize_int(input, n_bit=n_bit, zero_point=False, q_group_size=group_size, is_input=True)
#                 else:
#                     tensor_deq = get_quant(input, quant_grid, alpha=search_alpha, is_input=True)
#                 # reshape from (-1) to (seq_len, hidden), for mm operation
#                 tensor_deq = tensor_deq.reshape(org_shape)
#                 # transform to float to avoid overflow in MSE compute
#                 deq_output = torch.mm(tensor_deq, weight.T).to(torch.float)
#             else:
#                 if n_bit > 6:
#                     tensor_deq = pseudo_quantize_int(weight, n_bit=n_bit, zero_point=False, q_group_size=group_size, is_input=False)
#                 else:
#                     tensor_deq = get_quant(weight, quant_grid, alpha=search_alpha, is_input=False)
#                 deq_output = torch.mm(input, tensor_deq.T).to(torch.float)

#             mse = (deq_output - org_output).pow(2).mean()

#             if mse < min_mse:
#                 min_mse = mse
#                 final_tensor = tensor_deq
#                 best_mode = mode
#                 best_alpha = search_alpha

#     if is_input:
#         self.input_quant_grid = quant_grid_set[best_mode]
#         self.input_alpha = best_alpha
#         quant_obj = 'input'
#     else:
#         self.weight_quant_grid = quant_grid_set[best_mode]
#         self.weight_alpha = best_alpha
#         quant_obj = 'weight'

#     print(f"layer: {layer_id}, tensor: {layer_name}, {quant_obj} quant, best mode: {best_mode}, mse: {min_mse}, alpha: {best_alpha}")

#     return final_tensor



class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, input_x, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                        
                # # RTN quant
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                
                # # sec quant
                # q = self.secquant(weight=w.unsqueeze(0),input_x=input_x, group_size=-1, n_bit=4).squeeze(0)
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')
        

def quant_with_scale(W, scales, quant_grid):
    labels = ((W / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    W_deq = quant_grid[labels] * scales
    return W_deq

def gptq_quant(
    weight, input_x, blocksize=32, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
):
    actorder = True
    
    quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    
    quant_grid = torch.tensor(quant_grid).to(dtype=torch.half).to(weight.device)
    
    W = weight.clone()
    W = W.float()
    
    columns = W.shape[1]

    tick = time.time()


    # compute Hessian
    
    H = torch.zeros((columns, columns), device=W.device)
    batch_size, seq_len, hidden_dim = input_x.shape
    
    inp = input_x.clone().detach().reshape((-1, input_x.shape[-1]))
    inp = inp.t()
    inp = math.sqrt(2 / batch_size) * inp.float()
    H += inp.matmul(inp.t())

    # H = self.H
    # del self.H
    
    
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # if static_groups:
    #     import copy
    #     groups = []
    #     for i in range(0, columns, groupsize):
            
    #         # find quantization parameters
    #         quantizer = copy.deepcopy(self.quantizer)
    #         quantizer.find_params(W[:, i:(i + groupsize)])
    #         groups.append(quantizer)

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        
        input_xx = input_x.clone().detach().reshape((-1, input_x.shape[-1]))[:, perm]
        
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=W.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        
        # find quantization param
        x = input_xx.reshape((-1, input_x.shape[-1]))
        _, _, _, scales = get_quant_weight_mxfp(W1.half(), input_x=x[:,i1*32:(i1+1)*32], quant_grid=quant_grid, zero_point=False, round_method="w_search")

        scales = scales.to(W.device)
        
        
        for i in range(count):
            w = W1[:, i]
            
            d = Hinv1[i, i]

            # if groupsize != -1:
            #     if not static_groups:
            #         if (i1 + i) % groupsize == 0:
            #             self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
            #     else:
            #         idx = i1 + i
            #         if actorder:
            #             idx = perm[idx]
            #         self.quantizer = groups[idx // groupsize]
                    
            # # RTN quant
            q = quant_with_scale(w.unsqueeze(1), scales, quant_grid)
            
            # print(q.shape, flush=True)
            # print(scales.shape, flush=True)
            # print(w.shape, flush=True)
            # exit()
            
            # # sec quant
            # q = self.secquant(weight=w.unsqueeze(0),input_x=input_x, group_size=-1, n_bit=4).squeeze(0)
            
            q = q.squeeze(1)
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    torch.cuda.synchronize()

    if actorder:
        Q = Q[:, invperm]

    weight = Q.reshape(weight.shape).to(weight.dtype)
    if torch.any(torch.isnan(weight)):
        logging.warning('NaN in weights')
        # import pprint
        # pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
        raise ValueError('NaN in weights')
    return weight, perm
    

def error_diffusion_quant(w, ed_quant_grid, calib_input):
    K = w.shape[1]
    N = w.shape[0]
    input = calib_input.reshape(-1, calib_input.shape[-1])
    
    M = input.shape[0]
    group_size = 32
    group_num = K / group_size
    
    w_quant = torch.zeros_like(w).to(w.device)
    
    # print((input == 0).to(torch.int32).sum(dim=0).tolist())
    # exit()
    
    
    # derive delta_O = delta_X * W
    
    q_x = torch.zeros_like(input).to(input.device)
    for i1 in range(0, K, group_size):
        i2 = min(i1 + group_size, K)
        q_x[:,i1:i2], _, _, _ = get_quant_act_mxfp(x=input[:,i1:i2], weight=None, quant_grid=ed_quant_grid, zero_point=False, round_method="up")
        
    delta_x = input - q_x
    delta_O_global = torch.mm(delta_x, w.T)
        
    U = torch.zeros(size=(M,N), dtype=w.dtype, device=w.device)
    
    for i1 in range(0, K, group_size):
        i2 = min(i1 + group_size, K)
        
        w_slice = w[:, i1:i2]
        x_slice = input[:, i1:i2]
        
        x_slice_q, _, _, _ = get_quant_act_mxfp(x=x_slice, weight=None, quant_grid=ed_quant_grid, zero_point=False, round_method="up")
        delta_x = x_slice - x_slice_q
        
        x_slice_q_pow = torch.pow(x_slice_q.to(torch.float32), 2)
        x_slice_q_squ_sum = x_slice_q_pow.sum(dim=0, keepdim=True)
        
        zero_num = (x_slice_q_squ_sum == 0).to(torch.int32).sum(dim=1).item()
        
        # delta_O = torch.mm(delta_x, w_slice.T)
        E = U + delta_O_global / group_num
        # E = delta_O_global / group_num + delta_O + U
        
        l_update = E / (group_size - zero_num)
        
        
        assert torch.isinf(x_slice_q_squ_sum).to(torch.int32).sum(0).sum(0) == 0, "overflow!"
        assert torch.isnan(x_slice_q_squ_sum).to(torch.int32).sum(0).sum(0) == 0, "x_square NAN!"
        
        
        # print(x_slice_q_squ_sum)

        
        x_slice_norm = x_slice.to(torch.float32) / (x_slice_q_squ_sum)

        # print(x_slice_q_squ_sum)
        # print(x_slice_norm[:,13].tolist())
        
        
        if torch.isinf(x_slice_norm).to(torch.int32).sum(0).sum(0) != 0:    
            x_slice_norm = torch.where(torch.isinf(x_slice_norm), torch.zeros_like(x_slice_norm).to(x_slice_norm.device), x_slice_norm)
        if torch.isnan(x_slice_norm).to(torch.int32).sum(0).sum(0) != 0:
            x_slice_norm = torch.where(torch.isnan(x_slice_norm), torch.zeros_like(x_slice_norm).to(x_slice_norm.device), x_slice_norm)
        
        
        # print(x_slice_norm[:,13].tolist())
        
        assert torch.isnan(x_slice_norm).to(torch.int32).sum(0).sum(0) == 0, "x_norm NAN"
        
        # print(torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32)))
        w_slice_new = w_slice + torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32)).to(torch.float16)
        # if i1==5024:
        #     # print(torch.isnan(torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32))).any())
        #     print(torch.isinf(l_update).any())
        #     print(torch.isinf(x_slice_norm).any())
        #     tmp = torch.mm(l_update.T.to(torch.float32), x_slice_norm.to(torch.float32))
        #     print(torch.isinf(tmp.to(torch.float16)).any())
        #     print(tmp.amax(dim=0).amax(dim=0))
        #     max_idx = torch.argmax(tmp)
        #     print(torch.argmax(tmp))
            
        #     print(tmp[max_idx//tmp.shape[1]][max_idx%tmp.shape[1]])
        #     print(w_slice[max_idx//tmp.shape[1]][max_idx%tmp.shape[1]])
        #     print(torch.isinf(w_slice_new).any())
        #     print("\n")
        assert torch.isnan(w_slice_new).any() == False, f"{i1} weight NAN"
        # print(w_slice_new)
        
        w_slice_quant, _, _, _ = get_quant_weight_mxfp(w_slice_new, quant_grid=ed_quant_grid, input_x=x_slice_q, zero_point=False, round_method="w_search")
        
        # U = U + torch.mm(x_slice_q, (w_slice - w_slice_quant).T) + delta_O_global / group_num
        U = U + torch.mm(x_slice_q, (w_slice - w_slice_quant).T) + delta_O_global / group_num
        
        if torch.isnan(w_slice_quant).any() or torch.isinf(w_slice_quant).any():
            print(torch.isnan(w_slice_quant).any())
            print(torch.isinf(w_slice_quant).any())
            print(i1)
        
        w_quant[:,i1:i2] = w_slice_quant
        
    origin_output = torch.mm(input, w.T)
    ed_output = torch.mm(q_x, w_quant.T)
    
    normal_deq_weight = torch.zeros_like(w).to(w.device)
    
    for i1 in range(0, K, group_size):
        i2 = min(i1 + group_size, K)
        # deq_weight[i*N : (i+1)*N, :], _, up_ratio = get_quant_weight_mxfp(weight[i*N : (i+1)*N, :], input_x=input[i*M : (i+1)*M, :], quant_grid=self.quant_grid, zero_point=False, round_method="up")
        normal_deq_weight[:,i1:i2], _, up_ratio, _ = get_quant_weight_mxfp(w[:,i1:i2], input_x=input[:,i1:i2], quant_grid=ed_quant_grid, zero_point=False, round_method="w_search")
        
    normal_output = torch.mm(q_x, normal_deq_weight.T)
    
    normal_mse = (normal_output - origin_output).pow_(2).mean(dim=0).mean(dim=0)
    ed_mse = (ed_output - origin_output).pow_(2).mean(dim=0).mean(dim=0)
    
    
    
    return w_quant, normal_mse, ed_mse
        
        
        
        
        
        
    
    
    

@torch.no_grad()
def get_quant_weight_mxfp(w, quant_grid, input_x=None, zero_point=True, q_group_size=-1, pos_value=None, round_method="rtn", error=None):
    '''
    return : dequantized weight, mse?
    '''
    quant_grid = quant_grid.to(w.device)
    
    # round_method = "normal"
    

    max_val = w.abs().amax(dim=1, keepdim=True)

    if pos_value is None or pos_value == True:
        max_quant_val = max(quant_grid)
    elif pos_value == False:
        max_quant_val = abs(min(quant_grid))
    else:
        raise NotImplementedError 
    
    # Compute the scaling factor
    
    # RTN rounding
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_rtn = torch.pow(2, exp)
    
    # All round down
    scales_down = torch.pow(2, (torch.floor(torch.log2(max_val/max_quant_val))))
    
    # # All round up
    scales_up = torch.pow(2, (torch.ceil(torch.log2(max_val/max_quant_val))))
    
    
    
    zeros = 0
        
    # Normal rounding
    scales_n = max_val / max_quant_val
        
    if round_method == "rtn":
        scales = scales_rtn
    elif round_method == "up":
        scales = scales_up
    elif round_method == "down":
        scales = scales_down
    elif round_method == "nvfp":
        scales = scales_n.view(torch.short)
        hi = scales & 0xFF80
        r = scales & 0x0040
        
        scales = hi + r * 2
        
        scales = scales.view(torch.half)
    elif round_method == "w_search" and input_x != None:
        labels_up = (((w + zeros) / scales_up).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        labels_down = (((w + zeros) / scales_down).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        
        w_deq_up = quant_grid[labels_up] * scales_up - zeros
        w_deq_down = quant_grid[labels_down] * scales_down - zeros
        
        input_g = input_x.reshape(-1, 32)
        
        # deq_input_g, _, _, _ = get_quant_act_mxfp(x=input_g, weight=None, quant_grid=quant_grid, zero_point=False, round_method="up")
        
        out_origin = torch.mm(input_g, w.T)
        
        # out_up = torch.mm(input_g, w_deq_up.T)
        # out_down = torch.mm(input_g, w_deq_down.T)
        
        out_up = torch.mm(input_g, w_deq_up.T)
        out_down = torch.mm(input_g, w_deq_down.T)
        
        if error is None:
            up_mse = (out_origin - out_up).pow(2).mean(dim=0)
            down_mse = (out_origin - out_down).pow(2).mean(dim=0)
        else:
            up_mse = (out_up - out_origin + error).pow(2).mean(dim=0)
            down_mse = (out_down - out_origin + error).pow(2).mean(dim=0)
        
        # 1 for down, 0 for up
        mask = (((up_mse - down_mse) > 0).to(torch.int32)).unsqueeze(1)
        
        up_ratio = 1 - mask.sum() / mask.shape[0]
        
        scales = scales_down * mask + scales_up * (1 - mask)
    elif round_method == "normal":
        scales = scales_n
        

    labels = (((w + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    w_deq = quant_grid[labels] * scales - zeros

    
    quant_mse = (w_deq-w).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # if get_labels:
    #     return w_deq, quant_mse_sum, labels, quant_grid * scales
    # else:
    if round_method == "w_search":
        return w_deq, quant_mse_sum, up_ratio, scales
    else:
        return w_deq, quant_mse_sum, 1, scales
    


@torch.no_grad()
def get_quant_act_mxfp(x, weight=None, zero_point=True, q_group_size=-1, pos_value=None, round_method="rtn", x_clip_r=1.0):
    '''
    return : dequantized weight, mse?
    '''
    
    quant_grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                          -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
    quant_grid = torch.tensor(quant_grid).to(dtype=torch.half)
    
    # print(x.shape, flush=True)
    quant_grid = quant_grid.to(x.device)
    
    max_val = x.abs().amax(dim=1, keepdim=True)

    if pos_value is None or pos_value == True:
        max_quant_val = max(quant_grid)
    elif pos_value == False:
        max_quant_val = abs(min(quant_grid))
    else:
        raise NotImplementedError 
    
    # Compute the scaling factor
    
    # Normal rounding
    scales_n = max_val / max_quant_val
    
    # RTN rounding
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_rtn = torch.pow(2, exp)
    
    # All round down
    scales_down = torch.pow(2, (torch.floor(torch.log2(max_val/max_quant_val))))
    
    # # All round up
    scales_up = torch.pow(2, (torch.ceil(torch.log2(max_val * x_clip_r/max_quant_val))))
    
    
    zeros = 0
        
    if round_method == "rtn":
        scales = scales_rtn
    elif round_method == "up":
        scales = scales_up
    elif round_method == "down":
        scales = scales_down
    elif round_method == "x_search" and weight != None:
        labels_up = (((x + zeros) / scales_up).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        labels_down = (((x + zeros) / scales_down).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        
        x_deq_up = quant_grid[labels_up] * scales_up - zeros
        x_deq_down = quant_grid[labels_down] * scales_down - zeros
        
        
        out_origin = torch.mm(x, weight.T)
        
        out_up = torch.mm(x_deq_up, weight.T)
        out_down = torch.mm(x_deq_down, weight.T)
        
        up_mse = (out_origin - out_up).pow(2).mean(dim=1)
        down_mse = (out_origin - out_down).pow(2).mean(dim=1)
        
        # 1 for down, 0 for up
        mask = (((up_mse - down_mse) > 0).to(torch.int32)).unsqueeze(1)
        
        up_ratio = 1 - mask.sum() / mask.shape[0]
        # print(mask.shape, flush=True)
        # print(scales_down.shape, flush=True)
        
        scales = scales_down * mask + scales_up * (1 - mask)
        
    elif round_method == "bit_op_rtn":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        r_scales_bi = scale_s_e.view(torch.half)
        scales = r_scales_bi
        
    elif round_method == "bit_op_down":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        
        # scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        sign_bit = (max_val_bi & 0x0200) << 1
        r_scales_bi = (scale_s_e - (0x0400 - sign_bit)).view(torch.half)
        
        scales = r_scales_bi
        
    elif round_method == "bit_op_up":
        max_val_bi = max_val.view(torch.short)
        scales_bi = max_val_bi - 0x0800
        
        # scales_bi = max_val_bi - 0x0800
        scale_s_e = scales_bi & 0xFC00
        
        sign_bit = (max_val_bi & 0x0200) << 1
        r_scales_bi = (scale_s_e + sign_bit).view(torch.half)
        
        scales = r_scales_bi
        
        # print(f"Bit op scale up : {scales}", flush=True)
        # print(f"Normal scale up : {scales_up}", flush=True)
        # print(f"Scaling factor error : {(scales_up - scales).T} \n", flush=True)
        
    elif round_method == "normal":
        scales = scales_n
        
    # print(x.shape, flush=True)
    labels = (((x + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    x_deq = quant_grid[labels] * scales - zeros
    
    
    # deal with zero channels
    
    # channel_sum = x_deq.sum(dim=0, keepdim=True)
    # zero_mask = (channel_sum == 0).to(torch.int32)
    
    # # assert (zero_mask == 1).any() == False, f"zero input channel, {zero_mask.tolist()}"
    # x_deq = x_deq + zero_mask * x 
    
    quant_mse = (x_deq - x).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # if get_labels:
    #     return w_deq, quant_mse_sum, labels, quant_grid * scales
    # else:
    if round_method == "x_search":
        return x_deq, quant_mse_sum, up_ratio, labels
    else:
        return x_deq, quant_mse_sum, 1, labels


class MCQ_Linear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, ant_config, layer_id, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size 
        self.ant_config = ant_config

        self.layer_id = layer_id
        self.layer_name = layer_name
        
        # GPTQ option
        self.gptq = False
        self.gptq_perm = None
        
        # Error diffusion option
        self.erd = False
        
        # Permutation option
        self.permute = False
        self.idx = None
        
        
        # activation clipping parameter
        self.search_tw = False
        self.x_clip_tw = 1.0
        
        # chunk-wise activation clipping
        self.search_cw = False
        self.x_clip_cw = []
        
        
        self.w_rounding_method = None
        self.x_rounding_method = None

        # ANT param
        self.weight_quant_grid = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_alpha = -1
        
        self.quant_grid = None

        assert self.in_features % self.group_size == 0

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None
            
        self.skip_q_flag = False
        self.store_q_flag = True
        
        self.id = -1

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None, quant_mode=None, mxfp_config=None):

        mxfp_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading sd
            return mxfp_linear

        mxfp_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            mxfp_linear.bias = linear.bias.clone().half()
            
        quant_mode = mxfp_config["mx_type"]
        mxfp_linear.w_rounding_method = mxfp_config["w_round"]
        mxfp_linear.x_rounding_method = mxfp_config["x_round"]

        if w_bit > 6:
            mxfp_linear.ant_config['ant_mode'] = 'int'
            
        if quant_mode == "fp4_e2m1":
            quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        elif quant_mode == "fp6_e2m3":  
            quant_grid = [0.0,  0.125,  0.25,  0.375,  0.5,  0.625,  0.75,  0.875,
                                      -0.0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875,
                                      1.0,  1.125,  1.25,  1.375,  1.5,  1.625,  1.75,  1.875,
                                     -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
                                      2.0,  2.25,  2.5,  2.75,  3.0,  3.25,  3.5,  3.75,
                                     -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
                                      4.0,  4.5,  5.0,  5.5,  6.0,  6.5,  7.0,  7.5,
                                     -4.0, -4.5,  -5.0, -5.5, -6.0, -6.5, -7.0, -7.5]
        elif quant_mode == "fp6_e3m2":
            quant_grid = [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75,
                                      1.0, -1.0, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75,
                                      2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5,
                                      4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0,
                                      8.0, -8.0, 10.0, -10.0, 12.0, -12.0, 14.0, -14.0,
                                      16.0, -16.0, 20.0, -20.0, 24.0, -24.0, 28.0, -28.0,
                                      32.0, -32.0, 40.0, -40.0, 48.0, -48.0, 56.0, -56.0,
                                      64.0, -64.0, 80.0, -80.0, 96.0, -96.0, 112.0, -112.0]
            
        elif quant_mode == "int4":
            quant_grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                          -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
        mxfp_linear.quant_grid = torch.tensor(quant_grid).to(dtype=torch.half)
        
        
        
        return mxfp_linear
    
    @torch.no_grad()
    def forward(self, x):
        
        error_compensation = True
            
        out_shape = x.shape[:-1] + (self.out_features, )
        
        input = x.reshape(-1, x.shape[-1])

        if self.weight_quant_grid is None:
            
            inlier_g_size = 128
            outlier_g_size = 8

            org_w_shape = self.weight.shape
            
            N = org_w_shape[0]
            K = org_w_shape[1]
            M = input.shape[0]
            
            
            # weight = self.weight.reshape(-1, self.group_size)
            
            weight = self.weight
            
            deq_weight = weight.clone().detach()
            deq_input = input.clone().detach()
            
            error_list = [] # average error of each chunk
            
            inlier, outlier, inlier_mask = split_by_3sigma(weight)
            deq_inlier = inlier.clone().detach()
                

            # inlier quantization
            for i1 in range(0, K, inlier_g_size):
                i2 = min(i1 + inlier_g_size, K)
                # deq_weight[i*N : (i+1)*N, :], _, up_ratio = get_quant_weight_mxfp(weight[i*N : (i+1)*N, :], input_x=input[i*M : (i+1)*M, :], quant_grid=self.quant_grid, zero_point=False, round_method="up")
                deq_inlier[:,i1:i2], _, up_ratio, _ = get_quant_weight_mxfp(inlier[:,i1:i2], input_x=None, quant_grid=self.quant_grid, zero_point=False, round_method="rtn")
                
            assert torch.isnan(deq_weight).to(torch.int32).sum(0).sum(0) == 0, "weight NAN"

            # outlier count
            outlier_org = outlier.shape
            outlier = outlier.reshape(-1, outlier_g_size)
            non_zero_counts = torch.count_nonzero(outlier, dim=1)
            
            # outlier quantization
            outlier_quant_grid = torch.tensor([0.0,  0.125,  0.25,  0.375,  0.5,  0.625,  0.75,  0.875,
                                      -0.0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875,
                                      1.0,  1.125,  1.25,  1.375,  1.5,  1.625,  1.75,  1.875,
                                     -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
                                      2.0,  2.25,  2.5,  2.75,  3.0,  3.25,  3.5,  3.75,
                                     -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
                                      4.0,  4.5,  5.0,  5.5,  6.0,  6.5,  7.0,  7.5,
                                     -4.0, -4.5,  -5.0, -5.5, -6.0, -6.5, -7.0, -7.5]).to(dtype=torch.half)
            
            deq_outlier, _, up_ratio, _ = get_quant_weight_mxfp(outlier, input_x=None, quant_grid=outlier_quant_grid, zero_point=False, round_method="rtn")
                
            
            outlier = deq_outlier.reshape(outlier_org)
            
            sparsed_num = non_zero_counts * 2
            
            
            
            # N:M sparsity
            inlier = deq_inlier.reshape(-1, outlier_g_size)
            
            _, sorted_indices = inlier.abs().sort(dim=1)
            row_indices = torch.arange(inlier.size(0), device=inlier.device).unsqueeze(1).repeat(1, inlier.size(1))
            inlier_zero_mask = torch.arange(inlier.size(1), device=inlier.device).unsqueeze(0) < sparsed_num.unsqueeze(1)  
            zero_indices = (row_indices[inlier_zero_mask], sorted_indices[inlier_zero_mask])
            
            
            inlier[zero_indices] = 0.0
            
            
            
            
            
            
            inlier = inlier.reshape(outlier_org)
            
            deq_weight = inlier * inlier_mask + outlier * (~inlier_mask)
            deq_weight = deq_weight.reshape(org_w_shape)
            
            
            # deq_weight = ant_quant(self, self.w_bit, self.weight, input, self.ant_config, self.group_size, self.layer_id, self.layer_name, is_input=False)
            # deq_input = ant_quant(self, self.w_bit, deq_weight, input, self.ant_config, self.group_size, self.layer_id, self.layer_name, is_input=True)
            
            # deq_input, _, _, _ = get_quant_act_mxfp(x=input, weight=None, quant_grid=self.quant_grid, zero_point=False, round_method="up", x_clip_r=self.x_clip_tw)
            
            # deq_input = input
            
            # quantize weight only once
            if self.store_q_flag:
                self.weight = deq_weight
                # deq_input = input
                
                # deq_input, _, _ = get_quant_act_mxfp(x=input, weight=None, quant_grid=self.quant_grid, zero_point=False, round_method="up")
                
                self.weight_quant_grid = self.quant_grid
            

        # quantize input based on the selected data type and alpha
        else:
            org_inp_shape = input.shape
            
            M = org_inp_shape[0]
            K = org_inp_shape[1]
            N = self.weight.shape[0]
            # print(org_inp_shape)
            
            # input = input.reshape(-1, self.group_size)
            
            # if self.w_bit > 6:
            #     deq_input = pseudo_quantize_int(input.view(-1), n_bit=self.w_bit, zero_point=False, q_group_size=self.group_size, alpha=self.input_alpha, is_input=True)
            # else:
            #     deq_input = get_quant(input.view(-1), self.input_quant_grid, alpha=self.input_alpha, is_input=True)
            
            # deq_input = torch.zeros_like(input)
            
            ratio_list = []
            

            
            # Round up / RTN
            
            deq_input = input.reshape(-1, self.group_size)
            
            # deq_input = input
            
            if self.gptq and self.gptq_perm!=None:
                deq_input = deq_input[:, self.gptq_perm]
            
            
            assert torch.isnan(deq_input).to(torch.int32).sum(0).sum(0) == 0, "origin activation NAN"
            # options of rounding method: rtn, up, down, x_search, bit_op_up, bit_op_rtn, bit_op_down, normal

            deq_input, _, _, labels = get_quant_act_mxfp(x=deq_input, weight=None, zero_point=False, round_method="rtn", x_clip_r=1.0)



            if self.gptq and self.gptq_perm!=None:
                invperm = torch.argsort(self.gptq_perm)
                deq_input = deq_input[:, invperm]
            
            # max_exi_num = count_exist(labels, self.quant_grid, 6.0)
            # max_non_exi_ratio = 1 - max_exi_num / labels.shape[0]
            # print(f"Layer {self.layer_id}, name {self.layer_name}, {max_non_exi_ratio} max value does not exist")
            
            assert torch.isnan(deq_input).to(torch.int32).sum(0).sum(0) == 0, "dequant activation NAN"
            
            # deq_input, _ = get_quant_weight(input, quant_grid=self.quant_grid, mode=None, zero_point=False, q_group_size=-1)


            deq_input = deq_input.reshape(org_inp_shape)

        out = F.linear(deq_input, self.weight)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    
def split_by_3sigma(tensor):

    mean = torch.mean(tensor)
    std = torch.std(tensor)
    
    
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std

    mask = (tensor >= lower_bound) & (tensor <= upper_bound)
    
    inliers = tensor * mask
    outliers = tensor * (~mask)  

    
    return inliers, outliers, mask