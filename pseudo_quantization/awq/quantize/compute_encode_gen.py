import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

ant_nf_set = {
    'float': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, -6.0, 6.0],
    'nf4': [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0],
    'int': [-0.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    'flint': [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, -6.0, 6.0],
    'pot': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0, -64.0, 64.0]}

def encode_gen(w_bit, return_list=False):
    """
    Generate the computable codebook from the index list
    Computable codebook: a*index + 2^index * b
    """
    coefficient_list = []
    # 选定系数 a
    for coefficient in range(0, 128, 10):
        coefficient_list.append(coefficient)
    # supply some specific data type, merge them after removing duplicates
    supply_list = [0, 5, 17, 18, 20, 30, 40, 50, 70]
    merged_list = list(set(coefficient_list + supply_list))

    codebook_dict = {}
    b = 1
    for coefficient in merged_list:
    # for coefficient in coefficient_list:
        codebook_list = []
        # for item in range((2 ** w_bit) // 2 - 1):
        for item in range(2 ** w_bit):
            # 0~15 -> -7~8
            index = item - ((2 ** (w_bit-1)) - 1)
            if index < 0:
                index = (-index)
                codebook_list.append(-(coefficient * index + (2 ** index * b)))
            elif index == 0:
                codebook_list.append(0.)
            elif index > 0 and index < (2 ** w_bit // 2):
                codebook_list.append(coefficient * index + (2 ** index * b))
            elif index == (2 ** w_bit // 2):
                codebook_list.append(-0.)

        codebook_list = torch.tensor(codebook_list).to(dtype=torch.half)
        codebook_list.sort()
        
        # Normalization
        codebook_list = codebook_list / codebook_list.max()
        # to list if need
        if return_list:
            codebook_list = codebook_list.tolist()

        codebook_dict[f"coefficient_{coefficient}"] = codebook_list

    # codebook_dict['int'] = torch.tensor(ant_nf_set['int'])
    return codebook_dict

def get_quant_weight(w, quant_grid, q_group_size=-1, alpha=1.0, get_labels=False):

    quant_grid = quant_grid.to(w.device)
    max_val = w.amax(dim=1, keepdim=True)

    max_quant_val = max(abs(quant_grid))

    # print(max_val, max_quant_val)
    # Compute the scaling factor
    scales = ((max_val * alpha) / max_quant_val).half()
    print(max_val, max_quant_val, scales)

    zeros = 0
    labels = (((w + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    w_deq = quant_grid[labels] * scales - zeros

    quant_mse = (w_deq-w).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    w_deq = w_deq.half()
    if get_labels:
        return w_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return w_deq, quant_mse_sum

def match_type_search(codebook_dict, ant_nf_set, top_k=1):
    """
    对于ant_nf_set中的每种数据类型 找到与之分布最相近的top_k个codebook
    """
    matched_codebooks = {}
    def calculate_mse(list1, list2):
        """
        计算两个列表之间的均方误差（MSE）
        """
        # 确保两个列表长度相同
        if len(list1) != len(list2):
            return float('inf')
        return np.mean((np.array(list1) - np.array(list2)) ** 2)
    
    # 遍历ant_nf_set中的每种数据类型
    for data_type, ant_nf_list in ant_nf_set.items():
        mse_list = []
        # 比较与codebook_dict中的每个codebook
        for codebook_name, codebook_list in codebook_dict.items():
            mse = calculate_mse(ant_nf_list, codebook_list)
            mse_list.append((codebook_name, mse))
        
        # 根据MSE排序并选出最小的top_k个
        top_matched = sorted(mse_list, key=lambda x: x[1])[:top_k]
        # 保存匹配结果
        for match in top_matched:
            matched_name = f"{data_type}_codebook_{match[0]}"
            matched_codebooks[matched_name] = codebook_dict[match[0]]
    
    return matched_codebooks
    

if __name__ == '__main__':
    # quant_verify()
    # match_type_search()
    
    # draw 分布情况
    # codebook_dict = encode_gen(4, return_list=True)
    codebook_dict = encode_gen(4, return_list=False)
    
    print(codebook_dict)
    # draw_distribution(codebook_dict)
    # draw_distribution_single(codebook_dict)

    # match_codebook = match_type_search(codebook_dict, ant_nf_set)
    # print(match_codebook)