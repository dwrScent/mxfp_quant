import torch
from collections import defaultdict
def calculate_max_error(tensor_value, tensor_deq, q_group_size=-1):
    if q_group_size > 0:
        tensor_value = tensor_value.reshape(-1, q_group_size)
        tensor_deq = tensor_deq.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    max_val_deq = tensor_deq.abs().amax(dim=1, keepdim=True)

    error_mean = (max_val - max_val_deq).abs().mean()
    error_max = (max_val - max_val_deq).abs().max()
    error_min = (max_val - max_val_deq).abs().min()

    relative_error = (max_val - max_val_deq).abs() / (max_val + 1e-8)  # 防止除以零
    relative_error_mean = relative_error.mean()
    relative_error_max = relative_error.max()
    relative_error_min = relative_error.min()

    mse_error = ((max_val - max_val_deq) ** 2).mean()

    print(f'Abs Error, mean: {error_mean}, max: {error_max}, min: {error_min}, MSE: {mse_error}')
    print(f'Relative Error, mean: {relative_error_mean}, max: {relative_error_max}, min: {relative_error_min}')

# 全局统计字典，用于统计所有 tensor 的 exp 出现次数
exp_stats = defaultdict(lambda: defaultdict(int))

def calculate_scale_range(tensor_value, quant_grid, layer_id, layer_name, q_group_size, is_input):
    org_shape = tensor_value.shape

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    # print(tensor_value.dtype)
    quant_grid = quant_grid.to(tensor_value.device)
    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_quant_val = max(quant_grid)
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = torch.pow(2, exp)

    # 统计 exp 数值的频率
    unique_exp, counts = exp.unique(return_counts=True)
    counts = counts.float() / 100000.

    assert torch.all(counts > 0), "All counts should be greater than 0 after scaling."

    # 将统计结果更新到全局字典中
    unique_exp = unique_exp.cpu().tolist()
    counts = counts.cpu().tolist()
    key_name = f"weight_{layer_name}" if not is_input else f"input_{layer_name}"
    layer_exp_stats = exp_stats[key_name]
    for e, c in zip(unique_exp, counts):
        layer_exp_stats[e] += c

    # 输出当前 layer 的 exp 统计结果
    total_counts = sum(layer_exp_stats.values())
    exp_values = torch.tensor(list(layer_exp_stats.keys()))
    exp_counts = torch.tensor(list(layer_exp_stats.values()))
    percentages = (exp_counts / total_counts) * 100

    if layer_id == 31:
        print(f"Layer: {key_name}")
        for e, c, p in zip(exp_values, exp_counts, percentages):
            print(f"  exp: {e.item()}, count: {c.item()}, percentage: {p.item():.4f}%")

outlier_stats = defaultdict(lambda: defaultdict(int))
def calculate_outlier_exp(tensor_value, quant_grid, layer_id, layer_name, q_group_size, is_input):
    org_shape = tensor_value.shape
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    # Extract maximum absolute value and clamp to avoid zero-division
    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)

    tensor_exp = tensor_value.view(torch.int16)
    tensor_exp = (tensor_exp >> 10) & 0x1F
    max_val_exp = max_val.view(torch.int16)
    max_val_exp = (max_val_exp >> 10) & 0x1F
    outlier_mask = (tensor_exp == max_val_exp)

    # 统计 tensor_value 中大于 exp_value 的元素数量
    # tensor_outlier_num = (tensor_value.abs() > exp_value).sum(dim=1)
    tensor_outlier_num = outlier_mask.to(dtype=int).sum(dim=1)

    # 分类统计 outlier 数量
    key_name = f"weight_{layer_name}" if not is_input else f"input_{layer_name}"
    layer_outlier_stats = outlier_stats[key_name]

    # 统计分类
    for i in range(1, 11):
        layer_outlier_stats[f"num_{i}"] += (tensor_outlier_num == i).sum().item() / 100000.
    layer_outlier_stats["num_>10"] += (tensor_outlier_num > 10).sum().item() / 100000.

    # 计算总数，用于求百分比
    total_count = sum(layer_outlier_stats.values())

    # 输出统计结果，包括数量和百分比
    if layer_id == 31:
        print(f"Layer: {key_name}")
        for category, count in layer_outlier_stats.items():
            percentage = (count / total_count) * 100 if total_count > 0 else 0.0
            print(f"  {category}: {count:.6f}, {percentage:.2f}%")
    
