import torch
from hif4_quant_func import get_quant_hifes
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)
try:
    from .nvesm2_quant_func import get_quant_nvesm2_hw
except ImportError:
    from nvesm2_quant_func import get_quant_nvesm2_hw

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_EPS = 2 ** (-9)
FLOAT8_E4M4_EPS = 2 ** (-10)
FLOAT8_E4M3_MAX = 448.0
LEVEL_2_MAX = 7.05


def draw_mse_comp_with_gaussian():

    import torch
    import matplotlib.pyplot as plt

    # 存储结果用于绘图
    x_axis = []

    mse1, mse2, mse3, mse4 = [], [], [], []
    mse5, mse6, mse7, mse8 = [], [], [], []
    entropy1, entropy2, entropy3, entropy4 = [], [], [], []
    entropy5, entropy6, entropy7, entropy8 = [], [], [], []

    # 采样次数
    num_samples = 100


    for j in range(1, 34):
        x_val = j / 2
        x_axis.append(x_val) # 修正1：填充横坐标
        sigma = 0.01 * 2 ** (j / 2)
        m1, m2, m3, m4 = 0.0, 0.0, 0.0, 0.0
        m5, m6, m7, m8 = 0.0, 0.0, 0.0, 0.0
        e1, e2, e3, e4 = 0.0, 0.0, 0.0, 0.0
        e5, e6, e7, e8 = 0.0, 0.0, 0.0, 0.0
        # 计算当前信号的理论方差，用于归一化
        # 因为 b = randn * (sigma^2)，其方差是 (sigma^2)^2
        signal_variance = sigma ** 2

        for i in range(num_samples):
            b = torch.randn(8192) * sigma 
            # 假设这些函数已经在你的命名空间中定义
            res1 = get_quant_nvfp(b, 16)
            res2 = get_quant_nves(b, 16)
            res3 = get_quant_nvint4(b, 16)
            res4 = get_quant_nvesm2(b, 16)
            res5 = get_quant_nvintesm2(b, 16)
            res6 = get_quant_nvesm2_hw(b, 16)
            res7 = get_quant_nvesm2_kur(b, 16)

            # 累加 MSE
            m1 += (res1 - b).pow(2).mean().item()
            m2 += (res2 - b).pow(2).mean().item()
            m3 += (res3 - b).pow(2).mean().item()
            m4 += (res4 - b).pow(2).mean().item()
            m5 += (res5 - b).pow(2).mean().item()
            m6 += (res6 - b).pow(2).mean().item()
            m7 += (res7 - b).pow(2).mean().item()
            # m8 += (res8 - b).pow(2).mean().item()
            # e1 += entropy(res1, num_bins=512).item()
            # e2 += entropy(res2, num_bins=512).item()
            # e3 += entropy(res3, num_bins=512).item()
            # e4 += entropy(res4, num_bins=512).item()
            # e5 += entropy(res5, num_bins=512).item()
            # e6 += entropy(res6, num_bins=512).item()
            # e7 += entropy(res7, num_bins=512).item()
            # e8 += entropy(res8, num_bins=512).item()

        # 修正2：均值除以样本数，再除以信号方差实现归一化
        mse1.append((m1 / num_samples) / signal_variance)
        mse2.append((m2 / num_samples) / signal_variance)
        mse3.append((m3 / num_samples) / signal_variance)
        mse4.append((m4 / num_samples) / signal_variance)
        mse5.append((m5 / num_samples) / signal_variance)
        mse6.append((m6 / num_samples) / signal_variance)
        mse7.append((m7 / num_samples) / signal_variance)
        # mse8.append((m8 / num_samples) / signal_variance)
        # entropy1.append(e1 / num_samples)
        # entropy2.append(e2 / num_samples)
        # entropy3.append(e3 / num_samples)
        # entropy4.append(e4 / num_samples)
        # entropy5.append(e5 / num_samples)
        # entropy6.append(e6 / num_samples)
        # entropy7.append(e7 / num_samples)
        # entropy8.append(e8 / num_samples)


    # --- 绘图部分 ---
    plt.figure(figsize=(12, 8))

    # 使用线性坐标轴，因为已经归一化了，数值应该在可比范围内
    plt.plot(x_axis, mse1, label='NVFP (4.5)', marker='o', markersize=4)
    plt.plot(x_axis, mse2, label='NVES (4.75)', marker='s', markersize=4)
    plt.plot(x_axis, mse3, label='NVINT (4.5)', marker='^', markersize=4)
    plt.plot(x_axis, mse4, label='NVESM2 (4.75)', marker='x', markersize=4)
    plt.plot(x_axis, mse5, label='NVINTESM2 (4.75)', marker='D', markersize=6)
    plt.plot(x_axis, mse6, label='NVESM2_HW (4.75)', marker='v', markersize=4)
    plt.plot(x_axis, mse7, label='NEW', marker='*', markersize=4)
    # plt.plot(x_axis, mse8, label='NVESEM (4.625)', marker='P', markersize=4)


    plt.xlabel('x variance = 0.01*2^(x)')
    plt.ylabel('Normalized MSE (MSE / Variance)')
    plt.title('Normalized Quantization Error vs. Signal Range')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    # 保存并显示
    plt.savefig('dump/quant_mse_comparison.png')
    plt.show()


@torch.no_grad()
def draw_mse_comp_with_real_weight_per_tensor():

    import os
    import gc

    dump_dir = "dump"
    device = torch.device('cuda:0')

    files = [f for f in os.listdir(dump_dir) if f.endswith(".pt")]
    print(f"找到 {len(files)} 个文件，准备开始处理...")

    quant_method = [ "nvfp", "nves", "nvesm2", "nvint4", "nvintesm2", "nvesm" ]
    # only process last 1 of them
    files = files[:30]
    files = files[::2]
    mse_tensor = torch.zeros((len(quant_method), len(files)), dtype=torch.float32)

    for name in files:

        print(f"正在处理文件: {name}")
        file_path = os.path.join(dump_dir, name)

        x = torch.load("dump/" + name)  # [N, T, Cin]
        x = x.to(device)
        x = x.reshape(-1, x.shape[-1])

        for mode in quant_method:
            quant_func = QUANT_METHOD_MAP[mode]
            group_size = 16
            mse = (quant_func(x, group_size) - x).pow(2).mean().item()
            mse_tensor[quant_method.index(mode)][files.index(name)] = mse

        del x
        gc.collect()
        torch.cuda.empty_cache()

    # draw mse comp
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    x_axis = range(len(files))
    for i, mode in enumerate(quant_method):
        plt.plot(x_axis, mse_tensor[i].cpu(), label=mode, marker='o', markersize=4)
    plt.xlabel('File Index')
    plt.ylabel('MSE')
    plt.title('MSE Comparison with Real Weights')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('dump/mse_comparison_with_real_weight_per_tensor.png')


@torch.no_grad()
def draw_mse_comp_with_real_weight():

    import os
    import gc

    dump_dir = "dump"
    device = torch.device('cuda:0')

    files = [f for f in os.listdir(dump_dir) if f.endswith(".pt")]
    print(f"找到 {len(files)} 个文件，准备开始处理...")

    quant_method = [ "nvfp", "nves", "nvesm2", "nvesm2_hw", "nvint4", "nvintesm2", "nvesm" ]
    bits = ["(4.5)", "(4.75)", "(4.75)", "(4.75)", "(4.5)", "(4.75)", "(4.625)"]
    # only process the 32th of the files
    files = files[::35]
    block_num = 32
    mse_tensor = torch.zeros((len(quant_method), block_num), dtype=torch.float32)

    for name in files:

        print(f"正在处理文件: {name}")
        file_path = os.path.join(dump_dir, name)

        x = torch.load("dump/" + name)  # [N, T, Cin]
        x = x.to(device)
        x = x.reshape(-1, x.shape[-1])

        x = x.reshape(block_num, -1)

        for mode in quant_method:
            quant_func = QUANT_METHOD_MAP[mode]
            group_size = 16
            mse = (quant_func(x, group_size) - x).pow(2).mean(dim=1)
            mse_tensor[quant_method.index(mode)] = mse

        del x
        gc.collect()
        torch.cuda.empty_cache()

        # draw mse comp
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        x_axis = range(block_num)
        for i, mode in enumerate(quant_method):
            label = mode + " " + bits[i]
            plt.plot(x_axis, mse_tensor[i].cpu(), label=label, marker='o', markersize=4)
        plt.xlabel('Block Index')
        plt.ylabel('MSE')
        plt.title('MSE Comparison with Real Weights ' + name)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig('dump/mse_comparison_with_real_weight_' + name + '.png')


@torch.no_grad()
def fp16(tensor_value: torch.Tensor, group_size: int):
    return tensor_value


def float_value(exp_bit, man_bit):
    bias = (2 ** (exp_bit - 1)) - 1
    values = []
    min_to_zero = True
    subnormal = True
    for i in range(2**exp_bit):
        for j in range(2**man_bit):
            if min_to_zero:
                values.append(0.0)
                min_to_zero = False
            else:
                if subnormal:
                    values.append((2 ** (1 - bias)) * (j * 2 ** (-man_bit)))
                else:
                    values.append((2 ** (i - bias)) * (1 + j * 2 ** (-man_bit)))

        subnormal = False

    return values

def exp_man_value(exp_bit, man_bit):
    bias = -48
    values = []
    for i in range(2**exp_bit):
        for j in range(2**man_bit):
            values.append((2 ** (i - bias)) * (1 + j * 2 ** (-man_bit)))
    return values


# FP4_E2M1_GRID = torch.tensor(float_value(2, 1), device="cuda")
# FP6_E2M3_GRID = torch.tensor(float_value(2, 3), device="cuda")
FP4_E2M1_GRID = torch.tensor(float_value(2, 1))
FP6_E2M3_GRID = torch.tensor(float_value(2, 3))
FP8_E5M3_GRID = torch.tensor(float_value(5, 3))
FP8_E4M4_GRID = torch.tensor(float_value(4, 4))
E6M2_GRID = torch.tensor(exp_man_value(6, 2))

def quantize_to_grid(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    levels = levels.to(x.device)
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    odd_boundaries = boundaries[1::2]
    mask = torch.isin(x, odd_boundaries)
    x = x + 0.0000005 * mask  # round to even
    indices = torch.bucketize(x, boundaries)
    indices.clamp_(0, len(levels) - 1)

    quantized = levels[indices]
    return quantized, indices


def cast_to_fp4(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    x_quant, _ = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    return x_quant * sign


def cast_to_fp4_em(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    fp4, fp4_index = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    _, fp6_index = quantize_to_grid(x_abs, FP6_E2M3_GRID)
    # print("previous fp6:")
    # print(FP6_E2M3_GRID.to(x.device)[fp6_index])
    fp6_index.clamp_(min=fp4_index * 4 - 1, max=fp4_index * 4 + 2)
    fp6 = FP6_E2M3_GRID.to(x.device)[fp6_index]

    return fp4 * sign, fp6 * sign

def cast_to_E6M2(x: torch.Tensor):
    x = x.clamp(min=2 ** (-48) * 1.0, max=2 ** 15 * 1.5)
    E = torch.floor(torch.log2(x))
    return torch.round(x * 2 ** (-E + 2)) * 2 ** (E - 2)

@torch.no_grad()
def get_quant_mxfp(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)
    tensor_quant = cast_to_fp4(tensor_value / scales) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


def get_quant_mxem(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for mantissa in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    fp4, fp6 = cast_to_fp4_em(tensor_value / scales)

    tmp = fp4.reshape(-1, sub_group_size)
    outlier_mask = torch.zeros_like(tmp, dtype=tensor_value.dtype).to(
        tensor_value.device
    )

    _, indices = torch.topk(tmp.abs(), 1)
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, group_size)
    tensor_quant = (fp4 * (1 - outlier_group_mask) + fp6 * outlier_group_mask) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_mxes(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    range_ = range(-1, 2)
    for bias in range_:
        scales = torch.pow(2, exp + bias)
        sub_groups_per_group = group_size // sub_group_size
        # turn scales to (N_subgroups, 1)
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device
        )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1)
        cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq


@torch.no_grad()
def get_quant_nvfp(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / FLOAT4_E2M1_MAX
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    tensor_quant = cast_to_fp4(tensor_value / scales) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)




@torch.no_grad()
def get_quant_nves(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    )
    # print("org_scales:", scales)
    exp = torch.floor(torch.log2(scales))
    man_value = scales / torch.pow(2, exp)
    bias_mse = {}
    range_ = range(-1, 2)
    for bias in range_:
        # scales = torch.pow(2, exp + bias)

        scales = torch.pow(2, exp) * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
        sub_groups_per_group = group_size // sub_group_size
        # N_subgroups, 1
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [0, 0.03125, 0.0625, 0.09375], dtype=tensor_value.dtype, device=tensor_value.device
        )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)
        man_value_expanded = man_value.expand(-1, sub_groups_per_group).reshape(-1, 1).unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1) + scales_expanded * man_value_expanded
        # print(cand_scales, scales_expanded * man_value_expanded)
        cand_qval = cast_to_fp4(x_expanded / cand_scales / global_scale) * cand_scales * global_scale
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq


@torch.no_grad()
def get_quant_nvesem2(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale
    exp = torch.floor(torch.log2(scales))
    # exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    range_ = range(-1, 2)
    org_scales = scales
    for bias in range_:
        # scales = torch.pow(2, exp + bias)

        scales = org_scales * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
        sub_groups_per_group = group_size // sub_group_size
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device
        )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1)
        cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq
@torch.no_grad()
def get_quant_nvesm(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale
    exp = torch.floor(torch.log2(scales))
    # exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    # range_ = range(-1, 2)
    range_ = {0}
    org_scales = scales
    for bias in range_:
        # scales = torch.pow(2, exp + bias)

        scales = org_scales * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
        sub_groups_per_group = group_size // sub_group_size
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device
        )
        # ratios = torch.tensor(
        #     [1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device
        # )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1)
        cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq

E4M5_MAX = 2 ** 8 * 1.9735
E4M5_GRID = torch.tensor(float_value(4, 5))
@torch.no_grad()
def cast_to_E4M5(x: torch.Tensor):
    x_quant, _ = quantize_to_grid(x, E4M5_GRID)
    return x_quant


@torch.no_grad()
def get_quant_nvem(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 4  # extra 2 bit for mantissa in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / FLOAT4_E2M1_MAX
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    fp4, fp6 = cast_to_fp4_em(tensor_value / scales)
    # print(fp4, "\n", fp6)

    tmp = fp4.reshape(-1, sub_group_size)
    outlier_mask = torch.zeros_like(tmp, dtype=tensor_value.dtype).to(
        tensor_value.device
    )

    _, indices = torch.topk(tmp.abs(), 1)
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, group_size)
    tensor_quant = (fp4 * (1 - outlier_group_mask) + fp6 * outlier_group_mask) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


def nvfp_anal(x: torch.Tensor, group_size=16):
    import torch
    import numpy as np
    x = x.reshape(-1, group_size)
    max_ = 6.0
    tensor_value = x.float()
    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / max_
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale
    x = x / scales

    total_hist = None
    # x = x[::8, :]

    # bin_edges 假设是等间距的，例如从 0 到 6 分 100 份
    num_bins = 100
    min_val, max_val_bin = 0.0, 7.0 # 根据你的需求设置边界
    draw_histogram(x, min_val, max_val_bin, num_bins, "nvfp")


@torch.no_grad()
def get_quant_nvint4(tensor_value: torch.Tensor, group_size: int):
    
    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / 7.0
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    tensor_quant = torch.clamp(torch.round(tensor_value / scales), min=-7.0, max=7.0) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_nvintesm2(tensor_value: torch.Tensor, group_size=16):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / 7.0
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    sub_group_size = 8
    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # ratio = torch.tensor([1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device)
    ratio = torch.tensor([1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device)
    scales = scales.reshape(-1, 1).expand(-1, group_size // sub_group_size).reshape(-1, 1)
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)
    cand_scales = scales_expanded * ratio.view(1, 1, -1)
    cand_qval = torch.clamp(torch.round(x_expanded / cand_scales), min=-7.0, max=7.0) * cand_scales
    mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
    best_ratio_idx = mse_per_ratio.argmin(dim=1)
    row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
    best_dqval = cand_qval[row_idx, :, best_ratio_idx]

    tensor_quant = best_dqval

    return tensor_quant.reshape(org_shape).to(org_dtype)



def entropy(x: torch.Tensor, num_bins: int = 256):
    hist = torch.histc(x, bins=num_bins, min=x.min().item(), max=x.max().item())
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    ent = -torch.sum(prob * torch.log2(prob))
    return ent


def grp_entropy_vec(x: torch.Tensor, group_size: int, num_bins: int = 256):
    import torch
    """
    x: [N] or [..., C]
    """
    x = x.reshape(-1, group_size)          # [G, group_size]
    G, K = x.shape

    # --- 1. 统一 bin 边界（非常重要） ---
    xmin = x.min()
    xmax = x.max()

    # [num_bins + 1]
    bin_edges = torch.linspace(
        xmin, xmax, num_bins + 1, device=x.device
    )

    # --- 2. bucketize：每个元素 -> bin index ---
    # [G, K], in [0, num_bins-1]
    bin_idx = torch.bucketize(x, bin_edges) - 1
    bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)

    # --- 3. 统计 group-wise histogram ---
    # 构造 group index
    group_idx = torch.arange(G, device=x.device).unsqueeze(1).expand(G, K)

    # 展平后用 bincount
    flat_idx = group_idx.reshape(-1) * num_bins + bin_idx.reshape(-1)

    hist = torch.bincount(
        flat_idx,
        minlength=G * num_bins
    ).reshape(G, num_bins).float()

    # --- 4. 概率 & entropy ---
    prob = hist / hist.sum(dim=1, keepdim=True)
    prob = prob.clamp(min=1e-12)  # 防 log(0)

    ent = -(prob * torch.log2(prob)).sum(dim=1)

    return ent.mean()


# # search for qsinr instead.
# @torch.no_grad()
# def get_quant_nvesm2(tensor_value: torch.Tensor, group_size: int):
#
#     sub_group_size = 8  # extra 2 bit for scale in subgroup
#     assert group_size % sub_group_size == 0
#
#     org_shape = tensor_value.shape
#     org_dtype = tensor_value.dtype
#
#     tensor_value = tensor_value.float()
#
#     if group_size > 0:
#         assert org_shape[-1] % group_size == 0
#         tensor_value = tensor_value.reshape(-1, group_size)
#
#     max_val = tensor_value.abs().amax(dim=1, keepdim=True)
#     # avoid divide a too small value
#     max_val = max_val.clamp(min=1e-8)
#
#     max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)
#
#     scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val
#
#     tensor_value = tensor_value.reshape(-1, sub_group_size)
#     # Compute the scaling factor
#     global_scale = scales.max() / FLOAT8_E4M3_MAX
#     scales = (
#         (scales / global_scale)
#         .clamp(min=FLOAT8_E4M3_EPS)
#         .to(torch.float8_e4m3fn)
#         .to(tensor_value.dtype)
#     ) * global_scale
#     exp = torch.floor(torch.log2(scales))
#     # exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
#     bias_mse = {}
#     bias_qsinr = {}
#     # range_ = range(-1, 2)
#     range_ = {0}
#     org_scales = scales
#     bias = 0
#
#     scales = org_scales * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
#     sub_groups_per_group = group_size // sub_group_size
#     scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
#     ratios = torch.tensor(
#         [1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device
#     )
#     # ratios = torch.tensor(
#     #     [1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device
#     # )
#     x_expanded = tensor_value.unsqueeze(2)
#     scales_expanded = scales.unsqueeze(2)
#
#     cand_scales = scales_expanded * ratios.view(1, 1, -1)
#     cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
#     qsinr_per_ratio = torch.log10(cand_qval.pow(2).mean(dim=1) / (cand_qval - x_expanded).pow(2).mean(dim=1) + 1e-12)
#     best_ratio_idx = qsinr_per_ratio.argmax(dim=1)
#     row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
#     best_dqval = cand_qval[row_idx, :, best_ratio_idx]
#     quant_qsinr_per_subgrp = qsinr_per_ratio[row_idx, best_ratio_idx]
#     tensor_deq = best_dqval.reshape(-1, group_size)
#     quant_qsinr_sum = quant_qsinr_per_subgrp.view(-1, sub_groups_per_group).mean(
#         dim=1, keepdim=True
#     )
#     bias_qsinr[bias] = (tensor_deq, quant_qsinr_sum)
#
#     all_qsinr = torch.cat([bias_qsinr[b][1] for b in range_], dim=1)
#     best_bias_idx = all_qsinr.argmax(dim=1)
#     all_deq = torch.stack([bias_qsinr[b][0] for b in range_], dim=0)
#     all_deq = all_deq.view(len(range_), -1, group_size)
#     idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
#     final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
#     tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
#
#     # archive
#     label = best_ratio_idx
#     tensor_value = tensor_value.reshape(-1, group_size) / org_scales
#     tensor_value = tensor_value.reshape(-1, sub_group_size)
#     for lab in range(4):
#         print(tensor_value[label == lab].shape)
#         container[lab].append(tensor_value[label == lab])
#
#     return tensor_deq

@torch.no_grad()
def get_quant_nvesm2(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # ratio = torch.tensor([1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device)
    ratio = torch.tensor([1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device)
    scales = scales.reshape(-1, 1).expand(-1, group_size // sub_group_size).reshape(-1, 1)
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)
    cand_scales = scales_expanded * ratio.view(1, 1, -1)
    cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
    # mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
    mse_per_ratio = (cand_qval - x_expanded).abs().mean(dim=1)
    best_ratio_idx = mse_per_ratio.argmin(dim=1)
    row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
    best_dqval = cand_qval[row_idx, :, best_ratio_idx]

    tensor_deq = best_dqval.reshape(org_shape).to(org_dtype)
    return tensor_deq


@torch.no_grad()
def get_quant_nvesm2_kur(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # ratio = torch.tensor([1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device)
    ratio = torch.tensor([1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device)
    scales = scales.reshape(-1, 1).expand(-1, group_size // sub_group_size).reshape(-1, 1)
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)
    cand_scales = scales_expanded * ratio.view(1, 1, -1)
    cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
    # mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
    # calculate kurtosis instead
    mean = cand_qval.mean(dim=1)
    centered = cand_qval - mean.unsqueeze(1)
    var = centered.pow(2).mean(dim=1).clamp(min=1e-12)
    fourth = centered.pow(4).mean(dim=1)
    kurt_per_ratio = fourth / (var ** 2)
    best_ratio_idx = kurt_per_ratio.argmin(dim=1)
    # mse_per_ratio = (cand_qval - x_expanded).abs().mean(dim=1)
    # best_ratio_idx = mse_per_ratio.argmin(dim=1)
    row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
    best_dqval = cand_qval[row_idx, :, best_ratio_idx]

    tensor_deq = best_dqval.reshape(org_shape).to(org_dtype)
    return tensor_deq

def draw_histogram(x: torch.Tensor, min_val, max_val, num_bins, method):
    import matplotlib.pyplot as plt
    import numpy as np
    total_hist = None
    if isinstance(x, torch.Tensor): x = x.cpu()
    if isinstance(min_val, torch.Tensor): min_val = min_val.cpu()
    if isinstance(max_val, torch.Tensor): max_val = max_val.cpu()
    print("Total Histogram: ")
    total_hist = torch.histc(x.float().abs(), bins=num_bins, min=min_val, max=max_val)
    print(total_hist)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, total_hist, color='royalblue', linewidth=2)
    plt.fill_between(bin_centers, total_hist, alpha=0.2, color='royalblue')
    plt.title("Activation Value Distribution " + method)
    plt.savefig("dump/" + "Histogram_" + method + ".png", dpi=150)

def draw_histogram_for_different_ratio():
    import os

    dump_dir = "dump"
    device = torch.device('cuda:0')

    files = [f for f in os.listdir(dump_dir) if f.endswith(".pt")]
    print(f"找到 {len(files)} 个文件，准备开始处理...")

    # ratio_labels = ["1", "1.25", "1.5", "1.75"]
    ratio_labels = ["1", "1.5"]

    # only process last 1 of them
    # files = files[-5:]
    files = files[:11]
    sub_group_size = 8
    for name in files:

        print(f"正在处理文件: {name}")
        file_path = os.path.join(dump_dir, name)
        x = torch.load(file_path)

        x = torch.load("dump/" + name)  # [N, T, Cin]
        x = x.to(device)
        x = x.reshape(-1, x.shape[-1])
        get_quant_nvesm2(x, group_size=16)

        # for lab in range(len(ratio_labels)):
        #     data_tensor = torch.cat(container[lab], dim=0).to(device).abs().reshape(-1, sub_group_size)
        #     num_gt_2 = (data_tensor > 2.0).sum(dim=1)
        #     percentage_gt_2 = (num_gt_2 > 3.0).float().mean().item() * 100
        #     print(f"Ratio: {ratio_labels[lab]}, Percentage of subgroup that has >=3 elem>2 : {percentage_gt_2:.2f}%")


    # draw 4 lines in one graph
    import matplotlib.pyplot as plt

    # 准备绘图窗口
    plt.figure(figsize=(10, 6))

    num_bins = 100

    for lab in range(len(ratio_labels)):
        # 1. 拼接并确保在正确的设备上
        # 假设 container[lab] 里的元素已经是 Tensor，如果不是，先转换
        data_tensor = torch.cat(container[lab], dim=0).to(device).abs()

        # --- 去掉归一化到 6.0 的步骤 ---
        # tensor_value = data_tensor / data_tensor.max() * 6.0  <-- 注释掉或删除
        tensor_value = data_tensor 

        # 2. 计算直方图数据 (在 GPU 上计算)
        # 统一设置一个合理的 min/max 范围，或者根据当前 tensor 动态获取
        v_min, v_max = tensor_value.min().item(), tensor_value.max().item()
        
        # torch.histc 返回的是频数
        hist = torch.histc(tensor_value, bins=num_bins, min=v_min, max=v_max)
        
        # 3. 准备横坐标 (Bins 边缘)
        x_bins = torch.linspace(v_min, v_max, num_bins)

        # 4. 绘制曲线
        # 必须搬回 CPU 才能画图
        plt.plot(x_bins.cpu().numpy(), hist.cpu().numpy(), label=f"Ratio: {ratio_labels[lab]}", alpha=0.8)

        tensor_value = tensor_value.reshape(-1, sub_group_size)
        num_gt_2 = (tensor_value > 3.5).sum(dim=1)
        count_num_gt_2 = []
        for i in range(sub_group_size+1):
            count_num_gt_2.append((num_gt_2 == i).float().mean().item() * 100)
            print(f"Ratio: {ratio_labels[lab]}, Percentage of subgroup that has {i} elem>3.5 : {count_num_gt_2[-1]:.2f}%")

    # 5. 修饰图表
    plt.title("Histogram Comparison of Different Ratios")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. 保存或显示
    plt.savefig('dump/combined_histogram_comp_for_different_ratio.png')
    plt.show()


def kurtosis_excess(x: torch.Tensor, eps: float = 1e-12):
    mean = x.mean(dim=1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=1, keepdim=True).clamp_min(eps)
    fourth = ((x - mean) ** 4).mean(dim=1, keepdim=True)
    return (fourth / (var ** 2)).squeeze(1) - 3.0


def draw_kurtosis_histogram_for_different_ratio():
    import os

    # dump_dir = "dump"
    device = torch.device('cuda:0')
    #
    # files = [f for f in os.listdir(dump_dir) if f.endswith(".pt")]
    # print(f"找到 {len(files)} 个文件，准备开始处理...")
    #
    # # only process last 1 of them
    # files = files[-1:]
    # for name in files:
    #
    #     print(f"正在处理文件: {name}")
    #     file_path = os.path.join(dump_dir, name)
    #     x = torch.load(file_path)
    #
    #     x = torch.load("dump/" + name)  # [N, T, Cin]
    #     x = x.to(device)
    #     x = x.reshape(-1, x.shape[-1])
    #     get_quant_nvesm2(x, group_size=16)

    # use gaussian data
    x_axis = []
    for j in range(1, 34):
        x_val = j / 2
        x_axis.append(x_val) # 修正1：填充横坐标
        sigma = 0.01 * 2 ** (j / 2)
        data = torch.randn(2 ** 18) * sigma
        data = data.to(device)  
        quant_data = get_quant_nvesm2(data, group_size=16)


    # draw 4 lines in one graph
    import matplotlib.pyplot as plt

    ratio_labels = ["1", "1.25", "1.5", "1.75"]
    kurtosis = [0] * 4
    num_bins = 100

    for lab in range(4):
        # 1. 拼接并确保在正确的设备上
        # 假设 container[lab] 里的元素已经是 Tensor，如果不是，先转换
        data_tensor = torch.cat(container[lab], dim=0).to(device).abs()
        data_tensor = data_tensor.reshape(-1, 8) # sub_group_size
        kurtosis[lab] = kurtosis_excess(data_tensor)
        
        # also print percentage of kurtosis > 0
        print(f"Ratio: {ratio_labels[lab]}, Kurtosis > 0 percentage: {(kurtosis[lab] > 0).float().mean().item() * 100:.2f}%")

    # draw kurtosis histogram of each ratio
    plt.figure(figsize=(10, 6))
    for lab in range(4):
        plt.hist(kurtosis[lab].cpu().numpy(), bins=num_bins, alpha=0.6, label=f"Ratio: {ratio_labels[lab]}") 
    plt.title("Kurtosis Comparison of Different Ratios")
    plt.xlabel("kurtosis")
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    plt.savefig('dump/kurtosis_comp_for_different_ratio.png')


def get_quant_new(tensor_value: torch.Tensor, group_size):
    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    subgroup_per_group = group_size // sub_group_size
    scales = scales.expand(tensor_value.shape[0], subgroup_per_group).reshape(-1).unsqueeze(1)
    tensor_value = tensor_value.reshape(-1, sub_group_size)

    num_gt_2 = (tensor_value.abs() / scales > 4).sum(dim=1, keepdim=True) / sub_group_size
    extra_scale = 1 + 0.5 * ( num_gt_2 >= 0.25 )
    # num_gt_2 = (tensor_value.abs() / scales > 2).sum(dim=1, keepdim=True)
    # extra_scale = 1 + 0.5 * (num_gt_2 >= 4).float()

    tensor_quant = cast_to_fp4(tensor_value / (scales * extra_scale)) * (scales * extra_scale)

    return tensor_quant.reshape(org_shape).to(org_dtype)


def compare_quant_err_with_different_sf():
    a = torch.tensor([1,2,3,4,5,6,7,8])
    b = 2 * a
    # generate random data
    a = torch.randn(10000) * 5
    b = a * 2
    scale_a = a.abs().amax() / FLOAT4_E2M1_MAX
    scale_b = b.abs().amax() / FLOAT4_E2M1_MAX
    quant_a = cast_to_fp4(a / scale_a) * scale_a
    quant_b = cast_to_fp4(b / scale_b) * scale_b
    mse_a = ((quant_a - torch.tensor(a)) ** 2).mean()
    mse_b = ((quant_b - torch.tensor(b)) ** 2).mean()
    qsinr_a = ( quant_a.pow(2).mean() / (quant_a - a).pow(2).mean() + 1e-12 ).log10()
    qsinr_b = ( quant_b.pow(2).mean() / (quant_b - b).pow(2).mean() + 1e-12 ).log10()
    print(f"mse_a: {mse_a.item()}, mse_b: {mse_b.item()}")
    print(f"qsinr_a: {qsinr_a.item()}, qsinr_b: {qsinr_b.item()}")


QUANT_METHOD_MAP = {
    "mxfp": get_quant_mxfp,
    "nvfp": get_quant_nvfp,
    "mxes": get_quant_mxes,
    "mxem": get_quant_mxem,
    "nves": get_quant_nves,
    "nvem": get_quant_nvem,
    "nvesm": get_quant_nvesm,
    "nvesm2": get_quant_nvesm2,
    "nvesm2_hw": get_quant_nvesm2_hw,
    "nvesem2": get_quant_nvesem2,
    "nvint4": get_quant_nvint4,
    "nvintesm2": get_quant_nvintesm2,
    # "hif4": get_quant_hif4,
    # "hifem": get_quant_hifem,
    # "hifes": get_quant_hifes,
    # "nvgt4": get_quant_gt4,
}


__name__ = "__main__"

container = {0: [], 1: [], 2: [], 3: []}
# draw_histogram_for_different_ratio()
# draw_kurtosis_histogram_for_different_ratio()
draw_mse_comp_with_gaussian()
# draw_mse_comp_with_real_weight()

# compare_quant_err_with_different_sf()
