import torch
from KLdiv import manual_kl_divergence

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_EPS = 2 ** (-9)
FLOAT8_E4M4_EPS = 2 ** (-10)
FLOAT8_E4M3_MAX = 448.0
LEVEL_2_MAX = 7.05

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


__name__ = "__main__"

import torch
import numpy as np
# name = "model_layers_0_self_attn_o_proj.pt"
# # name = "model_layers_8_mlp_down_proj.pt"
# tensor_value = torch.load('dump/' + name)
#
# device = torch.device('cuda:0')
# tensor_value = tensor_value.to(device).float()

# device is gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# generate gaussian data
sigma = 0.01 * 2 ** 8
tensor_value = torch.randn(8192 * 2, device=device) * sigma

x_axis = []
kldiv_good_acc = []
kldiv_bad_acc = []
good_groups_acc = []
bad_groups_acc = []
kurt_good_acc = []
kurt_bad_acc = []
from scipy.stats import kurtosis
for j in range(1, 34):
    x_val = j / 2
    x_axis.append(x_val) # 修正1：填充横坐标
    sigma = 0.01 * 2 ** (j / 2)
    signal_variance = sigma ** 2

    tensor_value = torch.randn(8192 * 1024, device=device) * sigma

    group_size = 8
    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    l1_group_size = group_size * 2
    tensor_value = tensor_value.reshape(-1, l1_group_size)
    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)
    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)
    scales = max_val / max_quant_val

    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    # quantization
    tensor_quant = cast_to_fp4(tensor_value / scales) * scales

    #compute mse for each l1_group
    mse_per_l1_group = (tensor_quant - tensor_value).pow(2).mean(dim=1)
    # get 10% largest mse groups and 10% smallest mse groups
    good_threshold = torch.quantile(mse_per_l1_group, 0.1)
    bad_threshold = torch.quantile(mse_per_l1_group, 0.9)
    tensor_value = tensor_value / scales
    good_groups = tensor_value[mse_per_l1_group < good_threshold]
    bad_groups = tensor_value[mse_per_l1_group > bad_threshold]

    good_groups_acc.append(good_groups.abs().cpu())
    bad_groups_acc.append(bad_groups.abs().cpu())
    
    # compute kl divergence for groups in good_groups and bad_groups
    good_groups = good_groups.reshape(-1, group_size)
    bad_groups = bad_groups.reshape(-1, group_size)
    kldiv_good = []
    kldiv_bad = []
    for i in range(0, good_groups.shape[0], 2):
        p = good_groups[i].cpu()
        q = good_groups[i + 1].cpu()
        kldiv_good.append(manual_kl_divergence(p, q))
    for i in range(0, bad_groups.shape[0], 2):
        p = bad_groups[i].cpu()
        q = bad_groups[i + 1].cpu()
        kldiv_bad.append(manual_kl_divergence(p, q))

    # append kldiv mean value of good groups and bad groups
    kldiv_good_acc.append(torch.tensor(kldiv_good).mean().item())
    kldiv_bad_acc.append(torch.tensor(kldiv_bad).mean().item())

    # compute kurtosis distribution for good groups and bad groups
    for i in range(good_groups.shape[0]):
        p = good_groups[i].cpu()
        kurt_good_acc.append(kurtosis(p))
    for i in range(bad_groups.shape[0]):
        p = bad_groups[i].cpu()
        kurt_bad_acc.append(kurtosis(p))


import matplotlib.pyplot as plt

# draw histogram of kurtosis distribution for good groups and bad groups
plt.figure(figsize=(12, 8))
min_good = torch.tensor(kurt_good_acc).min().item()
max_good = torch.tensor(kurt_good_acc).max().item()
min_bad = torch.tensor(kurt_bad_acc).min().item()
max_bad = torch.tensor(kurt_bad_acc).max().item()
hist_bins = 100
hist_good = torch.histc(torch.tensor(kurt_good_acc), bins=hist_bins, min=min_good, max=max_good)
hist_bad = torch.histc(torch.tensor(kurt_bad_acc), bins=hist_bins, min=min_bad, max=max_bad)
x_bins_good = torch.linspace(min_good, max_good, hist_bins)
x_bins_bad = torch.linspace(min_bad, max_bad, hist_bins)
plt.plot(x_bins_good.cpu().numpy(), hist_good.cpu().numpy(), label="Good Groups", alpha=0.8)
plt.plot(x_bins_bad.cpu().numpy(), hist_bad.cpu().numpy(), label="Bad Groups", alpha=0.8)
plt.title("Kurtosis Distribution of Good Groups and Bad Groups")
plt.xlabel("Kurtosis")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dump/kurtosis_distribution_comparison.png')
plt.show()

#draw line graph for kl divergence accumulation of good groups and bad groups
plt.figure(figsize=(14, 9))
plt.plot(x_axis, kldiv_good_acc, label="Good Groups KL Divergence", marker='o')
plt.plot(x_axis, kldiv_bad_acc, label="Bad Groups KL Divergence", marker='o')
plt.title("KL Divergence Comparison of Good Groups and Bad Groups")
plt.xlabel("Group Index")
plt.ylabel("KL Divergence Acc")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dump/kl_divergence_comparison.png')
plt.show()

# draw histogram of good groups and bad groups
plt.figure(figsize=(12, 8))
min_good = torch.cat(good_groups_acc).min().item()
max_good = torch.cat(good_groups_acc).max().item()
min_bad = torch.cat(bad_groups_acc).min().item()
max_bad = torch.cat(bad_groups_acc).max().item()
hist_bins = 100
hist_good = torch.histc(torch.cat(good_groups_acc), bins=hist_bins, min=min_good, max=max_good)
hist_bad = torch.histc(torch.cat(bad_groups_acc), bins=hist_bins, min=min_bad, max=max_bad)
x_bins_good = torch.linspace(min_good, max_good, hist_bins)
x_bins_bad = torch.linspace(min_bad, max_bad, hist_bins)
plt.plot(x_bins_good.cpu().numpy(), hist_good.cpu().numpy(), label="Good Groups", alpha=0.8)
plt.plot(x_bins_bad.cpu().numpy(), hist_bad.cpu().numpy(), label="Bad Groups", alpha=0.8)
plt.title("Histogram of Good Groups and Bad Groups")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dump/good_bad_groups_data_histogram_comparison.png')
plt.show()
