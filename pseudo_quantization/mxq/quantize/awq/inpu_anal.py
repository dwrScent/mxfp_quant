import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

def draw_3d_activation_surface():
    dump_dir = "dump"

    files = [f for f in os.listdir(dump_dir) if f.endswith(".pt")]
    print(f"找到 {len(files)} 个文件，准备开始处理...")

    for name in files:
        print(f"正在处理文件: {name}")
        file_path = os.path.join(dump_dir, name)
        x = torch.load(file_path)

        x = torch.load("dump/" + name)  # [N, T, Cin]
        x = x.reshape(-1, x.shape[-1])

        # === 1. 采样与数据处理 ===
        # 3D 绘图点数建议控制在 10000 个以内，否则渲染极慢
        stride_n = max(1, x.shape[0] // 64)  # 行采样
        stride_c = 16  # 列采样（组内采样）
        group_size = 256  # 多少个 Channel 为一组

        def streaming_quantile(x, q=0.999, chunk_size=1_000_000):
            x = x.flatten()
            qs = []
            for i in range(0, x.numel(), chunk_size):
                chunk = x[i : i + chunk_size]
                qs.append(torch.quantile(chunk.float(), q))
            return torch.quantile(torch.stack(qs), q)

        plot_data = x[::stride_n, :].abs().cpu().numpy()
        p99 = streaming_quantile(x.abs().float(), 0.999).item()
        plot_data = np.clip(plot_data, 1e-6, p99)

        # === 2. 设置颜色库 ===
        # 使用几种对比明显的颜色图循环切换
        cmap_names = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        num_groups = (plot_data.shape[1] + group_size - 1) // group_size

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        # === 3. 分组循环绘制 ===
        for g in range(num_groups):
            c_start = g * group_size
            c_end = (g + 1) * group_size
            
            # 组内进一步采样以保证性能
            Z = plot_data[:, c_start:c_end:stride_c]
            
            rows, cols = Z.shape
            # 生成对应的坐标网格
            c_idx = np.arange(c_start, c_end, stride_c)
            n_idx = np.arange(0, plot_data.shape[0] * stride_n, stride_n)
            Cin_grid, N_grid = np.meshgrid(c_idx, n_idx)

            # 为当前组选择颜色
            current_cmap = cmap_names[g % len(cmap_names)]
            
            surf = ax.plot_surface(
                Cin_grid,
                N_grid,
                Z,
                cmap=current_cmap,
                # 去掉 LogNorm，改用线性缩放
                vmin=0,           # 绝对值最小通常为 0
                vmax=p99,         # 颜色映射的最大值设定为 p99
                linewidth=0,
                antialiased=True,
                alpha=1.0
            )
            # surf = ax.plot_surface(
            #     Cin_grid,
            #     N_grid,
            #     Z,
            #     cmap=current_cmap,
            #     norm=LogNorm(vmin=1e-6, vmax=p99),
            #     linewidth=0,
            #     antialiased=True,
            #     alpha=0.6  # 略带透明度，防止遮挡
            # )

        # 装饰
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Sample Index")
        ax.set_zlabel("Magnitude ")
        ax.set_title(f"Grouped 3D Activations: {name}\n(Group Size={group_size})")

        # 视角优化
        ax.view_init(elev=30, azim=-60)

        # 注意：先保存再 show
        plt.tight_layout()
        plt.savefig(f"dump/{name.replace('.pt', '.png')}", dpi=150)
        plt.show()
        plt.close()


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
# FLOAT8_E4M3_EPS = 2 ** (-9)
FLOAT8_E4M4_EPS = 2 ** (-10)
FLOAT8_E4M3_MAX = 448.0
LEVEL_2_MAX = 7

#
# @torch.no_grad()
# def fp16(tensor_value: torch.Tensor, group_size: int):
#     return tensor_value


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


FP4_E2M1_GRID = torch.tensor(float_value(2, 1), device="cuda")
FP6_E2M3_GRID = torch.tensor(float_value(2, 3), device="cuda")
FP8_E5M3_GRID = torch.tensor(float_value(5, 3), device="cuda")
FP8_E4M4_GRID = torch.tensor(float_value(4, 4), device="cuda")


def quantize_to_grid(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    global grid_cnt
    global org_grid_cnt
    levels = levels.to(x.device)
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    odd_boundaries = boundaries[1::2]
    mask = torch.isin(x, odd_boundaries)
    x = x + 0.0000005 * mask  # round to even
    indices = torch.bucketize(x, boundaries)
    indices.clamp_(0, len(levels) - 1)
    quantized = levels[indices]

    # # record quantized value counts
    # val_elements, counts = torch.unique(quantized, sorted=True, return_counts=True)
    # grid_cnt_local = dict(zip(val_elements.cpu().numpy(), counts.cpu().numpy()))
    # for level, count in grid_cnt_local.items():
    #     grid_cnt[level] += count

    return quantized, indices


def cast_to_fp4(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    x_quant, _ = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    return x_quant * sign


@torch.no_grad()
def get_quant_nvfp(tensor_value: torch.Tensor, group_size: int):
    global grid_cnt

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

    tensor_quant = torch.clamp(torch.round(tensor_value / scales), min=-7.0, max=7.0)
    global grid_cnt
    val_elements, counts = torch.unique(tensor_quant, sorted=True, return_counts=True)
    grid_cnt_local = dict(zip(val_elements.cpu().numpy(), counts.cpu().numpy()))
    for level, count in grid_cnt_local.items():
        grid_cnt[level] += count

    tensor_quant = tensor_quant * scales
    return tensor_quant.reshape(org_shape).to(org_dtype)


def entropy(x: torch.Tensor, num_bins: int = 256):
    hist = torch.histc(x, bins=num_bins, min=x.min().item(), max=x.max().item())
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    ent = -torch.sum(prob * torch.log2(prob))
    return ent


def grp_entropy(x: torch.Tensor, group_size, num_bins=256):
    x = x.reshape(-1, group_size)
    ent_list = []
    for i in range(x.shape[0]):
        ent = entropy(x[i], num_bins=num_bins)
        ent_list.append(ent.item())
    return np.mean(ent_list)


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



def draw_histogram(x: torch.Tensor, min_val, max_val, num_bins, method):

    total_hist = None
    print("Total Histogram: ")
    total_hist = torch.histc(x.float().abs(), bins=num_bins, min=min_val, max=max_val)
    print(total_hist)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, total_hist, color='royalblue', linewidth=2)
    plt.fill_between(bin_centers, total_hist, alpha=0.2, color='royalblue')
    plt.title("Activation Value Distribution " + method + " for " + name)
    plt.savefig("dump/" + "Histogram_" + method + "_for_" + name.replace(".pt", ".png"), dpi=150)


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


    # ent_grp = grp_entropy(x, group_size, num_bins=256)
    ent_grp = grp_entropy_vec(x.abs(), group_size, num_bins=256)
    print(f"Entropy before cast to fp4: {ent_grp:.4f}")
    # --- 2. 向量化归一化与量化 ---
    # 整个矩阵并行量化
    x_quant = cast_to_fp4(x.abs())
    # ent_grp = grp_entropy(x, group_size, num_bins=256)
    ent_grp = grp_entropy_vec(x_quant, group_size, num_bins=256)
    print(f"Entropy after cast to fp4: {ent_grp:.4f}")


def cast_to_E6M2(x: torch.Tensor):
    x = x.clamp(min=2 ** (-48) * 1.0, max=2 ** 15 * 1.5)
    E = torch.floor(torch.log2(x))
    return torch.round(x * 2 ** (-E + 2)) * 2 ** (E - 2)


def hif4_anal(tensor_value: torch.Tensor, group_size=64):
    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    assert group_size == 64
    tensor_value = tensor_value.reshape(-1, group_size)

    sign = torch.sign(tensor_value)

    v_max16 = torch.zeros((tensor_value.shape[0], 16), device=tensor_value.device)
    v_max8 = torch.zeros((tensor_value.shape[0], 8), device=tensor_value.device)
    v_max16 = tensor_value.abs().reshape(tensor_value.shape[0], -1, 4).amax(dim=2)
    v_max8 = v_max16.reshape(tensor_value.shape[0], -1, 2).amax(dim=2)
    v_max = v_max8.amax(dim=1, keepdim=True)
    SF = cast_to_E6M2(v_max / LEVEL_2_MAX)
    E1_8 = (v_max8 / SF) >= 4
    E1_8 = E1_8.to(v_max8.dtype)
    E1_8x2 = E1_8.repeat_interleave(2, dim=1)
    E1_16 = (v_max16 / SF * 2.0 ** (-E1_8x2)) >= 2
    E1_16 = E1_16.to(v_max16.dtype)
    DE16 = E1_16 + E1_8x2
    DE64 = DE16.repeat_interleave(4, dim=1)

    data = tensor_value.abs() / (SF * 2.0 ** DE64)
    min_val, max_val = 0.0, 3.0
    num_bins = 100
    draw_histogram(data, min_val, max_val, num_bins, "hif4")

    ent_grp = grp_entropy_vec(data, group_size, num_bins=256)
    print(f"Entropy before quantization: {ent_grp:.4f}")

    # multiply 2^2, plus 0.5, floor, then multiply 2^-2, 
    # it's round to nearest with 2 bit mantissa
    in_grp = torch.floor(tensor_value.abs() / (SF * 2.0 ** (DE64 - 2)) + 0.5) * 2.0 ** (-2)
    in_grp[in_grp >= 2.0] = 1.75

    ent_grp = grp_entropy_vec(in_grp, group_size, num_bins=256)
    print(f"Entropy after quantization: {ent_grp:.4f}")


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
    ratio_div = torch.tensor([1.0, 0.78125, 0.65625, 0.5713125], dtype=tensor_value.dtype, device=tensor_value.device)
    scales = scales.reshape(-1, 1).expand(-1, group_size // sub_group_size).reshape(-1, 1)
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)
    scales_expanded_div = 1 / scales_expanded
    cand_scales_div = scales_expanded_div * ratio_div.view(1, 1, -1)
    cand_scales = scales_expanded * ratio.view(1, 1, -1)
    cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
    norm_val = (x_expanded / cand_scales)
    pre_scale_val = (tensor_value / scales).unsqueeze(2) * torch.ones(1, 1, len(ratio), device=tensor_value.device)
    norm_val = x_expanded * cand_scales_div
    # err_per_ratio = (norm_val - cast_to_fp4(norm_val)).abs() * ratio.view(1, 1, -1)
    err_per_ratio = (pre_scale_val - cast_to_fp4(norm_val) * ratio.view(1, 1, -1)).abs()
    err_count.append(err_per_ratio)
    err_per_ratio = (err_per_ratio - torch.round(err_per_ratio)).abs()
    # err_count.append((norm_val - cast_to_fp4(norm_val)).abs() * ratio.view(1, 1, -1))
    # mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
    mse_per_ratio = (cand_qval - x_expanded).abs().mean(dim=1)
    best_ratio_idx = mse_per_ratio.argmin(dim=1)
    row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
    best_dqval = cand_qval[row_idx, :, best_ratio_idx]

    tensor_deq = best_dqval.reshape(org_shape).to(org_dtype)
    return tensor_deq


def nvesm2_anal(x: torch.Tensor, group_size=16):
    org_shape = x.shape
    org_dtype = x.dtype

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    x = x.reshape(-1, group_size).float()

    import torch
    import numpy as np

    max_ = 6.0
    max_val = x.abs().amax(dim=1, keepdim=True)
    scales = max_val / max_
    x = x.reshape(-1, sub_group_size)
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    global_scale = global_scale.clamp(min=1e-8)
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(x.dtype)
    )

    bias_mse = {}
    range_ = {0}
    org_scales = scales
    sub_groups_per_group = group_size // sub_group_size
    tensor_value = x

    bias = 0
    scales = org_scales * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
    scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
    ratios = torch.tensor(
        [1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device
    )
    # ratios = torch.tensor(
    #     [1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device
    # )
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)

    cand_scales = scales_expanded * ratios.view(1, 1, -1)
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

    # all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    # best_bias_idx = all_mse.argmin(dim=1)
    # all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    # all_deq = all_deq.view(len(range_), -1, group_size)
    # idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    # final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    # tensor_deq = final_deq.reshape(org_shape).to(org_dtype)


    # bin_edges 假设是等间距的，例如从 0 到 6 分 100 份
    num_bins = 100
    min_val, max_val_bin = 0.0, 7.0 # 根据你的需求设置边界
    cand_data = x_expanded / cand_scales / global_scale
    x = cand_data[row_idx, :, best_ratio_idx]

    draw_histogram(x, min_val, max_val_bin, num_bins, "nvesm2")

    ent_grp = grp_entropy_vec(x, group_size, num_bins=256)
    print(f"Entropy before quantization: {ent_grp:.4f}")
    x_quant = cast_to_fp4(x.abs())
    # ent_grp = grp_entropy(x, group_size, num_bins=256)
    ent_grp = grp_entropy_vec(x_quant, group_size, num_bins=256)
    print(f"Entropy after cast to fp4: {ent_grp:.4f}")


if __name__ == "__main__":

    dump_dir = "dump"
    device = torch.device('cuda:0')

    files = [f for f in os.listdir(dump_dir) if f.endswith(".pt")]
    print(f"找到 {len(files)} 个文件，准备开始处理...")

    files = files[32:37]
    err_count = []
    for name in files:
        print(f"正在处理文件: {name}")
        x = torch.load("dump/" + name)  # [N, T, Cin]
        x = x.reshape(-1, x.shape[-1])

        # nvfp_anal(x, group_size=16)
        # hif4_anal(x, group_size=64)
        # nvesm2_anal(x, group_size=16)
        get_quant_nvesm2(x, group_size=16)

        # draw histogram of err_count
        err_count_tensor = torch.cat(err_count, dim=0).cpu()
        num_bins = 100
        min_val = err_count_tensor.min().item()
        max_val = err_count_tensor.max().item()
        draw_histogram(err_count_tensor, min_val, max_val, num_bins, "nvesm2_error")
