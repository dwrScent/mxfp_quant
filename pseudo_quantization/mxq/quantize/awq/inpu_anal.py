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
    # import numpy as np
    # import torch
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import LogNorm
    #
    # name = "model_layers_8_mlp_up_proj.pt"
    # x = torch.load("dump/" + name)  # [N, T, Cin]
    # x = x.reshape(-1, x.shape[-1])
    #
    # # === 核心改动 ===
    # plot_data = x.abs().cpu().numpy()
    #
    # def streaming_quantile(x, q=0.999, chunk_size=1_000_000):
    #     x = x.abs().flatten()
    #     qs = []
    #     for i in range(0, x.numel(), chunk_size):
    #         chunk = x[i : i + chunk_size]
    #         qs.append(torch.quantile(chunk.float(), q))
    #     return torch.quantile(torch.stack(qs), q)
    #
    # p99 = streaming_quantile(x, 0.999)
    #
    # # clip to expose bulk
    # # p99 = torch.quantile(x.abs().float(), 0.99).item()
    # plot_data = np.clip(plot_data, 1e-6, p99)
    #
    # # structured subsample
    # # plot_data = plot_data[::16, ::16]
    #
    # group_size = 256
    # num_groups = plot_data.shape[1] // group_size
    #
    # cmaps = [
    #     "viridis", "plasma", "inferno", "magma",
    #     "cividis", "turbo"
    # ]
    #
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection="3d")
    # for g in range(num_groups):
    #     c_start = g * group_size
    #     c_end = min((g + 1) * group_size, plot_data.shape[1])
    #
    #     Z = plot_data[:, c_start:c_end]
    #     if Z.size == 0:
    #         continue
    #
    #     rows, cols = Z.shape
    #     Cin_sub, N_sub = np.meshgrid(
    #         np.arange(c_start, c_end),
    #         np.arange(rows)
    #     )
    #
    #     surf = ax.plot_surface(
    #         Cin_sub,
    #         N_sub,
    #         Z,
    #         cmap=cmaps[g % len(cmaps)],
    #         norm=LogNorm(vmin=1e-6, vmax=p99),
    #         linewidth=0,
    #         antialiased=False,
    #         alpha=0.85,
    #     )
    #
    # # rows, cols = plot_data.shape
    # # N_grid, Cin_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    # #
    # # fig = plt.figure(figsize=(12, 8))
    # # ax = fig.add_subplot(111, projection="3d")
    # #
    # # surf = ax.plot_surface(
    # #     Cin_grid,
    # #     N_grid,
    # #     plot_data,
    # #     cmap="viridis",
    # #     norm=LogNorm(vmin=1e-6, vmax=p99),
    # #     linewidth=0,
    # #     antialiased=False,
    # # )
    #
    # fig.colorbar(surf, ax=ax, shrink=0.5)
    # ax.set_xlabel("Channel")
    # ax.set_ylabel("Sample")
    # ax.set_zlabel("|Activation| (clipped, log)")
    # ax.set_title("3D Activation Surface for " + name)
    #
    # ax.view_init(elev=30, azim=45)
    # plt.show()
    # plt.savefig("dump/" + name.replace(".pt", "_fixed.png"))


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
    levels = levels.to(x.device)
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    odd_boundaries = boundaries[1::2]
    mask = torch.isin(x, odd_boundaries)
    x = x + 0.0000005 * mask  # round to even
    indices = torch.bucketize(x, boundaries)
    indices.clamp_(0, len(levels) - 1)

    quantized = levels[indices]
    val_elements, counts = torch.unique(quantized, sorted=True, return_counts=True)
    grid_cnt_local = dict(zip(val_elements.cpu().numpy(), counts.cpu().numpy()))
    for level, count in grid_cnt_local.items():
        grid_cnt[level] += count
    return quantized, indices


def cast_to_fp4(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    x_quant, _ = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    return x_quant * sign


@torch.no_grad()
def get_quant_nvess(tensor_value: torch.Tensor, group_size: int):

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


if __name__ == "__main__":

    # x = torch.load("dump/model_layers_8_mlp_up_proj.pt")  # [N, T, Cin]
    x = torch.load("dump/model_layers_0_self_attn_q_proj.pt")  # [N, T, Cin]
    x = x.reshape(-1, x.shape[-1])

    # from collections import defaultdict
    # grid_cnt = defaultdict(int)
    # x_quant = get_quant_nvfp(x, group_size=16)
    # # x_quant = get_quant_nvess(x, group_size=16)
    # print("Quantization Level Counts:")
    # for level in sorted(grid_cnt.keys()):
    #     print(f"Value: {level:.2f}, Count: {grid_cnt[level]}")

    import numpy as np
    group_size = 16
    x = x.reshape(-1, group_size)
    bins = 100
    max_ = 6.0
    bin_edges = np.linspace(0, max_, bins + 1)
    total_hist = None
    for i in range(x.shape[0]):
        row = x[i, :].abs().cpu().numpy()
        hist, _ = np.histogram(row, bins=bin_edges)
        if i == 0:
            total_hist = hist
        else:
            total_hist += hist

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, total_hist, color='royalblue', linewidth=2)
    plt.fill_between(bin_centers, total_hist, alpha=0.2, color='royalblue')
    plt.savefig("dump/activation_histogram.png", dpi=150)
