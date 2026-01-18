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

if __name__ == "__main__":

    x = torch.load("dump/model_layers_8_mlp_up_proj.pt")  # [N, T, Cin]
    x = x.reshape(-1, x.shape[-1])


