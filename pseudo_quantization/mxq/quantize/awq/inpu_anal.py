import torch
import numpy as np
import matplotlib.pyplot as plt
name = "model_layers_0_mlp_down_proj.pt"
x = torch.load("awq_activation_dump/" + name)  # [N, T, Cin]
print(x.shape)  # [N, T, Cin]

x = x.reshape(-1, x.shape[-1])  # [N*T, Cin]

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

name = "model_layers_0_mlp_down_proj.pt"
x = torch.load("awq_activation_dump/" + name)  # [N, T, Cin]
x = x.reshape(-1, x.shape[-1])

# === 核心改动 ===
plot_data = x.abs().cpu().numpy()

def streaming_quantile(x, q=0.999, chunk_size=1_000_000):
    x = x.abs().flatten()
    qs = []
    for i in range(0, x.numel(), chunk_size):
        chunk = x[i : i + chunk_size]
        qs.append(torch.quantile(chunk.float(), q))
    return torch.quantile(torch.stack(qs), q)

p99 = streaming_quantile(x, 0.999)

# clip to expose bulk
# p99 = torch.quantile(x.abs().float(), 0.99).item()
plot_data = np.clip(plot_data, 1e-6, p99)

# structured subsample
plot_data = plot_data[::16, ::16]

rows, cols = plot_data.shape
N_grid, Cin_grid = np.meshgrid(np.arange(cols), np.arange(rows))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    Cin_grid,
    N_grid,
    plot_data,
    cmap="viridis",
    norm=LogNorm(vmin=1e-6, vmax=p99),
    linewidth=0,
    antialiased=False,
)

fig.colorbar(surf, ax=ax, shrink=0.5)
ax.set_xlabel("Channel")
ax.set_ylabel("Sample")
ax.set_zlabel("|Activation| (clipped, log)")
ax.set_title("3D Activation Surface (AWQ-aware)")

ax.view_init(elev=30, azim=45)
plt.show()
plt.savefig("awq_activation_dump/" + name.replace(".pt", "_fixed.png"))

# p99 = torch.quantile(x.abs().float(), 0.99, dim=0)
#
# plt.figure(figsize=(12,8))
# plt.plot(p99.cpu().numpy())
# # plt.yscale("log")
# plt.xlabel("Channel index")
# plt.ylabel("p99(|activation|)")
# plt.title("Channel-wise p99")
# plt.show()
# plt.savefig("awq_activation_dump/" + name.replace(".pt", "_p99.png"))

# plot_data = x.cpu().numpy()
#
# # 2. 创建网格坐标
# rows, cols = plot_data.shape
# n_range = np.arange(rows)
# cin_range = np.arange(cols)
# N_grid, Cin_grid = np.meshgrid(cin_range, n_range)
#
# # 3. 绘图
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # plot_surface 绘制曲面
# surf = ax.plot_surface(N_grid, Cin_grid, plot_data, cmap='viridis', edgecolor='none', alpha=0.8)
#
# # 添加颜色条
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#
# # 设置标签
# ax.set_xlabel('Input Channels (Cin)')
# ax.set_ylabel('Samples (N * T)')
# ax.set_zlabel('Activation Value')
# ax.set_title('3D Visualization of Layer Activations')
#
# # 调整视角
# ax.view_init(elev=30, azim=45)
#
# plt.show()
# plt.savefig("awq_activation_dump/" + name.replace(".pt", ".png"))
