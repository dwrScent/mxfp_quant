import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

import os, sys
import math

def outlier_ratio_stat(w_data, q_config, outlier_ratio, i=None, n=None):            
    ratio_stats = {
        '1-1.2': torch.tensor(0.),
        '1.2-1.5': torch.tensor(0.),
        '1.5-2': torch.tensor(0.),
        '>2': torch.tensor(0.)
    }
    w = w_data.clone()
    ic, oc = w.shape[1], w.shape[0]


    outlier_num = math.ceil(ic * outlier_ratio)
    outlier_masks_mw = torch.zeros_like(w, dtype=torch.int8).to(w.device)
    _, outlier_index = torch.topk(w.abs(), outlier_num)
    outlier_masks_mw.scatter_(1, outlier_index, 1)

    non_outlier_masks_mw = (outlier_masks_mw == 0).to(dtype=torch.int8)
    w = non_outlier_masks_mw * w_data
    outlier_value =  outlier_masks_mw * w_data

    if q_config['q_group_size'] > 0:
        w = w.reshape(-1, q_config['q_group_size'])
        outlier_value = outlier_value.reshape(-1, q_config['q_group_size'])
    outlier_value = outlier_value.abs()

    outlier_max = outlier_value.abs().amax(dim=1, keepdim=True).clone()

    # 0 值先换成 1000，为了计算 amin
    outlier_value = torch.where(outlier_value.abs() > torch.tensor(0.), outlier_value, torch.tensor(1e3) )
    outlier_min = outlier_value.abs().amin(dim=1, keepdim=True).clone()
    normal_max = w.abs().amax(dim=1, keepdim=True)

    zero_mask = torch.zeros_like(outlier_min, dtype=torch.int8).to(outlier_min.device)
    outlier_min = torch.where(outlier_min > torch.tensor(999.), zero_mask, outlier_min)

    ratio = outlier_min / normal_max
    ratio_max = outlier_max / normal_max

    # print(f"layer{i}, {n}, max ratio o_min / n_max: {torch.max(ratio)} shape: {w.shape}")
    # print(f"layer{i}, {n}, max ratio o_max / n_max: {torch.max(ratio_max)} shape: {w.shape}")
    w = w.reshape(w_data.shape[0], w_data.shape[1])

    zero_mask = torch.zeros_like(ratio, dtype=torch.int8).to(ratio.device)
    one_mask = torch.ones_like(ratio, dtype=torch.int8).to(ratio.device)

    for key in ratio_stats:
        ratio_stats[key] = ratio_stats[key].to(ratio_max.device)
    ratio_stats['1-1.2'] += torch.sum(torch.where((ratio_max >= 1) & (ratio_max < 1.2), one_mask, zero_mask))
    ratio_stats['1.2-1.5'] += torch.sum(torch.where((ratio_max >= 1.2) & (ratio_max < 1.5), one_mask, zero_mask))
    ratio_stats['1.5-2'] += torch.sum(torch.where((ratio_max >= 1.5) & (ratio_max < 2), one_mask, zero_mask))
    ratio_stats['>2'] += torch.sum(torch.where( ratio_max > 2, one_mask, zero_mask))
    assert w.shape == w_data.shape

    # make_heat_map(outlier_masks_mw, i, n, 10000)
    # make_distribution_channel(w_data, w, i, n, 100, "outlier") 
    
    # ratio_threshold = 4
    # if torch.max(ratio) > ratio_threshold:
    #     group_dist_outlier(w_data, w, q_config['q_group_size'],i, n, 10000, ratio=ratio, ratio_threshold=ratio_threshold) 
    
    stat_sum = ratio_stats['1-1.2'] + ratio_stats['1.2-1.5'] + ratio_stats['1.5-2'] + ratio_stats['>2']
    print(f"stats: ratio 1-1.2: {ratio_stats['1-1.2']}, ratio 1.2-1.5: {ratio_stats['1.2-1.5']}, ratio 1.5-2: {ratio_stats['1.5-2']}, ratio >2: {ratio_stats['>2']}")
    print(f"stats ratio: ratio 1-1.2: {ratio_stats['1-1.2']/stat_sum * 100:.3f}%, ratio 1.2-1.5: {ratio_stats['1.2-1.5']/stat_sum * 100:.3f}%, ratio 1.5-2: {ratio_stats['1.5-2']/stat_sum * 100:.3f}%, ratio >2: {ratio_stats['>2']/stat_sum * 100:.3f}%")

    return ratio_max
    
def outlier_count(w_data, q_config, outlier_ratio, i, n):
    ratio_stats = {
        '0': torch.tensor(0.),
        '1': torch.tensor(0.),
        '1-5': torch.tensor(0.),
        '>5': torch.tensor(0.)
    }

    w = w_data.clone()
    ic, oc = w.shape[1], w.shape[0]


    outlier_num = math.ceil(ic * outlier_ratio)
    outlier_masks_mw = torch.zeros_like(w, dtype=torch.int8).to(w.device)
    
    _, outlier_index = torch.topk(w.abs(), outlier_num)
    outlier_masks_mw.scatter_(1, outlier_index, 1)

    non_outlier_masks_mw = (outlier_masks_mw == 0).to(dtype=torch.int8)
    w = non_outlier_masks_mw * w_data
    w = w.reshape(-1, q_config['q_group_size'])

    outlier_masks_mw = outlier_masks_mw.reshape(-1, q_config['q_group_size'])

    outlier_group_count = torch.sum(outlier_masks_mw, dim=1, keepdim=True)

    # ratio_threshold = 5
    # group_dist_outlier(w_data, w, q_config['q_group_size'],i, n, 10000, ratio=outlier_group_count, ratio_threshold=ratio_threshold) 

    zero_mask = torch.zeros_like(outlier_group_count, dtype=torch.int8).to(outlier_group_count.device)
    one_mask = torch.ones_like(outlier_group_count, dtype=torch.int8).to(outlier_group_count.device)

    for key in ratio_stats:
        ratio_stats[key] = ratio_stats[key].to(outlier_group_count.device)

    ratio_stats['0'] += torch.sum(torch.where((outlier_group_count == 0), one_mask, zero_mask))
    ratio_stats['1'] += torch.sum(torch.where((outlier_group_count == 1), one_mask, zero_mask))
    ratio_stats['1-5'] += torch.sum(torch.where((outlier_group_count > 1) & (outlier_group_count <= 5), one_mask, zero_mask))
    ratio_stats['>5'] += torch.sum(torch.where( outlier_group_count > 5, one_mask, zero_mask))

    stat_sum = ratio_stats['0'] + ratio_stats['1'] + ratio_stats['1-5'] + ratio_stats['>5']
    print(f"stats: count 0: {ratio_stats['0']}, count 1: {ratio_stats['1']}, count 1-5: {ratio_stats['1-5']}, count > 6: {ratio_stats['>5']}")
    print(f"stats percent: count 0: {ratio_stats['0']/stat_sum * 100:.3f}%, count 1: {ratio_stats['1']/stat_sum * 100:.3f}%, count 1-5: {ratio_stats['1-5']/stat_sum * 100:.3f}%, count > 5: {ratio_stats['>5']/stat_sum * 100:.3f}%")
    # exit(0)

def make_heat_map(outlier_mask, layer_idx=0, layer_name="", max_fig=1000, desc=""):
    file_path = os.getcwd()
    save_path = f'{file_path}/distri_img'
    os.system(f'mkdir -p {save_path}')

    outlier_mask = outlier_mask.cpu()

    data_list = outlier_mask.numpy()
    print(data_list.max(), data_list.min(), data_list.sum())

    plt.imshow(data_list, cmap='cool', interpolation='none')
    # ax = sns.heatmap(data_list, linewidth=0.3)
    plt.show()
    plt.savefig(f'{save_path}/layer{layer_idx}_{layer_name}_{desc}.png')
    plt.clf()
    exit(0)

def group_dist_outlier(w_data, w_data_clip_outlier, group_size=-1, layer_idx=0, layer_name="", max_fig=1000, ratio=None, ratio_threshold=10.0, desc=""):

    if group_size > 0 and w_data.shape[-1] % group_size != 0:
        print(f"Input channel: {w_data.shape[-1]} is not divisible by group_size: {group_size}")
        return
    
    # Prepare data based on group_size
    w_data_group = w_data.reshape(-1, group_size) if group_size > 0 else w_data
    w_data_clip_outlier_g = w_data_clip_outlier.reshape(-1, group_size) if group_size > 0 else w_data_clip_outlier

    plt_title = "weight group distribution" if group_size > 0 else "weight in-channel distribution"
    plt.title(plt_title)
    plt.ylabel('number')
    plt.xlabel('value')

    save_path = os.path.join(os.getcwd(), 'distri_img')
    os.makedirs(save_path, exist_ok=True) 

    for idx, group in enumerate(w_data_group):
        if ratio[idx] > ratio_threshold:
            if idx > max_fig:
                print(f"up to the max number of figures: {max_fig}")
                return
            data_list = group.view(-1).tolist()
            data_list_clip_outlier = w_data_clip_outlier_g[idx].view(-1).tolist()

            data_list = group.view(-1).tolist()
            interval = (max(data_list) - min(data_list)) / 16  # Adjust the number of bins.
            bins = np.arange(min(data_list) - interval * 3, max(data_list) + interval * 3, interval)
            bins = np.sort(np.insert(bins, 0, 0))

            plt.title(f"outlier count = {ratio[idx].cpu().numpy()}")

            plt.hist( data_list , bins=bins, color='red',label="weight outlier")
            plt.hist( data_list_clip_outlier , bins=bins, color='blue',label="weight")
            plt.legend()
            plt.savefig(f'{save_path}/layer{layer_idx}_{layer_name}_group_{idx}_{desc}.png')
            plt.clf()

def group_dist(w_data, group_size=-1, layer_idx=0, layer_name="", max_fig=1000, desc=""):
    if group_size > 0 and w_data.shape[-1] % group_size != 0:
        print(f"Input channel: {w_data.shape[-1]} is not divisible by group_size: {group_size}")
        return
    
    # Prepare data based on group_size
    w_data_group = w_data.reshape(-1, group_size) if group_size > 0 else w_data

    plt_title = "weight group distribution" if group_size > 0 else "weight in-channel distribution"
    plt.title(plt_title)
    plt.ylabel('number')
    plt.xlabel('value')

    max_val = torch.max(torch.abs(w_data_group), dim=1, keepdim=True).values
    var_group = torch.var(w_data_group / max_val, dim=1, keepdim=True)

    save_path = os.path.join(os.getcwd(), 'distri_img')
    os.makedirs(save_path, exist_ok=True) 

    for idx, group in enumerate(w_data_group):
        if idx > max_fig:
            print(f"up to the max number of figures: {max_fig}")
            break
        
        data_list = group.view(-1).tolist()
        interval = (max(data_list) - min(data_list)) / 100  # Adjust the number of bins.
        bins = np.arange(min(data_list) - interval * 3, max(data_list) + interval * 3, interval)
        bins = np.sort(np.insert(bins, 0, 0))

        print((f'layer{layer_idx}_{layer_name}_group_{idx}_{desc} {max(data_list)} {min(data_list)}'))
        plt.hist( data_list , bins=bins, color='blue',label=f"weight {var_group[idx]}")
        plt.legend()
        plt.savefig(f'{save_path}/layer{layer_idx}_{layer_name}_group_{idx}_{desc}.png')
        plt.clf()

def distri_3d(w_data, group_size=-1, layer_idx=0, layer_name="", max_fig=1000, desc=""):
    if group_size > 0 and w_data.shape[-1] % group_size != 0:
        print(f"Input channel: {w_data.shape[-1]} is not divisible by group_size: {group_size}")
        return
    
    # Prepare data based on group_size
    w_data_group = w_data.reshape(-1, group_size) if group_size > 0 else w_data

    # Prepare 3D visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare X, Y, and Z values
    group_indices = np.arange(w_data_group.shape[0])
    element_indices = np.arange(w_data_group.shape[1])
    X, Y = np.meshgrid(element_indices, group_indices)
    Z = np.abs(w_data_group.cpu().numpy())  # Use absolute value for better visualization

    # Plot surface
    percentile_range = [10, 99.7]  # 10%到99%的分位数范围
    z_min, z_max = np.percentile(Z, percentile_range)  
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8, vmin=z_min, vmax=z_max)
    # surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8, norm=LogNorm(vmin=z_min, vmax=z_max))
    # fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)  # 添加颜色条
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, extend='both')  # extend='both' 确保颜色条包含超出范围的值

    # 标注最大值
    max_idx = np.unravel_index(Z.argmax(), Z.shape)
    ax.text(X[max_idx], Y[max_idx], Z.max(), f'max: {Z.max():.2f}', color='red')
        # 添加分位数信息到图像顶部
    fig.text(
        0.5, 0.8,  # 中心对齐，位于顶部
        f"Percentile Range: {percentile_range[0]}% = {z_min:.4f}, {percentile_range[1]}% = {z_max:.4f}",
        fontsize=12,
        color='red',
        ha='center',  # 水平居中
        va='top',  # 垂直顶部对齐
        bbox=dict(boxstyle="round", edgecolor="black", facecolor="white", alpha=0.8)
    )

    # elev (Elevation): 从 z 轴方向观察的角度（俯仰角），默认值是 30°。越小视角越接近xy平面，值越大越类似俯视
    # azim (Azimuth): 绕 z 轴旋转的角度（方位角）。
    ax.view_init(elev=10, azim=20)

    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Output Dimension (Group Index)')
    ax.set_zlabel('Absolute Value of W')
    ax.set_title(f"Layer {layer_idx} - {layer_name} - 3D Distribution")

    # Save figure
    current_time = datetime.now().strftime("_%m%d%H%M")  # 格式: 月日时分 (_01071127)

    save_path = os.path.join(os.getcwd(), 'distri_img')
    os.makedirs(save_path, exist_ok=True)
    # file_name = f"layer{layer_idx}_{layer_name}_3d_{desc}.png"
    file_name = f"layer{layer_idx}_{layer_name}_3d_{desc}_{current_time}.png"
    plt.savefig(os.path.join(save_path, file_name), dpi=600)
    plt.close()

    print(f"3D distribution plot saved at: {os.path.join(save_path, file_name)}")


if __name__ == '__main__':
    w = torch.normal(0, 1, size=(4096, 4096))
    w_non_zero_mask = (w > 0.)
    w_non_zero = w_non_zero_mask * w + ~w_non_zero_mask * (1.0)
    print(w_non_zero.abs().min(), w.max(), w)
    distri_3d(w)

    # group_dist_outlier(w, group_size=16)
            # if i % 10 == 0:
            #     w = m.weight.data
            #     ic, oc = w.shape[1], w.shape[0]
                
            #     outlier_num = math.ceil(ic * outlier_ratio)
            #     outlier_masks_mw = torch.zeros_like(w, dtype=torch.int8).to(w.device)
            #     for column in range(oc):
            #         value, outlier_index = torch.topk(w[column].abs(), outlier_num)
            #         outlier_masks_mw[column][outlier_index] = 1
                
            #         # print("outlier value", value)
            #     outlier_masks_mw = outlier_masks_mw.reshape(w.shape)

            #     non_outlier_masks_mw = (outlier_masks_mw == 0).to(dtype=torch.int8)
            #     w = non_outlier_masks_mw * w  
            #     make_distribution_channel(m.weight.data, w, i, n, 30, "outlier") 
            #     group_dist_outlier(m.weight.data, w, 128,i, n, 30, "outlier") 