import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter

cumulative_data = {}

def compute_cdf(data):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
    return sorted_data, yvals

def smooth_curve(x, y, num_points=1000, window_length=175, polyorder=2):
    if len(x) < window_length:
        window_length = len(x) // 2 * 2 + 1  # Make sure window_length is odd and less than data length
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_smooth = savgol_filter(np.interp(x_new, x, y), window_length, polyorder)
    return x_new, y_smooth

def normalize_data(data):
    max_val = np.max(np.abs(data))
    return data / max_val

def group_cdf(w_data, group_size=-1, layer_idx=0, layer_name="", max_fig=1000, desc="", stride=1, accumulate=False, plot_final=False):
    # Prepare data based on group_size
    if group_size > 0 and w_data.shape[-1] % group_size != 0:
        print(f"Input channel: {w_data.shape[-1]} is not divisible by group_size: {group_size}")
        return
    elif group_size == -1:
        w_data_group = w_data
    elif group_size > 0:
        w_data_group = w_data.reshape(-1, group_size)
    else:
        pass
    
    global cumulative_data

    if accumulate:
        if layer_name not in cumulative_data:
            cumulative_data[layer_name] = {}
        if desc not in cumulative_data[layer_name]:
            cumulative_data[layer_name][desc] = []

    # Create directories for saving plots
    save_path = os.path.join(os.getcwd(), 'cdf_img')
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 6))  # Adjusted figure size
    plt.title("CDF of Weights")
    plt.ylabel('CDF')
    plt.xlabel('Value')

    if group_size == -2:
        data_list = w_data.view(-1).tolist()
        data_list = normalize_data(data_list)  # Normalize data
        sorted_data, yvals = compute_cdf(data_list)
        x_smooth, y_smooth = smooth_curve(sorted_data, yvals, window_length=10)
        if accumulate:
            cumulative_data[layer_name][desc].append((x_smooth, y_smooth, f"Tensor-level CDF {layer_idx}"))
        else:
            plt.plot(x_smooth, y_smooth, label=f"Tensor-level CDF {layer_idx}", alpha=0.7, linewidth=2)  # Adjusted line width
    else:
        for idx in range(0, len(w_data_group), stride):
            if idx >= max_fig:
                print(f"Up to the max number of figures: {max_fig}")
                break
            group = w_data_group[idx]
            data_list = group.view(-1).tolist()
            data_list = normalize_data(data_list)  # Normalize data
            sorted_data, yvals = compute_cdf(data_list)
            if group_size > 0 and w_data.shape[-1] % group_size != 0:
                x_smooth, y_smooth = smooth_curve(sorted_data, yvals, num_points=2000, window_length=150)
            else:
                x_smooth, y_smooth = smooth_curve(sorted_data, yvals, window_length=20)
            
            if accumulate:
                cumulative_data[layer_name][desc].append((x_smooth, y_smooth, f"Group {idx} {layer_idx}"))
            else:
                plt.plot(x_smooth, y_smooth, alpha=0.7, linewidth=2, label=f"Group {idx}")  # Adjusted line width

    if not accumulate:
        plt.savefig(f'{save_path}/layer{layer_idx}_{layer_name}_{desc}.png')
        plt.show()
    elif plot_final:
        for x_smooth, y_smooth, label in cumulative_data[layer_name][desc]:
            plt.plot(x_smooth, y_smooth, alpha=0.7, linewidth=2, label=label)  # Adjusted line width
        plt.savefig(f'{save_path}/layer_{layer_name}_{desc}.png')
        plt.show()
        # Clear data after plotting
        cumulative_data[layer_name][desc] = []

if __name__ == '__main__':
    w = torch.normal(0, 5, size=(128, 512))
    # group_cdf(w, group_size=-2)
    group_cdf(w, group_size=-2, stride=16)
    # group_cdf(w, group_size=-1, stride=16)
