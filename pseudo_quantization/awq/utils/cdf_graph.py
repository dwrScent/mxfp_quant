import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter
import csv
from scipy.interpolate import UnivariateSpline



cumulative_data = {}

def generate_cdf_points(x, y, x_new):
    y_new = np.interp(x_new, x, y)
    return y_new

def compute_cdf(data):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))

    # show the percentage
    yvals = yvals * 100
    return sorted_data, yvals

def smooth_curve(x, y, num_points=1000, window_length=175, polyorder=2):
    if len(x) < window_length:
        window_length = len(x) // 2 * 2 + 1  # Make sure window_length is odd and less than data length
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_smooth = savgol_filter(np.interp(x_new, x, y), window_length, polyorder)
    return x_new, y_smooth

def smooth_curve_group(x, y, num_points=1000, s=0.1):
    x_new = np.linspace(x.min(), x.max(), num_points)
    spline = UnivariateSpline(x, y, s=s)
    y_smooth = spline(x_new)
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

    # plt.title("CDF of Weights")
    # plt.ylabel('CDF')
    # plt.xlabel('Value')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.weight'] = 'bold'
    plt.xticks(np.arange(-1, 1.5, step=0.5), fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    if group_size == -2:
        data_list = w_data.view(-1).tolist()
        data_list = normalize_data(data_list)  # Normalize data
        assert np.max(data_list) <= 1 and np.min(data_list) >= -1

        sorted_data, yvals = compute_cdf(data_list)
        x_smooth, y_smooth = smooth_curve(sorted_data, yvals, window_length=5)
        if accumulate:
            cumulative_data[layer_name][desc].append((x_smooth, y_smooth, f"Tensor-level CDF {layer_idx}"))
        else:
            plt.plot(x_smooth, y_smooth, label=f"Tensor-level CDF {layer_idx}", alpha=0.7, linewidth=1.5)  # Adjusted line width
    else:
        for idx in range(0, len(w_data_group), stride):
            # if idx >= max_fig:
            #     print(f"Up to the max number of figures: {max_fig}")
            #     break
            group = w_data_group[idx]
            data_list = group.view(-1).tolist()
            data_list = normalize_data(data_list)  # Normalize data
            # print(np.max(data_list), np.min(data_list))
            assert np.max(data_list) <= 1 and np.min(data_list) >= -1
            sorted_data, yvals = compute_cdf(data_list)
            # print(group_size, )
            if group_size > 0:
                x_smooth, y_smooth = smooth_curve(sorted_data, yvals, num_points=1000, window_length=101, polyorder=3)
                # x_smooth, y_smooth = smooth_curve_group(sorted_data, yvals, num_points=1000, s=10)
            else:
                x_smooth, y_smooth = smooth_curve(sorted_data, yvals, window_length=21)
            
            if accumulate:
                cumulative_data[layer_name][desc].append((x_smooth, y_smooth, f"Group {idx} {layer_idx}"))
            else:
                plt.plot(x_smooth, y_smooth, alpha=0.7, linewidth=1.5, label=f"Group {idx}")  # Adjusted line width

    # plt.gca().xaxis.set_ticklabels([])  # Hide x-axis labels
    # plt.gca().yaxis.set_ticklabels([])  # Hide y-axis labels

    plt.tight_layout(pad=0.1)
    if not accumulate:
        plt.savefig(f'{save_path}/layer{layer_idx}_{layer_name}_{desc}.png')
        # plt.savefig(f'{save_path}/layer_{layer_name}_{desc}.pdf', format='pdf', dpi=600)
        plt.show()
    elif plot_final:
        for x_smooth, y_smooth, label in cumulative_data[layer_name][desc]:
            plt.plot(x_smooth, y_smooth, alpha=0.7, linewidth=1.5, label=label)  # Adjusted line width
        # plt.savefig(f'{save_path}/layer_{layer_name}_{desc}.png')
        plt.savefig(f'{save_path}/layer_{layer_name}_{desc}.pdf', format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        # Clear data after plotting
        cumulative_data[layer_name][desc] = []

def cdf_csv(w_data, group_size=-1, layer_idx=0, layer_name="", num_points=1000, stride=1, filename='cdf_data.csv'):
    # Generate a fixed x_new that will be used for all CDFs
    x_new = np.linspace(-1, 1, num_points)
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write metadata including layer_id, layer_name, and stride
        writer.writerow([layer_idx, layer_name, 'stride', stride])
        writer.writerow(x_new)
        
        if group_size == -2:
            # Tensor-level granularity
            print(w_data.shape)
            data_list = w_data.view(-1).tolist()
            data_list = normalize_data(data_list)
            assert np.max(data_list) <= 1 and np.min(data_list) >= -1

            sorted_data, yvals = compute_cdf(data_list)
            y_new = generate_cdf_points(sorted_data, yvals, x_new)

            writer.writerow(y_new)
        
        elif group_size == -1:
            # Channel-level granularity
            for i in range(0, w_data.size(0), stride):
                channel = w_data[i]
                data_list = channel.view(-1).tolist()
                data_list = normalize_data(data_list)
                assert np.max(data_list) <= 1 and np.min(data_list) >= -1

                sorted_data, yvals = compute_cdf(data_list)
                y_new = generate_cdf_points(sorted_data, yvals, x_new)

                writer.writerow(y_new)
        
        else:
            # Group-level granularity
            if w_data.shape[-1] % group_size != 0:
                print(f"Input channel: {w_data.shape[-1]} is not divisible by group_size: {group_size}")

                cut_size = w_data.shape[-1] // group_size
                w_data = w_data[:, :cut_size*group_size]
                print(f"cut it to {w_data.shape[-1]}")
                # return

            w_data_group = w_data.view(-1, group_size)

            for i in range(0, len(w_data_group), stride):
                group = w_data_group[i]
                data_list = group.view(-1).tolist()
                data_list = normalize_data(data_list)
                assert np.max(data_list) <= 1 and np.min(data_list) >= -1

                sorted_data, yvals = compute_cdf(data_list)
                y_new = generate_cdf_points(sorted_data, yvals, x_new)

                writer.writerow(y_new)
                
if __name__ == '__main__':
    w = torch.normal(0, 5, size=(128, 512))
    # group_cdf(w, group_size=-2)
    group_cdf(w, group_size=-2, stride=16)
    # group_cdf(w, group_size=-1, stride=16)
