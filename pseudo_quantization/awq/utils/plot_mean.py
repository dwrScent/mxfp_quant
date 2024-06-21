import os
import numpy as np
import matplotlib.pyplot as plt
import torch

cumulative_data = {}

def normalize_data(data):
    max_val = np.max(np.abs(data))
    return data / max_val

def compute_mean_variance(data):
    mean = np.mean(data)
    variance = np.var(data)
    return mean, variance

def group_mean_variance(w_data, group_size=-1, layer_idx=0, layer_name="", max_fig=1000, desc="", stride=1, accumulate=False, plot_final=False):
    global cumulative_data

    if accumulate:
        if layer_name not in cumulative_data:
            cumulative_data[layer_name] = {}
        if desc not in cumulative_data[layer_name]:
            cumulative_data[layer_name][desc] = []

    if w_data is not None:
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

        if group_size == -2:
            data_list = w_data.view(-1).tolist()
            mean, variance = compute_mean_variance(data_list)
            if accumulate:
                cumulative_data[layer_name][desc].append((mean, variance))
        else:
            for idx in range(0, len(w_data_group), stride):
                # if idx >= max_fig:
                #     print(f"Up to the max number of figures: {max_fig}")
                #     break
                group = w_data_group[idx]
                data_list = group.view(-1).tolist()
                data_list = normalize_data(data_list)  # Normalize data
                mean, variance = compute_mean_variance(data_list)
                if accumulate:
                    cumulative_data[layer_name][desc].append((mean, variance))

    if plot_final and accumulate:
        plot_mean_variance(layer_name, desc)

def plot_mean_variance(layer_name, desc):
    if layer_name not in cumulative_data or desc not in cumulative_data[layer_name]:
        print(f"No data for layer_name: {layer_name}, desc: {desc}")
        return

    data = cumulative_data[layer_name][desc]
    means = [item[0] for item in data]
    variances = [item[1] for item in data]
    x = np.arange(len(data))

    plt.figure(figsize=(8, 6))
    plt.plot(x, means, label='Mean', linewidth=2)
    plt.plot(x, variances, label='Variance', linewidth=2)
    # plt.xlabel('Sample Index', fontsize=16, fontweight='bold')
    # plt.ylabel('Value', fontsize=16, fontweight='bold')
    # plt.title(f'Mean and Variance for {layer_name} - {desc}', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    # 调整横坐标的刻度
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.0f}'.format(val)))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # plt.legend(fontsize=14)
    plt.tight_layout(pad=0.1)
    save_path = os.path.join(os.getcwd(), 'mean_variance_img')
    os.makedirs(save_path, exist_ok=True)
    # plt.savefig(f'{save_path}/{layer_name}_{desc}_mean_variance.png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'{save_path}/{layer_name}_{desc}_mean_variance.pdf', format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    w = torch.normal(0, 5, size=(128, 512))
    group_mean_variance(w, group_size=-2, layer_idx=0, layer_name='layer0', desc='tensor', stride=16, accumulate=True)
    group_mean_variance(w, group_size=-2, layer_idx=1, layer_name='layer0', desc='tensor', stride=16, accumulate=True, plot_final=True)
