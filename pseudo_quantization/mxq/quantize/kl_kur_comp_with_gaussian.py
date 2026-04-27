import os

import matplotlib.pyplot as plt
import torch

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = 2 ** (-9)
FLOAT8_E4M3_MAX = 448.0


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
            elif subnormal:
                values.append((2 ** (1 - bias)) * (j * 2 ** (-man_bit)))
            else:
                values.append((2 ** (i - bias)) * (1 + j * 2 ** (-man_bit)))
        subnormal = False
    return values


FP4_E2M1_GRID = torch.tensor(float_value(2, 1))


def quantize_to_grid(x: torch.Tensor, levels: torch.Tensor):
    levels = levels.to(x.device)
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    odd_boundaries = boundaries[1::2]
    mask = torch.isin(x, odd_boundaries)
    x = x + 0.0000005 * mask
    indices = torch.bucketize(x, boundaries)
    indices.clamp_(0, len(levels) - 1)
    return levels[indices], indices


def cast_to_fp4(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    x_quant, _ = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    return x_quant * sign


def pair_groups(groups: torch.Tensor):
    if groups.shape[0] < 2:
        return None, None
    even_rows = groups.shape[0] - (groups.shape[0] % 2)
    if even_rows == 0:
        return None, None
    groups = groups[:even_rows]
    return groups[0::2], groups[1::2]


def elem_prob(x: torch.Tensor, eps: float = 1e-12):
    x = x.abs()
    return x / (x.sum(dim=1, keepdim=True) + eps)


def cross_entropy_from_prob(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12):
    q = q.clamp_min(eps)
    return (-(p * torch.log2(q)).sum(dim=1)).mean().item()


def kl_div_from_prob(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (torch.log2(p) - torch.log2(q))).sum(dim=1).mean().item()


def hist_prob_from_pairs(p_vals: torch.Tensor, q_vals: torch.Tensor, num_bins: int):
    mins = torch.minimum(p_vals.min(dim=1).values, q_vals.min(dim=1).values)
    maxs = torch.maximum(p_vals.max(dim=1).values, q_vals.max(dim=1).values)
    maxs = torch.where(maxs == mins, mins + 1e-6, maxs)

    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=p_vals.device)
    p01 = (p_vals - mins.unsqueeze(1)) / (maxs - mins).unsqueeze(1)
    q01 = (q_vals - mins.unsqueeze(1)) / (maxs - mins).unsqueeze(1)
    p_idx = torch.bucketize(p01, edges) - 1
    q_idx = torch.bucketize(q01, edges) - 1
    p_idx = p_idx.clamp(0, num_bins - 1)
    q_idx = q_idx.clamp(0, num_bins - 1)

    n_pair, n_elem = p_idx.shape
    pair_idx = torch.arange(n_pair, device=p_vals.device).unsqueeze(1).expand(n_pair, n_elem)
    p_flat = pair_idx.reshape(-1) * num_bins + p_idx.reshape(-1)
    q_flat = pair_idx.reshape(-1) * num_bins + q_idx.reshape(-1)

    p_hist = torch.bincount(p_flat, minlength=n_pair * num_bins).reshape(n_pair, num_bins).float()
    q_hist = torch.bincount(q_flat, minlength=n_pair * num_bins).reshape(n_pair, num_bins).float()
    p_prob = p_hist / p_hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    q_prob = q_hist / q_hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    return p_prob, q_prob


def mutual_information_hist(p_vals: torch.Tensor, q_vals: torch.Tensor, num_bins: int = 16, eps: float = 1e-12):
    n_pair, n_elem = p_vals.shape
    mins = torch.minimum(p_vals.min(dim=1).values, q_vals.min(dim=1).values)
    maxs = torch.maximum(p_vals.max(dim=1).values, q_vals.max(dim=1).values)
    maxs = torch.where(maxs == mins, mins + 1e-6, maxs)

    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=p_vals.device)
    p01 = (p_vals - mins.unsqueeze(1)) / (maxs - mins).unsqueeze(1)
    q01 = (q_vals - mins.unsqueeze(1)) / (maxs - mins).unsqueeze(1)
    p_idx = (torch.bucketize(p01, edges) - 1).clamp(0, num_bins - 1)
    q_idx = (torch.bucketize(q01, edges) - 1).clamp(0, num_bins - 1)

    pair_idx = torch.arange(n_pair, device=p_vals.device).unsqueeze(1).expand(n_pair, n_elem)
    joint_flat = pair_idx.reshape(-1) * (num_bins * num_bins) + (p_idx * num_bins + q_idx).reshape(-1)
    joint = torch.bincount(joint_flat, minlength=n_pair * num_bins * num_bins).reshape(n_pair, num_bins, num_bins).float()
    joint = joint / joint.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    px = joint.sum(dim=2, keepdim=True)
    py = joint.sum(dim=1, keepdim=True)
    ratio = joint / (px * py + eps)
    mi = (joint * torch.log2(ratio.clamp_min(eps))).sum(dim=(1, 2))
    return mi.mean().item()


def wasserstein_empirical(p_vals: torch.Tensor, q_vals: torch.Tensor):
    p_sorted, _ = torch.sort(p_vals, dim=1)
    q_sorted, _ = torch.sort(q_vals, dim=1)
    p_sorted = p_vals
    q_sorted = q_vals
    return torch.mean(torch.abs(p_sorted - q_sorted), dim=1).mean().item()


def kurtosis_excess(x: torch.Tensor, eps: float = 1e-12):
    mean = x.mean(dim=1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=1, keepdim=True).clamp_min(eps)
    fourth = ((x - mean) ** 4).mean(dim=1, keepdim=True)
    return (fourth / (var ** 2)).squeeze(1) - 3.0


def append_group_metrics(groups: torch.Tensor, acc: dict, hist_bins: int = 16):
    p_vals, q_vals = pair_groups(groups)
    if p_vals is None:
        return

    p_elem = elem_prob(p_vals)
    q_elem = elem_prob(q_vals)
    p_sorted = torch.sort(p_vals.abs(), dim=1, descending=True).values
    q_sorted = torch.sort(q_vals.abs(), dim=1, descending=True).values
    p_elem_sorted = elem_prob(p_sorted)
    q_elem_sorted = elem_prob(q_sorted)
    p_hist, q_hist = hist_prob_from_pairs(p_vals, q_vals, num_bins=hist_bins)

    acc["kl_elem"].append(kl_div_from_prob(p_elem, q_elem))
    acc["kl_hist"].append(kl_div_from_prob(p_hist, q_hist))
    acc["ce_elem"].append(cross_entropy_from_prob(p_elem, q_elem))
    acc["ce_hist"].append(cross_entropy_from_prob(p_hist, q_hist))
    acc["kl_elem_sorted"].append(kl_div_from_prob(p_elem_sorted, q_elem_sorted))
    acc["ce_elem_sorted"].append(cross_entropy_from_prob(p_elem_sorted, q_elem_sorted))
    acc["wasserstein"].append(wasserstein_empirical(p_vals, q_vals))
    acc["mutual_info"].append(mutual_information_hist(p_vals, q_vals, num_bins=hist_bins))
    acc["kurtosis_avg"].append(kurtosis_excess(groups).mean().item())


def plot_metric(x_axis, good, bad, title, ylabel, out_path):
    plt.figure(figsize=(12, 7))
    plt.plot(x_axis, good, label="Good Groups", marker="o")
    plt.plot(x_axis, bad, label="Bad Groups", marker="o")
    plt.title(title)
    plt.xlabel("x (sigma = 0.01 * 2^(x/2))")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_histogram(data_good, data_bad, title, xlabel, out_path, bins=100):
    plt.figure(figsize=(12, 7))
    plt.hist(data_good.cpu().numpy(), bins=bins, alpha=0.6, label="Good Groups", density=True)
    plt.hist(data_bad.cpu().numpy(), bins=bins, alpha=0.6, label="Bad Groups", density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def init_acc():
    return {
        "kl_elem": [],
        "kl_hist": [],
        "ce_elem": [],
        "ce_hist": [],
        "kl_elem_sorted": [],
        "ce_elem_sorted": [],
        "wasserstein": [],
        "mutual_info": [],
        "kurtosis_avg": [],
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs("dump", exist_ok=True)

    x_axis = []
    good_acc = init_acc()
    bad_acc = init_acc()
    good_groups_abs = []
    bad_groups_abs = []
    good_kurt_all = []
    bad_kurt_all = []
    mse_good_acc = []
    mse_bad_acc = []

    group_size = int(os.getenv("GROUP_SIZE", "8"))
    l1_group_size = group_size * 2
    n_values = int(os.getenv("N_VALUES", str(8192 * 256)))
    min_j = int(os.getenv("MIN_J", "1"))
    max_j = int(os.getenv("MAX_J", "33"))

    for j in range(min_j, max_j + 1):
        x_val = j / 2
        x_axis.append(x_val)
        sigma = 0.01 * 2 ** (j / 2)

        tensor_value = torch.randn(n_values, device=device) * sigma
        tensor_value = tensor_value.float().reshape(-1, group_size)
        tensor_value = tensor_value.reshape(-1, l1_group_size)

        max_val = tensor_value.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scales = max_val / FLOAT4_E2M1_MAX
        global_scale = scales.max() / FLOAT8_E4M3_MAX
        scales = (
            (scales / global_scale)
            .clamp(min=FLOAT8_E4M3_EPS)
            .to(torch.float8_e4m3fn)
            .to(tensor_value.dtype)
        ) * global_scale

        tensor_quant = cast_to_fp4(tensor_value / scales) * scales
        mse_per_l1_group = (tensor_quant - tensor_value).pow(2).mean(dim=1)
        good_threshold = torch.quantile(mse_per_l1_group, 0.4)
        bad_threshold = torch.quantile(mse_per_l1_group, 0.6)

        # also collect mse of good and bad groups
        mse_good = mse_per_l1_group[mse_per_l1_group < good_threshold]
        mse_bad = mse_per_l1_group[mse_per_l1_group > bad_threshold]
        mse_good_acc.append(mse_good.cpu().mean() / sigma ** 2)
        mse_bad_acc.append(mse_bad.cpu().mean() / sigma ** 2)

        tensor_value_norm = tensor_value / scales
        good_groups = tensor_value_norm[mse_per_l1_group < good_threshold].reshape(-1, group_size)
        bad_groups = tensor_value_norm[mse_per_l1_group > bad_threshold].reshape(-1, group_size)

        good_groups_abs.append(good_groups.abs().detach().cpu().reshape(-1))
        bad_groups_abs.append(bad_groups.abs().detach().cpu().reshape(-1))

        good_kurt_all.append(kurtosis_excess(good_groups).detach().cpu())
        bad_kurt_all.append(kurtosis_excess(bad_groups).detach().cpu())

        append_group_metrics(good_groups, good_acc, hist_bins=16)
        append_group_metrics(bad_groups, bad_acc, hist_bins=16)

    plot_metric(
        x_axis,
        mse_good_acc,
        mse_bad_acc,
        "mse Comparison",
        "mse",
        "dump/mse_comparison_good_bad.png",
    )

    plot_metric(
        x_axis,
        good_acc["ce_elem"],
        bad_acc["ce_elem"],
        "Cross Entropy Comparison (elem/sum probability)",
        "Cross Entropy",
        "dump/cross_entropy_elem_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["ce_hist"],
        bad_acc["ce_hist"],
        "Cross Entropy Comparison (histogram probability)",
        "Cross Entropy",
        "dump/cross_entropy_hist_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["ce_elem_sorted"],
        bad_acc["ce_elem_sorted"],
        "Cross Entropy Comparison (elem/sum + sorted)",
        "Cross Entropy",
        "dump/cross_entropy_elem_sorted_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["wasserstein"],
        bad_acc["wasserstein"],
        "Wasserstein Distance Comparison",
        "Wasserstein Distance",
        "dump/wasserstein_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["mutual_info"],
        bad_acc["mutual_info"],
        "Mutual Information Comparison",
        "Mutual Information",
        "dump/mutual_information_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["kl_elem"],
        bad_acc["kl_elem"],
        "KL Divergence Comparison (elem/sum probability)",
        "KL Divergence",
        "dump/kl_divergence_elem_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["kl_elem_sorted"],
        bad_acc["kl_elem_sorted"],
        "KL Divergence Comparison (elem/sum + sorted)",
        "KL Divergence",
        "dump/kl_divergence_elem_sorted_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["kl_hist"],
        bad_acc["kl_hist"],
        "KL Divergence Comparison (histogram probability)",
        "KL Divergence",
        "dump/kl_divergence_hist_comparison.png",
    )
    plot_metric(
        x_axis,
        good_acc["kurtosis_avg"],
        bad_acc["kurtosis_avg"],
        "Kurtosis Comparison",
        "Excess Kurtosis",
        "dump/kurtosis_avg_comparison.png",
    )

    plot_histogram(
        torch.cat(good_groups_abs),
        torch.cat(bad_groups_abs),
        "Histogram of Absolute Values in Good/Bad Groups",
        "Absolute Value",
        "dump/good_bad_groups_data_histogram_comparison.png",
    )
    plot_histogram(
        torch.cat(good_kurt_all),
        torch.cat(bad_kurt_all),
        "Kurtosis Distribution of Good/Bad Groups",
        "Excess Kurtosis",
        "dump/kurtosis_distribution_comparison.png",
    )

    print("Saved plots to dump/:")
    print("- cross_entropy_elem_comparison.png")
    print("- cross_entropy_hist_comparison.png")
    print("- cross_entropy_elem_sorted_comparison.png")
    print("- wasserstein_comparison.png")
    print("- mutual_information_comparison.png")
    print("- kl_divergence_elem_comparison.png")
    print("- kl_divergence_hist_comparison.png")
    print("- kl_divergence_elem_sorted_comparison.png")
    print("- kurtosis_avg_comparison.png")
    print("- good_bad_groups_data_histogram_comparison.png")
    print("- kurtosis_distribution_comparison.png")


if __name__ == "__main__":
    main()
