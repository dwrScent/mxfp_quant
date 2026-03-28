import argparse
import json
from pathlib import Path

import torch

DEFAULT_GROUP_SIZES = (2, 4, 8, 16, 32, 64)


def kurtosis_excess(groups: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mean = groups.mean(dim=1, keepdim=True)
    centered = groups - mean
    var = centered.pow(2).mean(dim=1, keepdim=True).clamp_min(eps)
    fourth = centered.pow(4).mean(dim=1, keepdim=True)
    return (fourth / var.pow(2)).squeeze(1) - 3.0


def to_tensor(payload, src: Path) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in ("tensor", "x", "data", "hidden_states"):
            if key in payload and isinstance(payload[key], torch.Tensor):
                return payload[key]
        for value in payload.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported .pt payload type for {src}: {type(payload)}")


def find_dump_dir(cli_dump_dir: str | None) -> Path:
    if cli_dump_dir:
        path = Path(cli_dump_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"dump dir not found: {path}")
        return path

    file_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "dump",
        file_path.parent / "dump",
    ]
    candidates.extend(parent / "dump" for parent in file_path.parents)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if candidate.is_dir() and any(candidate.glob("*.pt")):
            return candidate

    raise FileNotFoundError("Cannot find dump/ containing .pt files; set --dump-dir explicitly.")


def summarize_kurtosis(values: torch.Tensor) -> dict:
    values = values.float()
    return {
        "num_groups": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        # "p50": float(values.quantile(0.50).item()),
        # "p90": float(values.quantile(0.90).item()),
        # "p99": float(values.quantile(0.99).item()),
        "max": float(values.max().item()),
        "positive_ratio": float((values > 0).float().mean().item()),
    }


def grouped_view(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    flat = x.float().reshape(-1)
    usable = (flat.numel() // group_size) * group_size
    dropped_tail = flat.numel() - usable
    if usable == 0:
        return torch.empty(0, group_size), dropped_tail
    return flat[:usable].reshape(-1, group_size), dropped_tail


def init_overall_acc() -> dict:
    return {
        str(g): {
            "num_groups": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "positive_count": 0,
        }
        for g in DEFAULT_GROUP_SIZES
    }


def update_overall_acc(acc: dict, group_size: int, kurt_vals: torch.Tensor) -> None:
    state = acc[str(group_size)]
    v = kurt_vals.float()
    n = int(v.numel())
    state["num_groups"] += n
    state["sum"] += float(v.sum().item())
    state["sum_sq"] += float(v.pow(2).sum().item())
    state["min"] = min(state["min"], float(v.min().item()))
    state["max"] = max(state["max"], float(v.max().item()))
    state["positive_count"] += int((v > 0).sum().item())


def finalize_overall_acc(acc: dict) -> dict:
    out = {}
    for group_size, state in acc.items():
        n = state["num_groups"]
        if n == 0:
            out[group_size] = {"num_groups": 0}
            continue
        mean = state["sum"] / n
        variance = max(state["sum_sq"] / n - mean * mean, 0.0)
        out[group_size] = {
            "num_groups": n,
            "mean": mean,
            "std": variance ** 0.5,
            "min": state["min"],
            "max": state["max"],
            "positive_ratio": state["positive_count"] / n,
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-group kurtosis for group sizes 2/4/8/16/32/64 from dump/*.pt."
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Path to dump directory. If omitted, script tries to auto-detect.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="metrics_block",
        help="Where to save per-group kurtosis tensors and summary json.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Only process first N files (0 means all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_dir = find_dump_dir(args.dump_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(dump_dir.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {dump_dir}")
    if args.max_files > 0:
        pt_files = pt_files[: args.max_files]

    all_summary = {
        "dump_dir": str(dump_dir),
        "output_dir": str(output_dir),
        "group_sizes": list(DEFAULT_GROUP_SIZES),
        "files": {},
        "overall": {},
    }
    overall_acc = init_overall_acc()

    print(f"Found {len(pt_files)} pt files in {dump_dir}")
    for idx, pt_path in enumerate(pt_files, start=1):
        print(f"[{idx}/{len(pt_files)}] processing: {pt_path.name}")
        payload = torch.load(pt_path, map_location="cpu")
        tensor = to_tensor(payload, pt_path).float()

        file_stat = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "group_stats": {},
        }

        for group_size in DEFAULT_GROUP_SIZES:
            groups, dropped_tail = grouped_view(tensor, group_size)
            if groups.numel() == 0:
                file_stat["group_stats"][str(group_size)] = {
                    "num_groups": 0,
                    "dropped_tail": dropped_tail,
                }
                continue

            kurt_vals = kurtosis_excess(groups).cpu()
            out_name = f"{pt_path.stem}_group{group_size}_kurtosis.pt"
            torch.save(kurt_vals, output_dir / out_name)

            summary = summarize_kurtosis(kurt_vals)
            summary["dropped_tail"] = dropped_tail
            summary["kurtosis_file"] = out_name
            file_stat["group_stats"][str(group_size)] = summary
            update_overall_acc(overall_acc, group_size, kurt_vals)

        all_summary["files"][pt_path.name] = file_stat

    all_summary["overall"] = finalize_overall_acc(overall_acc)
    summary_path = output_dir / "kurtosis_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(f"\nDone. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
