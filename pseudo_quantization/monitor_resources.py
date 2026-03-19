#!/usr/bin/env python3
"""Lightweight CPU/GPU/OOM monitor.

Usage examples:
  python monitor_resources.py --host-only
  python monitor_resources.py --pid 12345
  python monitor_resources.py --cmd "bash reasoning_pangu.sh SuperGPQA"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def read_first(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def parse_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                key, rest = line.split(":", 1)
                val = rest.strip().split()[0]
                out[key] = int(val)  # kB
    except Exception:
        pass
    return out


def read_proc_stat() -> tuple[int, int]:
    with open("/proc/stat", "r") as f:
        first = f.readline().strip().split()
    nums = [int(x) for x in first[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    total = sum(nums)
    return total, idle


def get_cpu_percent(prev: tuple[int, int] | None) -> tuple[float, tuple[int, int]]:
    cur = read_proc_stat()
    if prev is None:
        return 0.0, cur
    total_d = cur[0] - prev[0]
    idle_d = cur[1] - prev[1]
    if total_d <= 0:
        return 0.0, cur
    busy = max(0.0, 100.0 * (1.0 - idle_d / total_d))
    return busy, cur


def proc_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def proc_rss_vms_mb(pid: int) -> tuple[float, float]:
    status = Path(f"/proc/{pid}/status")
    rss_kb = 0
    vms_kb = 0
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("VmSize:"):
                vms_kb = int(line.split()[1])
    except Exception:
        return 0.0, 0.0
    return rss_kb / 1024.0, vms_kb / 1024.0


def read_cgroup_oom() -> dict[str, int]:
    candidates = [
        "/sys/fs/cgroup/memory.events",
        "/sys/fs/cgroup/memory.events.local",
    ]
    result = {"oom": -1, "oom_kill": -1}
    for path in candidates:
        txt = read_first(path)
        if not txt:
            continue
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            k, v = parts
            if k in result:
                try:
                    result[k] = int(v)
                except ValueError:
                    pass
        if result["oom"] != -1 or result["oom_kill"] != -1:
            break
    return result


def query_nvidia() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2)
        rows = [",".join(x.strip().split(", ")) for x in out.strip().splitlines() if x.strip()]
        return ";".join(rows) if rows else "NA"
    except Exception:
        return "NA"


def query_npu() -> str:
    # Keep this best-effort: just store one-line summary if npu-smi exists.
    try:
        out = subprocess.check_output(["npu-smi", "info"], stderr=subprocess.DEVNULL, text=True, timeout=2)
        line = " ".join(out.strip().splitlines()[:2])
        return line[:400] if line else "NA"
    except Exception:
        return "NA"


def build_paths(log_dir: Path, tag: str) -> tuple[Path, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{tag}_{stamp}"
    return log_dir / f"{prefix}.csv", log_dir / f"{prefix}.events.log"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor CPU/GPU usage and OOM events.")
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--host-only", action="store_true", help="Monitor whole host only (no target PID).")
    mode.add_argument("--pid", type=int, help="PID to monitor.")
    mode.add_argument("--cmd", type=str, help="Command to launch and monitor.")
    p.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    p.add_argument("--log-dir", type=str, default="./outputs/monitor_logs", help="Directory for logs.")
    p.add_argument("--tag", type=str, default="resource_monitor", help="Filename tag.")
    args = p.parse_args()
    if not args.host_only and args.pid is None and args.cmd is None:
        p.error("one of --host-only / --pid / --cmd is required")
    return args


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path, event_path = build_paths(log_dir, args.tag)

    child: subprocess.Popen[str] | None = None
    target_pid = -1
    if args.cmd:
        print(f"[{now_iso()}] launching: {args.cmd}")
        child = subprocess.Popen(shlex.split(args.cmd))
        target_pid = child.pid
    elif args.pid is not None:
        target_pid = args.pid

    stop = False

    def _handle_sig(signum, _frame):
        nonlocal stop
        stop = True
        print(f"\n[{now_iso()}] received signal {signum}, stopping monitor...")

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    mode_name = "host-only" if args.host_only else ("cmd" if args.cmd else "pid")
    print(f"[{now_iso()}] mode={mode_name} target_pid={target_pid}")
    print(f"[{now_iso()}] csv={csv_path}")
    print(f"[{now_iso()}] events={event_path}")

    headers = [
        "timestamp",
        "pid",
        "pid_alive",
        "host_cpu_percent",
        "host_mem_used_gb",
        "host_mem_total_gb",
        "host_mem_used_percent",
        "swap_used_gb",
        "load1",
        "load5",
        "load15",
        "pid_rss_mb",
        "pid_vms_mb",
        "cgroup_oom",
        "cgroup_oom_kill",
        "gpu_nvidia",
        "gpu_npu",
    ]

    prev_cpu = None
    prev_oom = read_cgroup_oom()

    with csv_path.open("w", newline="") as cf, event_path.open("w") as ef:
        writer = csv.DictWriter(cf, fieldnames=headers)
        writer.writeheader()
        ef.write(f"[{now_iso()}] start monitor mode={mode_name} pid={target_pid}\n")

        while not stop:
            alive = proc_alive(target_pid) if target_pid > 0 else False
            if args.pid is not None and not alive:
                ef.write(f"[{now_iso()}] pid {target_pid} exited\n")
                ef.flush()
                break

            mem = parse_meminfo()
            mem_total = mem.get("MemTotal", 0) / (1024.0 * 1024.0)
            mem_avail = mem.get("MemAvailable", 0) / (1024.0 * 1024.0)
            mem_used = max(0.0, mem_total - mem_avail)
            mem_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0
            swap_total = mem.get("SwapTotal", 0) / (1024.0 * 1024.0)
            swap_free = mem.get("SwapFree", 0) / (1024.0 * 1024.0)
            swap_used = max(0.0, swap_total - swap_free)

            cpu_pct, prev_cpu = get_cpu_percent(prev_cpu)
            load1, load5, load15 = os.getloadavg()
            rss_mb, vms_mb = proc_rss_vms_mb(target_pid) if alive else (0.0, 0.0)
            oom = read_cgroup_oom()

            if prev_oom and (
                oom.get("oom", -1) > prev_oom.get("oom", -1)
                or oom.get("oom_kill", -1) > prev_oom.get("oom_kill", -1)
            ):
                ef.write(
                    f"[{now_iso()}] OOM counter changed: "
                    f"oom {prev_oom.get('oom')} -> {oom.get('oom')}, "
                    f"oom_kill {prev_oom.get('oom_kill')} -> {oom.get('oom_kill')}\n"
                )
                ef.flush()
            prev_oom = oom

            row = {
                "timestamp": now_iso(),
                "pid": target_pid if target_pid > 0 else -1,
                "pid_alive": int(alive),
                "host_cpu_percent": f"{cpu_pct:.2f}",
                "host_mem_used_gb": f"{mem_used:.2f}",
                "host_mem_total_gb": f"{mem_total:.2f}",
                "host_mem_used_percent": f"{mem_pct:.2f}",
                "swap_used_gb": f"{swap_used:.2f}",
                "load1": f"{load1:.2f}",
                "load5": f"{load5:.2f}",
                "load15": f"{load15:.2f}",
                "pid_rss_mb": f"{rss_mb:.2f}",
                "pid_vms_mb": f"{vms_mb:.2f}",
                "cgroup_oom": oom.get("oom", -1),
                "cgroup_oom_kill": oom.get("oom_kill", -1),
                "gpu_nvidia": query_nvidia(),
                "gpu_npu": query_npu(),
            }
            writer.writerow(row)
            cf.flush()

            if child is not None and child.poll() is not None:
                ef.write(f"[{now_iso()}] child exited with code={child.returncode}\n")
                ef.flush()
                break

            time.sleep(max(0.1, args.interval))

    if child is not None:
        return int(child.returncode or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
