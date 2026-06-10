#!/usr/bin/env python3
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RTL_AREA_POWER = ROOT.parent / "rtl_area_power"
PPA_DIR = ROOT / "configs" / "ppa"
FREQUENCY_HZ = 500_000_000

AREA_RE = re.compile(r"Total cell area:\s*([0-9.]+)")
POWER_RE = re.compile(r"Total Dynamic Power\s*=\s*([0-9.]+)\s*([munp]?W)")
LEAK_RE = re.compile(r"Cell Leakage Power\s*=\s*([0-9.]+)\s*([munp]?W)")

FIELDNAMES = [
    "Max Precision (bits)",
    "Min Precision (bits)",
    "N",
    "M",
    "Area (um^2)",
    "Dynamic Power (nW)",
    "Leakage Power (nW)",
    "Frequency",
]

UNIT_TO_NW = {
    "mW": 1_000_000.0,
    "uW": 1_000.0,
    "nW": 1.0,
    "pW": 0.001,
    "W": 1_000_000_000.0,
}


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def parse_area(path):
    match = AREA_RE.search(read_text(path))
    if match is None:
        raise ValueError(f"missing Total cell area in {path}")
    return float(match.group(1))


def parse_power(path):
    text = read_text(path)
    dynamic = POWER_RE.search(text)
    leakage = LEAK_RE.search(text)
    if dynamic is None:
        raise ValueError(f"missing Total Dynamic Power in {path}")
    if leakage is None:
        raise ValueError(f"missing Cell Leakage Power in {path}")

    dynamic_nw = float(dynamic.group(1)) * UNIT_TO_NW[dynamic.group(2)]
    leakage_nw = float(leakage.group(1)) * UNIT_TO_NW[leakage.group(2)]
    return dynamic_nw, leakage_nw


def make_row(pmax, pmin, area_report, power_report, normalize=1):
    area = parse_area(RTL_AREA_POWER / area_report) / normalize
    dynamic, leakage = parse_power(RTL_AREA_POWER / power_report)
    return {
        "Max Precision (bits)": pmax,
        "Min Precision (bits)": pmin,
        "N": 1,
        "M": 1,
        "Area (um^2)": area,
        "Dynamic Power (nW)": dynamic / normalize,
        "Leakage Power (nW)": leakage / normalize,
        "Frequency": FREQUENCY_HZ,
    }


def fmt(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row[key]) for key in FIELDNAMES})


def main():
    ant_olive_rows = [
        make_row(
            8,
            4,
            "result/baselines/ant_olive/ant_pe_fusion_area_report.txt",
            "result/baselines/ant_olive/ant_pe_fusion_power_report.txt",
        )
    ]

    mant_rows = [
        make_row(
            8,
            2,
            "result/baselines/mant/pe88_withFusion_withShift_area_report.txt",
            "result/baselines/mant/pe88_withFusion_withShift_power_report.txt",
        )
    ]

    microscopiq_rows = [
        make_row(
            8,
            2,
            "result/baselines_deprecated/microscopiq/pe_microscopiq_area_report.txt",
            "result/baselines_deprecated/microscopiq/pe_microscopiq_power_report.txt",
        )
    ]

    m2xfp_rows = [
        make_row(
            4,
            4,
            "result/asplos26_28nm/pe_tile_v/area_28nm_report.txt",
            "result/asplos26_28nm/pe_tile_v/power_28nm_report.txt",
            normalize=8,
        )
    ]

    nvesm2_rows = [
        make_row(
            4,
            4,
            "result/nvesm2/pe_tile_v/pe_tile_nvfp_fp32_45nm_area_report.txt",
            "result/nvesm2/pe_tile_v/pe_tile_nvfp_fp32_45nm_power_report.txt",
            normalize=8,
        )
    ]


    write_csv(PPA_DIR / "systolic_array_synth_ant.csv", ant_olive_rows)
    write_csv(PPA_DIR / "systolic_array_synth_olive.csv", ant_olive_rows)
    write_csv(PPA_DIR / "systolic_array_synth_mant.csv", mant_rows)
    write_csv(PPA_DIR / "systolic_array_synth_microscopiq.csv", microscopiq_rows)
    write_csv(PPA_DIR / "systolic_array_synth_m2xfp.csv", m2xfp_rows)
    write_csv(PPA_DIR / "systolic_array_synth_nvesm2.csv", nvesm2_rows)


if __name__ == "__main__":
    main()
