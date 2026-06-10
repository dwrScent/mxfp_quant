#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULT_CSV = ROOT / "results" / "m2xfp_res.csv"
OUT_DIR = ROOT / "results"

ENERGY_COMPONENTS = ["Static", "Dram", "Buffer", "Core"]
ENERGY_COLORS = {
    "Static": "#4E79A7",
    "Dram": "#F28E2B",
    "Buffer": "#59A14F",
    "Core": "#E15759",
}
LATENCY_COLOR = "#4E79A7"


def clean(value):
    return value.strip()


def read_rows(path):
    with path.open(newline="") as handle:
        return [[clean(cell) for cell in row] for row in csv.reader(handle)]


def padded(row, length):
    return row + [""] * (length - len(row))


def extract_group(model_header, accel_header, value_row, group_name="Mean"):
    max_len = max(len(model_header), len(accel_header), len(value_row))
    model_header = padded(model_header, max_len)
    accel_header = padded(accel_header, max_len)
    value_row = padded(value_row, max_len)

    group_starts = [
        idx for idx, label in enumerate(model_header) if label == group_name
    ]
    if group_starts:
        start = group_starts[-1]
        end = max_len
    else:
        start = 1
        end = next(
            (
                idx
                for idx in range(start + 1, max_len)
                if model_header[idx]
            ),
            max_len,
        )

    values = {}
    for idx in range(start, end):
        accelerator = accel_header[idx]
        raw_value = value_row[idx]
        if not accelerator or not raw_value:
            continue
        values[accelerator] = float(raw_value)
    return values


def parse_result_blocks(rows):
    blocks = []
    idx = 0
    while idx < len(rows):
        row = rows[idx]
        if not row or row[0] != "Time":
            idx += 1
            continue

        if idx < 2 or idx + 6 >= len(rows):
            idx += 1
            continue

        time = extract_group(rows[idx - 2], rows[idx - 1], rows[idx])
        energy_header = rows[idx + 1]
        energy_accel_header = rows[idx + 2]
        energy = {}

        next_idx = idx + 3
        while next_idx < len(rows) and rows[next_idx]:
            component = rows[next_idx][0]
            if component not in ENERGY_COMPONENTS:
                break
            energy[component] = extract_group(
                energy_header, energy_accel_header, rows[next_idx]
            )
            next_idx += 1

        if time and all(component in energy for component in ENERGY_COMPONENTS):
            accelerators = list(time.keys())
            if all(
                all(accelerator in energy[component] for component in ENERGY_COMPONENTS)
                for accelerator in accelerators
            ):
                blocks.append(
                    {
                        "accelerators": accelerators,
                        "time": time,
                        "energy": energy,
                    }
                )
        idx = max(next_idx, idx + 1)
    return blocks


def nice_axis_limit(value):
    if value <= 0:
        return 1.0, 0.2
    rough_limit = value * 1.15
    rough_step = rough_limit / 5
    exponent = math.floor(math.log10(rough_step))
    base = 10 ** exponent
    fraction = rough_step / base
    if fraction <= 1:
        step = base
    elif fraction <= 2:
        step = 2 * base
    elif fraction <= 2.5:
        step = 2.5 * base
    elif fraction <= 5:
        step = 5 * base
    else:
        step = 10 * base
    limit = math.ceil(rough_limit / step) * step
    return limit, step


def format_value(value):
    if abs(value - 1.0) < 0.0005:
        return ""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def svg_text(x, y, text, size=14, anchor="middle", weight="400", extra=""):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" {extra}>{escape(text)}</text>'
    )


def render_svg(path, y_label, accelerators, series, colors, stacked):
    width = 760
    height = 460
    margin_left = 82
    margin_right = 28
    margin_top = 58
    margin_bottom = 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    totals = []
    for accelerator in accelerators:
        if stacked:
            totals.append(sum(series[name][accelerator] for name in series))
        else:
            totals.append(series[accelerator])

    y_limit, y_step = nice_axis_limit(max(totals))
    bar_gap = 36
    slot = plot_width / len(accelerators)
    bar_width = min(70, max(42, slot - bar_gap))

    def x_center(accel_idx):
        return margin_left + slot * accel_idx + slot / 2

    def y_pos(value):
        return margin_top + plot_height - (value / y_limit) * plot_height

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{fill:#222}.axis{stroke:#222;stroke-width:1.2}'
        '.grid{stroke:#d9d9d9;stroke-width:1}.tick{stroke:#222;stroke-width:1}'
        "</style>",
    ]

    tick = 0.0
    while tick <= y_limit + y_step * 0.5:
        y = y_pos(tick)
        elements.append(
            f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" '
            f'x2="{width - margin_right}" y2="{y:.2f}"/>'
        )
        elements.append(
            f'<line class="tick" x1="{margin_left - 5}" y1="{y:.2f}" '
            f'x2="{margin_left}" y2="{y:.2f}"/>'
        )
        elements.append(
            svg_text(margin_left - 10, y + 4, f"{tick:.2f}".rstrip("0").rstrip("."), 12, "end")
        )
        tick += y_step

    elements.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{width - margin_right}" y2="{margin_top + plot_height}"/>'
    )
    elements.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + plot_height}"/>'
    )
    elements.append(svg_text(margin_left, 46, y_label, 16, "start", "600"))

    for idx, accelerator in enumerate(accelerators):
        center = x_center(idx)
        left = center - bar_width / 2
        baseline = y_pos(0)
        total = totals[idx]

        if stacked:
            bottom = 0.0
            for name, values in series.items():
                value = values[accelerator]
                top = bottom + value
                y = y_pos(top)
                segment_height = y_pos(bottom) - y
                elements.append(
                    f'<rect x="{left:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{segment_height:.2f}" fill="{colors[name]}"/>'
                )
                bottom = top
        else:
            y = y_pos(total)
            elements.append(
                f'<rect x="{left:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{baseline - y:.2f}" fill="{colors}"/>'
            )

        label = format_value(total)
        if label:
            elements.append(svg_text(center, y_pos(total) - 9, label, 13, "middle", "600"))
        elements.append(svg_text(center, baseline + 26, accelerator, 14, "middle", "600"))

    if stacked:
        legend_x = margin_left + 4
        legend_y = 24
        offset = 0
        for name in series:
            elements.append(
                f'<rect x="{legend_x + offset}" y="{legend_y - 10}" '
                f'width="12" height="12" fill="{colors[name]}"/>'
            )
            elements.append(svg_text(legend_x + offset + 18, legend_y, name, 13, "start"))
            offset += 88

    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    rows = read_rows(RESULT_CSV)
    blocks = parse_result_blocks(rows)
    if not blocks:
        raise SystemExit(f"No complete result block found in {RESULT_CSV}")

    latest = blocks[-1]
    accelerators = latest["accelerators"]
    energy = latest["energy"]
    time = latest["time"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_svg(
        OUT_DIR / "norm_energy.svg",
        "Norm. Energy",
        accelerators,
        energy,
        ENERGY_COLORS,
        stacked=True,
    )
    render_svg(
        OUT_DIR / "norm_latency.svg",
        "Norm. Latency",
        accelerators,
        time,
        LATENCY_COLOR,
        stacked=False,
    )

    print(f"Wrote {OUT_DIR / 'norm_energy.svg'}")
    print(f"Wrote {OUT_DIR / 'norm_latency.svg'}")


if __name__ == "__main__":
    main()
