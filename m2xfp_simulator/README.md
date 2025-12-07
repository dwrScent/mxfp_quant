# M2XFP Simulator

This repository contains the simulation framework for **M2XFP**. The simulator models performance, energy, and area, building upon concepts from BitFusion and ANT. 
It supports flexible configuration for various baselines (ANT, MANT, OliVe, MicroscopiQ, etc.) and allows for detailed architecture exploration.

## 📂 Project Structure

```text
├── accelerator/            # Core simulation logic and source code
│   ├── sram/cacti/         # CACTI for memory modeling
│   └── src/                # Simulator core (graph, simulator, tensor ops, etc.)
├── benchmarks/             # LLM model shapes + accel-specific bit-width configs
│   ├── base_models.py      # Model architecture (GEMM shapes per block)
│   └── accel_model_configs.py  # Per-accelerator bit-width assignments
├── configs/
│   ├── accelerator/        # Hardware configs: systolic dims, buffers, if_width, pmax/pmin, etc.
│   └── ppa/                # Power/Performance/Area (PPA) CSVs for cores
├── results/                # Simulation outputs (.csv)
├── scripts/                # Helper scripts
├── run_simulator.py        # Main entry point
└── requirements.txt        # Python dependencies
```

## Installation

We recommend using Conda to manage the environment.

```shell
$ # Environment.
$ conda create -n m2xfp_sim python=3.10.14
$ conda activate m2xfp_sim  
$ pip install -r  requirements.txt

$ # Cacti for the memory simulation.
$ git clone https://github.com/HewlettPackard/cacti ./accelerator/sram/cacti/
$ make -C ./accelerator/sram/cacti/
```

## Methodology & Configuration

### 1. ISO-Accuracy Alignment
Different accelerators use varying quantization strategies. To ensure a fair comparison, we align all baselines to a target accuracy. Consequently, the bit-widths for each model layer differ across accelerators.
* **Configuration:** Layer-wise bit-widths are defined in `benchmarks/accel_model_configs.py`.

### 2. ISO-Area Hardware Configuration
The configuration files in `configs/accelerator/` (`conf_*.ini`) define the hardware parameters (Buffer size, PE count, Bandwidth).
* **Design Principle:** We align configurations based on **ISO-Area** constraints. Lower precision units allow for higher parallelism within the same area budget.
* **Example:** An 8-bit baseline (e.g., ANT) is configured as a **16x16** systolic array, while a 4-bit baseline (e.g., M2XFP) scales to a **32x32** array.

### 3. Core PPA Data (Energy / Area for PEs)

Core (PE array) power/area comes from:
`configs/ppa/systolic_array_synth.csv`
+ Used by most accelerators (ant, olive, microscopiq, m2xfp).
configs/ppa/systolic_array_synth_mant.csv
+ Used only for mant, whose PE tile implementation differs.

To ensure accurate power estimation, we use synthesis data based on 45nm FreePDK:
+ MicroscopiQ: `rtl_area_power/vsrc/baselines/microscopiq/pe_microscopiq_o32.v`
+ M2XFP: `rtl_area_power/vsrc/asplos26/microscopiq/pe_tile_mxfp_fp32.v`

The power of PE
+ Baselines (ANT, OliVe, MANT, MicroscopiQ): Derived from their synthesized 8-bit x 8-bit PE.
+ M2XFP: Derived by synthesizing a PE tile composed of eight 4x4 units and normalizing the results to a single PE.

## Running the Simulator

How to run:
```shell
python run_simulator.py \
  --models llama3_8b \
  --accelerators olive,ant,mant,microscopiq,m2xfp \
  --normalized-bench olive \
  --batch-size 1
```

Aggregated, normalized summary in `results/m2xfp_res.csv`.

This file includes the normalized data of accelerators across several LLMs.

