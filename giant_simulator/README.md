
This repository contains the code for the ANT simulator based on DNNWeaver and BitFusion.

## Prerequisite


## Getting Started

power of baseline PEs, use the data from 45nm FreePDK synthesis.
+ MicroscopiQ: `rtl_area_power/vsrc/baselines/microscopiq/pe_microscopiq_o32.v`
+ M2XFP: `rtl_area_power/vsrc/asplos26/microscopiq/pe_tile_mxfp_fp32.v`

microscopiq is the result of directly synthesizing an 8x8 PE at 45nm; 
M2XFP is the result of synthesizing PE tiles (eight 4x4 units), a 4x4 PE using PE tile power/8

ANT, OliVe, MANT use the previous data.

How to run:
```shell
python run_ant_refactor.py
```

## Evaluation

