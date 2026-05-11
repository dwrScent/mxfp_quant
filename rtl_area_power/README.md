# Rtl code of accelerator


## SRAM 

Generate the area and power of SRAM from CACTI 7.0

```shell
cd sram_stats
git clone https://github.com/HewlettPackard/cacti
cd cacti
make # get the executable cacti

# sram template: sample_config_files/wideio_cache.cfg

# output buffer configuration
./cacti -infile ../sram_28nm_OBUF.cfg

# weight/input buffer configuration
./cacti -infile ../sram_28nm_WBUF_IBUF.cfg

# gather the area and power statistics from the output file in *.cfg.out
```

NVESM2 45 nm buffer configs are in `sram_stats/nvesm2_45nm_WBUF_IBUF.cfg`
and `sram_stats/nvesm2_45nm_OBUF.cfg`. They use CACTI's 45 nm technology
point for comparison with the FreePDK45/Nangate synthesis reports.

```shell
cd sram_stats
cd cacti

# NVESM2 weight/input buffer configuration
./cacti -infile ../nvesm2_45nm_WBUF_IBUF.cfg

# NVESM2 output buffer configuration
./cacti -infile ../nvesm2_45nm_OBUF.cfg
```

`nvesm2_45nm_WBUF_IBUF.cfg` models one quantized operand buffer, matching the
single-buffer capacity style of `sram_28nm_WBUF_IBUF.cfg`. Use the result for
either WBUF or IBUF; when accounting for both, use `2 * WBUF_IBUF + OBUF`.
`nvesm2_45nm_OBUF.cfg` models the pre-quantization FP32/data output buffer:
512 entries, 64 B per entry, 512-bit data path, and no quant metadata.

## M2XFP units

+ Top-1 Decode Unit: `vsrc/asplos26/decode_unit_v`
+ Quantization Engine: `vsrc/asplos26/quant_engine`
+ PE Tile: `vsrc/asplos26/pe_tile_v`

## Baseline accelerator untis

+ ANT: `vsrc/baselines/ant_olive`
+ MANT: `vsrc/baselines/mant`
+ MicroScopiQ: `vsrc/baselines/microscopiq`
