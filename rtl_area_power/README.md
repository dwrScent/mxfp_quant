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

## M2XFP units

+ Top-1 Decode Unit: `vsrc/asplos26/decode_unit_v`
+ Quantization Engine: `vsrc/asplos26/quant_engine`
+ PE Tile: `vsrc/asplos26/pe_tile_v`

## Baseline accelerator untis

+ ANT: `vsrc/baselines/ant_olive`
+ MANT: `vsrc/baselines/mant`
+ MicroScopiQ: `vsrc/baselines/microscopiq`