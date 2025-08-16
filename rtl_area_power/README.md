# Rtl code of accelerator


# SRAM 

sram_stats 目录

使用 CACTI 得到 sram 的 buffer 和 power 数据。

```shell
https://github.com/HewlettPackard/cacti
make # 得到可执行程序 cacti

# sram 模板可以参考 sample_config_files/wideio_cache.cfg

./cacti -infile sram_512kb_28nm.cfg
./cacti -infile sram_512kb_16nm.cfg

# 跑完会得到输出文件
```

