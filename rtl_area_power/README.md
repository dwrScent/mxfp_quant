# Rtl code of accelerator


# SRAM 

sram_stats 目录

使用 CACTI 得到 sram 的 buffer 和 power 数据。

```shell
https://github.com/HewlettPackard/cacti
cd cacti
make # 得到可执行程序 cacti

# sram 模板可以参考 sample_config_files/wideio_cache.cfg

# output buffer 配置
./cacti -infile ../sram_28nm_OBUF.cfg

# weight/input buffer 配置
./cacti -infile ../sram_28nm_WBUF_IBUF.cfg

# 跑完会得到输出文件
```

