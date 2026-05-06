read_file -format verilog {fp32_add.v fxp_to_fp32.v mul_base_q2_comb.v pe_tile_nvfp_fp32.v}

current_design pe_tile_nvfp_fp32
link
check_design
compile

report_area > area_report.txt
report_power > power_report.txt
