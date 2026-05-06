read_file -format verilog {nvfp_fp32_mul.v nvfp_group_scale.v nvfp_quant_lane.v quant_engine32.v}

current_design quant_engine32
link
check_design
compile

report_area > area_report.txt
report_power > power_report.txt
