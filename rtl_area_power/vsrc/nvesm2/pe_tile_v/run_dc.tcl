read_file -format verilog {fp32_add.v fxp_to_fp32.v mul_base_q2_comb.v pe_tile_mxfp_fp32.v}
# read_file -format verilog {fp32_add.v}

# top module
# current_design fp32_add
current_design pe_tile_nvfp_fp32

set_app_var target_library "/home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
set_app_var link_library "* /home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
compile

report_area > area_report.txt
report_power > power_report.txt
