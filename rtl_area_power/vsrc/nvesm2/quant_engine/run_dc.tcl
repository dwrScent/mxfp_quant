# Synthesize the NVESM2 quant engine top with its helper modules.
read_file -format verilog {
    ../baseunit/nvesm2_fp32_mul.v
    nvesm2_fp32_abs_diff_pos.v
    nvesm2_group_scale.v
    nvesm2_quant_lane.v
    nvesm2_subgroup_accum.v
    quant_engine32.v
}

# top module
current_design quant_engine32

# Technology library used by the original area/power reports.
set_app_var target_library "/home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
set_app_var link_library "* /home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
compile

# Emit reports in the working directory used by Design Compiler.
report_area > area_report.txt
report_power > power_report.txt
