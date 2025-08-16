read_file -format verilog {max.v rounding.v quant_engine32_mx.v}

# top module
current_design quant_engine32_mx

set_app_var target_library "/home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
set_app_var link_library "* /home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
compile

report_area > area_report.txt
report_power > power_report.txt