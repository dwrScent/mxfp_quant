read_file -format verilog {encoder.v decoder.v}

# top module
current_design encode_unit
# current_design top1_detection_unit

set_app_var target_library "/home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
set_app_var link_library "* /home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
compile

report_area > area_report.txt
report_power > power_report.txt