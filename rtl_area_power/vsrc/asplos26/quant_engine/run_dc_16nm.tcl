# 检查是否定义了 module_name 变量
if {![info exists module_name]} {
    puts "Error: module_name variable is not defined. Set it before sourcing the script."
    exit
}

# 使用 module_name 变量
puts "Synthesis for module: $module_name"

read_file -format verilog "${module_name}.v"

set_app_var target_library "/home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
set_app_var link_library "* /home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"

compile

report_area > "${module_name}_area_report.txt"
report_power > "${module_name}_power_report.txt"

puts "Synthesis for $module_name completed."
