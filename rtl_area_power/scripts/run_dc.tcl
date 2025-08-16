# 从命令行参数获取模块名称
if {[llength $argv] > 0} {
    set module_name [lindex $argv 0]
} else {
    puts "Error: No module name provided. Usage: dc_shell -f synth.tcl <module_name>"
    exit 1
}

# 打印模块名称
puts "Synthesis for module: $module_name"

# 读取 Verilog 文件
read_file -format verilog "${module_name}.v"

# 设置目标库和链接库
set_app_var target_library "tcbn28hpcplusbwp7t40p140tt0p8v25c.db"
set_app_var link_library "* tcbn28hpcplusbwp7t40p140tt0p8v25c.db"

# 开始综合
compile

# 生成面积和功耗报告
report_area > "${module_name}_area_report.txt"
report_power > "${module_name}_power_report.txt"

puts "Synthesis for $module_name completed."