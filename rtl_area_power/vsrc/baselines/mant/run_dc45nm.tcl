# Synthesize the MANT PE used by m2xfp_simulator PPA CSVs.

set top_design pe88
set report_prefix pe88_withFusion_withShift
set report_dir "../../../result/baselines/mant"
set target_lib "/home/design/Desktop/pdk45/NangateOpenCellLibrary_typical.db"

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

read_file -format verilog pe88_withFusion_withShift.v

current_design $top_design
link
check_design
compile

file mkdir $report_dir
report_area > "${report_dir}/${report_prefix}_area_report.txt"
report_power > "${report_dir}/${report_prefix}_power_report.txt"

puts "Reports written to $report_dir."
puts "Synthesis for $top_design completed."
