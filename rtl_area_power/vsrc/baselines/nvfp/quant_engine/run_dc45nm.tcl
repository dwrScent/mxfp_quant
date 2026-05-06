# Synthesize the baseline NVFP quant engine top with Nangate 45 nm cells.
set target_lib "/home/design/Desktop/pdk45/NangateOpenCellLibrary_typical.db"
set top_design quant_engine32

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

set rtl_files [list \
    nvfp_fp32_mul.v \
    nvfp_group_scale.v \
    nvfp_quant_lane.v \
    quant_engine32.v \
]

foreach rtl_file $rtl_files {
    if {![file exists $rtl_file]} {
        puts "Error: missing RTL file: $rtl_file"
        exit 1
    }
    read_file -format verilog $rtl_file
}

current_design $top_design
link
check_design

if {![info exists clock_period]} { set clock_period 5.0 }
if {![info exists input_delay]}  { set input_delay  [expr {$clock_period * 0.20}] }
if {![info exists output_delay]} { set output_delay [expr {$clock_period * 0.20}] }
if {![info exists input_toggle_rate]} { set input_toggle_rate 0.02 }

create_clock -name clk -period $clock_period [get_ports clk]
set_clock_uncertainty [expr {$clock_period * 0.05}] [get_clocks clk]
set_input_delay  $input_delay  -clock clk [remove_from_collection [all_inputs]  [get_ports {clk rst_n}]]
set_output_delay $output_delay -clock clk [all_outputs]
set_switching_activity -toggle_rate $input_toggle_rate [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]

set_max_area 0
compile -map_effort medium -area_effort high

report_area > area_report.txt
report_area -hierarchy > area_hierarchy_report.txt
report_power > power_report.txt
report_power -hierarchy > power_hierarchy_report.txt

puts "Synthesis for $top_design completed."
