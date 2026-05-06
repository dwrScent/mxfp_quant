# Synthesize the ASPlOS26 MXFP quant engine with Nangate 45 nm cells.
set target_lib "/home/design/Desktop/pdk45/NangateOpenCellLibrary_typical.db"
set top_design quant_engine32_mx
set report_prefix quant_engine32_mx_45nm

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

set rtl_files [list \
    max_exp.v \
    quantize_fp32_to_e2m.v \
    quant_engine32_mx.v \
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
compile

report_area > "${report_prefix}_area_report.txt"
report_power > "${report_prefix}_power_report.txt"

puts "Synthesis for $top_design completed."
