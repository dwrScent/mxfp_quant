# Synthesize the NVESM2 quant engine top with Nangate 45 nm cells.
set target_lib "/home/design/Desktop/pdk45/NangateOpenCellLibrary_typical.db"

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

set rtl_files [list \
    ../baseunit/nvesm2_fp32_mul.v \
    nvesm2_fp32_abs_diff_pos.v \
    nvesm2_group_scale.v \
    nvesm2_quant_lane.v \
    nvesm2_subgroup_accum.v \
    quant_engine32.v \
]

foreach rtl_file $rtl_files {
    if {![file exists $rtl_file]} {
        puts "Error: missing RTL file: $rtl_file"
        exit 1
    }
    read_file -format verilog $rtl_file
}

current_design quant_engine32
link
check_design
compile

report_area > area_report.txt
report_power > power_report.txt
