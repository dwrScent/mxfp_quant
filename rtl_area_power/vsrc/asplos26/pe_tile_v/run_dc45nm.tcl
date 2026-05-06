# Synthesize the ASPlOS26 MXFP PE tile with Nangate 45 nm cells.
set target_lib "/home/design/Desktop/pdk45/NangateOpenCellLibrary_typical.db"
set top_design pe_tile_mxfp_fp32
set report_prefix pe_tile_mxfp_fp32_45nm

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

set rtl_files [list \
    fp32_add.v \
    fxp_to_fp32.v \
    ../../nvesm2/pe_tile_v/mul_base_q2_comb.v \
    pe_tile_mxfp_fp32.v \
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
