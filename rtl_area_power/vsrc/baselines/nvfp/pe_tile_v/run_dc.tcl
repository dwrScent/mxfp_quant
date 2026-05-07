# Synthesize the baseline NVFP PE tile with the 16 nm library.
set target_lib "/home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
set top_design pe_tile_nvfp_fp32

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

set rtl_files [list \
    fp32_add.v \
    fxp_to_fp32.v \
    mul_base_q2_comb.v \
    pe_tile_nvfp_fp32.v \
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

report_area > area_report.txt
report_power > power_report.txt

puts "Synthesis for $top_design completed."
