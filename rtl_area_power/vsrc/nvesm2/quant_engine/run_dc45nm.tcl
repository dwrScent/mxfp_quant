# Synthesize NVESM2 quant engine modules with Nangate 45 nm cells.
set target_lib "/home/design/Desktop/pdk45/NangateOpenCellLibrary_typical.db"
set report_suffix "_45nm"

set default_modules [list \
    nvesm2_fp32_mul \
    nvesm2_group_scale \
    nvesm2_quant_lane \
    nvesm2_subgroup_accum \
    quant_engine32 \
]

if {[info exists module_name]} {
    set synth_modules [list $module_name]
} elseif {[info exists argv] && [llength $argv] > 0} {
    set synth_modules $argv
} else {
    set synth_modules $default_modules
}

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

proc rtl_files_for_module {module_name} {
    switch -- $module_name {
        quant_engine32 {
            return [list \
                ../baseunit/nvesm2_fp32_mul.v \
                nvesm2_group_scale.v \
                nvesm2_quant_lane.v \
                nvesm2_subgroup_accum.v \
                quant_engine32.v \
            ]
        }
        nvesm2_quant_lane {
            return [list \
                ../baseunit/nvesm2_fp32_mul.v \
                nvesm2_quant_lane.v \
            ]
        }
        nvesm2_fp32_mul {
            return [list ../baseunit/nvesm2_fp32_mul.v]
        }
        default {
            return [list "${module_name}.v"]
        }
    }
}

proc read_rtl_files {rtl_files} {
    foreach rtl_file $rtl_files {
        if {![file exists $rtl_file]} {
            puts "Error: missing RTL file: $rtl_file"
            exit 1
        }
        read_file -format verilog $rtl_file
    }
}

proc synthesize_module {module_name report_suffix} {
    puts "Synthesis for module: $module_name"

    read_rtl_files [rtl_files_for_module $module_name]
    current_design $module_name
    link
    check_design
    compile

    report_area > "${module_name}${report_suffix}_area_report.txt"
    report_power > "${module_name}${report_suffix}_power_report.txt"

    puts "Synthesis for $module_name completed."
    remove_design -all
}

foreach module_name $synth_modules {
    synthesize_module $module_name $report_suffix
}
