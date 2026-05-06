# module_name is supplied by the caller so this script can synthesize either
# quant_engine32_mx with its dependencies or one standalone helper module.
if {![info exists module_name]} {
    puts "Error: module_name variable is not defined. Set it before sourcing the script."
    exit
}

puts "Synthesis for module: $module_name"

# The quant engine top depends on the local quant modules and the shared
# FP32 multiplier in ../baseunit.  Other module_name values are expected to
# be standalone Verilog files named ${module_name}.v, except baseunit modules
# listed below.
if {$module_name == "quant_engine32_mx"} {
    set rtl_files [list \
        ../baseunit/nvesm2_fp32_mul.v \
        nvesm2_fp32_abs_diff_pos.v \
        nvesm2_group_scale.v \
        nvesm2_quant_lane.v \
        nvesm2_subgroup_accum.v \
        quant_engine32_mx.v \
    ]
    foreach rtl_file $rtl_files {
        if {![file exists $rtl_file]} {
            puts "Error: missing RTL file: $rtl_file"
            exit 1
        }
        read_file -format verilog $rtl_file
    }
} elseif {$module_name == "nvesm2_fp32_mul"} {
    read_file -format verilog "../baseunit/${module_name}.v"
} elseif {$module_name == "nvesm2_quant_lane"} {
    set rtl_files [list \
        ../baseunit/nvesm2_fp32_mul.v \
        nvesm2_fp32_abs_diff_pos.v \
        nvesm2_quant_lane.v \
    ]
    foreach rtl_file $rtl_files {
        if {![file exists $rtl_file]} {
            puts "Error: missing RTL file: $rtl_file"
            exit 1
        }
        read_file -format verilog $rtl_file
    }
} else {
    read_file -format verilog "${module_name}.v"
}

current_design $module_name

# 16 nm target library used for area/power collection.
set_app_var target_library "/home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"
set_app_var link_library "* /home/design/Desktop/tcbn16ffcllbwp16p90tt1v85c.db"

compile

# Name reports after module_name so multiple syntheses can share a directory.
report_area > "${module_name}_area_report.txt"
report_power > "${module_name}_power_report.txt"

puts "Synthesis for $module_name completed."
