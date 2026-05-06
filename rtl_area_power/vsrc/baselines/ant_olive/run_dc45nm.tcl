# Synthesize the OLiVE baseline PE with Nangate 45 nm cells.
#
# Default:
#   dc_shell -f run_dc45nm.tcl
#
# Optional aliases:
#   olive_pe       -> pe88_withFusion_withoutShift.v / pe_mac_baseline
#   ant_pe         -> ant_pe.v / ant_pe
#   ant_fusion_pe  -> ant_pe_fusion.v / pe_mac_baseline
#   ant_fusion_o32 -> ant_pe_fusion_o32.v / pe_mac_baseline_32b
#   ant_tile       -> ant_pe_fusion_tile.v / tile_1x4_ant
if {![info exists module_name]} {
    if {[info exists argv] && [llength $argv] > 0} {
        set module_name [lindex $argv 0]
    } else {
        set module_name olive_pe
    }
}

set target_lib "/home/design/Desktop/pdk45/NangateOpenCellLibrary_typical.db"

if {![file exists $target_lib]} {
    puts "Error: missing target library: $target_lib"
    exit 1
}

switch -- $module_name {
    olive_pe {
        set rtl_files [list pe88_withFusion_withoutShift.v]
        set top_design pe_mac_baseline
        set report_prefix olive_pe_45nm
    }
    ant_pe {
        set rtl_files [list ant_pe.v]
        set top_design ant_pe
        set report_prefix ant_pe_45nm
    }
    ant_fusion_pe {
        set rtl_files [list ant_pe_fusion.v]
        set top_design pe_mac_baseline
        set report_prefix ant_fusion_pe_45nm
    }
    ant_fusion_o32 {
        set rtl_files [list ant_pe_fusion_o32.v]
        set top_design pe_mac_baseline_32b
        set report_prefix ant_fusion_o32_45nm
    }
    ant_tile {
        set rtl_files [list ant_pe_fusion_tile.v]
        set top_design tile_1x4_ant
        set report_prefix ant_tile_45nm
    }
    default {
        puts "Error: unknown module_name '$module_name'"
        puts "Valid aliases: olive_pe ant_pe ant_fusion_pe ant_fusion_o32 ant_tile"
        exit 1
    }
}

puts "Synthesis for alias: $module_name"
puts "Top design: $top_design"

set_app_var target_library [list $target_lib]
set_app_var link_library [list "*" $target_lib]

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

puts "Synthesis for $module_name completed."
