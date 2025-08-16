// =============================================================
// Quantization Engine (32 lanes, FP32 -> FP4_base + FP6 refine)
// Verilog-2001 Compliant Version
// - All combinational; pipeline can be added externally.
// =============================================================
module quant_engine32_mx (
    input  [32*32-1:0] in_fp32_bus,     // lane i: in_fp32_bus[32*i+31 : 32*i]
    output signed [7:0] shared_scale,   // k = e_max_unb - 2
    output [32*4-1:0] fp4_base_bus,     // lane i: [4*i+3:4*i]
    output [32*4-1:0] fp4_refine_bus,   // lane i: [4*i+3:4*i]
    output [32*2-1:0] metadata_bus      // lane i: [2*i+1:2*i]
);

    genvar gi;

    // ----------- [FIX-1] Pack 32 lane exponents into a single bus for max_exp -----------
    // Create a single 256-bit bus to hold all 32 exponent values,
    // which matches the input port of the modified max_exp module.
    wire [32*8-1:0] exp_bus_for_max;

    generate
      for (gi=0; gi<32; gi=gi+1) begin: PACK_EXPONENTS
        // Extract the 8-bit exponent from the i-th 32-bit float and
        // place it into the corresponding 8-bit slot in the packed bus.
        // [SV_FIX] Replaced `[gi*32+30 -: 8]` with standard Verilog-2001 syntax.
        assign exp_bus_for_max[gi*8 + 7 : gi*8] = in_fp32_bus[gi*32 + 30 : gi*32 + 23];
      end
    endgenerate

    wire [7:0] max_exp_biased;
    // [FIX-2] Instantiate max_exp with the new packed bus and correct port name.
    max_exp U_MAXEXP (
        .exp_bus(exp_bus_for_max), // Connect the newly created packed bus
        .max_exp_biased(max_exp_biased)
    );

    // k = (max_unbiased) - 2 = (max_biased - 127) - 2
    assign shared_scale = $signed({1'b0,max_exp_biased}) - 129; // Combined -127 and -2

    // ----------- 32-lane Quantization: FP4(E2M1) + FP6(E2M3) -----------
    // Unpacked arrays for holding results from each lane before final packing.
    wire [5:0] fp6_lane [0:31];
    wire [3:0] fp4b_lane[0:31];

    generate
      for (gi=0; gi<32; gi=gi+1) begin: QLANE
        // [SV_FIX] Replaced `[gi*32+31 -: 32]` with standard Verilog-2001 syntax.
        wire [31:0] current_fp32_lane = in_fp32_bus[gi*32 + 31 : gi*32];

        quantize_fp32_to_e2m #(.M(1)) U_Q4B (
            .x_fp32 ( current_fp32_lane ),
            .k      ( shared_scale ),
            .y_code ( fp4b_lane[gi] )
        );
        quantize_fp32_to_e2m #(.M(3)) U_Q6 (
            .x_fp32 ( current_fp32_lane ),
            .k      ( shared_scale ),
            .y_code ( fp6_lane[gi] )
        );
      end
    endgenerate

    // ----------- Split FP6 -> (refine4, meta2) and Pack Output Buses -----------
    generate
      for (gi=0; gi<32; gi=gi+1) begin: PACK_OUTPUTS
        // [SV_FIX] Replaced all `-:` part-selects with standard Verilog-2001 syntax.
        // high4 = {sign, e2[1:0], m3[2]}  ; low2 = m3[1:0]
        assign fp4_refine_bus[gi*4 + 3 : gi*4] = { fp6_lane[gi][5:3], fp6_lane[gi][2] };
        assign metadata_bus  [gi*2 + 1 : gi*2] =   fp6_lane[gi][1:0];
        assign fp4_base_bus  [gi*4 + 3 : gi*4] =   fp4b_lane[gi];
      end
    endgenerate

endmodule
