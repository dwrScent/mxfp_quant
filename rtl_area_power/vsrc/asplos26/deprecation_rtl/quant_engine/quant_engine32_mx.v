// =============================================================
// Quantization Engine (32 lanes, FP32 -> FP4_base + FP6 refine)
// - shared_scale = (max_exponent_unbiased) - 2
// - Outputs (packed buses, lane i at [4*i+3:4*i] or [2*i+1:2*i])
//   * fp4_base_bus   : 32 x FP4(E2M1)
//   * fp4_refine_bus : 32 x FP4(E2M1) = upper 4 bits of FP6(E2M3)
//   * metadata_bus   : 32 x 2b        = lower 2 bits of FP6 mantissa
// - All combinational; pipeline可后续插入
// =============================================================
module quant_engine32_mx (
    input  [32*32-1:0] in_fp32_bus,     // lane i: in_fp32_bus[32*i+31 : 32*i]
    output signed [7:0] shared_scale,   // k = e_max_unb - 2
    output [32*4-1:0] fp4_base_bus,     // lane i: [4*i+3:4*i]
    output [32*4-1:0] fp4_refine_bus,   // lane i: [4*i+3:4*i]
    output [32*2-1:0] metadata_bus      // lane i: [2*i+1:2*i]
);

    // ----------- 取 32 路 exponent(8b, biased) 并找最大 -----------
    wire [7:0] exp_biased [0:31];
    genvar gi;
    generate
      for (gi=0; gi<32; gi=gi+1) begin: EXTRACT
        assign exp_biased[gi] = in_fp32_bus[gi*32+30 -: 8]; // [30:23]
      end
    endgenerate

    wire [7:0] max_exp_biased;
    max_exp U_MAXEXP (.exp_bv(exp_biased), .max_exp_biased(max_exp_biased));

    // k = (max_unbiased) - 2 = (max_biased - 127) - 2
    assign shared_scale = $signed({1'b0,max_exp_biased}) - 127 - 2;

    // ----------- 32 路量化：FP4(E2M1) + FP6(E2M3) -----------
    wire [5:0] fp6_lane [0:31];
    wire [3:0] fp4b_lane[0:31];

    generate
      for (gi=0; gi<32; gi=gi+1) begin: QLANE
        quantize_fp32_to_e2m #(.M(1)) U_Q4B (
            .x_fp32 ( in_fp32_bus[gi*32+31 -: 32] ),
            .k      ( shared_scale ),
            .y_code ( fp4b_lane[gi] )   // 4b
        );
        quantize_fp32_to_e2m #(.M(3)) U_Q6 (
            .x_fp32 ( in_fp32_bus[gi*32+31 -: 32] ),
            .k      ( shared_scale ),
            .y_code ( fp6_lane[gi] )    // 6b
        );
      end
    endgenerate

    // ----------- 拆分 FP6 -> (refine4, meta2) 并打包总线 -----------
    generate
      for (gi=0; gi<32; gi=gi+1) begin: PACK
        // high4 = {sign, e2[1:0], m3[2]}  ; low2 = m3[1:0]
        assign fp4_refine_bus[gi*4+3 -: 4] = { fp6_lane[gi][5:3], fp6_lane[gi][2] };
        assign metadata_bus  [gi*2+1 -: 2] =   fp6_lane[gi][1:0];
        assign fp4_base_bus  [gi*4+3 -: 4] =   fp4b_lane[gi];
      end
    endgenerate

endmodule
