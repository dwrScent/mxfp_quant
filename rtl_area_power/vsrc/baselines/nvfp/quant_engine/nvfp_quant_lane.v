// =============================================================
// NVFP per-lane quantizer.
//
// Flow:
//   1. Apply the 16-lane shared E4M3 group scale.
//   2. Quantize abs(x / group_scale) to FP4 E2M1.
//   3. Reattach the original sign and force zero positive.
// =============================================================
module nvfp_quant_lane (
    input             clk,
    input             rst_n,
    input             in_valid,
    input      [31:0] x_fp32,
    input      [7:0]  group_scale,
    output reg        out_valid,
    output reg [3:0]  fp4
);

    function [31:0] pack_pos_fp32;
        input signed [10:0] exp_unb;
        input [22:0] frac;
        reg signed [11:0] exp_biased;
        begin
            exp_biased = exp_unb + 12'sd127;
            if (exp_biased >= 12'sd255)
                pack_pos_fp32 = 32'h7f800000;
            else if (exp_biased <= 12'sd0)
                pack_pos_fp32 = 32'h00000000;
            else
                pack_pos_fp32 = {1'b0, exp_biased[7:0], frac};
        end
    endfunction

    function [31:0] e4m3_inv_to_fp32;
        input [7:0] scale;
        reg [3:0] exp4;
        reg [2:0] mant3;
        begin
            exp4 = scale[6:3];
            mant3 = scale[2:0];
            if (scale[6:0] == 7'd0) begin
                e4m3_inv_to_fp32 = 32'h7f800000;
            end else if (exp4 == 4'd0) begin
                case (mant3)
                    3'd1: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd9,  23'h000000); // 512
                    3'd2: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd8,  23'h000000); // 256
                    3'd3: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd7,  23'h2aaaab); // 512/3
                    3'd4: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd7,  23'h000000); // 128
                    3'd5: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6,  23'h4ccccd); // 512/5
                    3'd6: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6,  23'h2aaaab); // 512/6
                    default: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6, 23'h124925); // 512/7
                endcase
            end else begin
                case (mant3)
                    3'd0: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd7 - $signed({1'b0, exp4}), 23'h000000);
                    3'd1: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6 - $signed({1'b0, exp4}), 23'h638e39);
                    3'd2: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6 - $signed({1'b0, exp4}), 23'h4ccccd);
                    3'd3: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6 - $signed({1'b0, exp4}), 23'h3a2e8c);
                    3'd4: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6 - $signed({1'b0, exp4}), 23'h2aaaab);
                    3'd5: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6 - $signed({1'b0, exp4}), 23'h1d89d9);
                    3'd6: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6 - $signed({1'b0, exp4}), 23'h124925);
                    default: e4m3_inv_to_fp32 = pack_pos_fp32(11'sd6 - $signed({1'b0, exp4}), 23'h088889);
                endcase
            end
        end
    endfunction

    function fp32_lt_pos;
        input [31:0] a;
        input [31:0] b;
        begin
            fp32_lt_pos = (a[30:0] < b[30:0]);
        end
    endfunction

    function [2:0] quant_mag_fp4_e2m1_fp32;
        input [31:0] mag_fp32;
        begin
            if      (fp32_lt_pos(mag_fp32, 32'h3ec00000)) quant_mag_fp4_e2m1_fp32 = 3'b000; // 0
            else if (fp32_lt_pos(mag_fp32, 32'h3f600000)) quant_mag_fp4_e2m1_fp32 = 3'b001; // 0.75
            else if (fp32_lt_pos(mag_fp32, 32'h3fa00000)) quant_mag_fp4_e2m1_fp32 = 3'b010; // 1.0
            else if (fp32_lt_pos(mag_fp32, 32'h3fe00000)) quant_mag_fp4_e2m1_fp32 = 3'b011; // 1.5
            else if (fp32_lt_pos(mag_fp32, 32'h40200000)) quant_mag_fp4_e2m1_fp32 = 3'b100; // 2.0
            else if (fp32_lt_pos(mag_fp32, 32'h40600000)) quant_mag_fp4_e2m1_fp32 = 3'b101; // 3.0
            else if (fp32_lt_pos(mag_fp32, 32'h40a00000)) quant_mag_fp4_e2m1_fp32 = 3'b110; // 4.0
            else                                          quant_mag_fp4_e2m1_fp32 = 3'b111; // 6.0
        end
    endfunction

    wire [31:0] abs_x_fp32 = {1'b0, x_fp32[30:0]};
    wire [31:0] group_scale_inv_fp32 = e4m3_inv_to_fp32(group_scale);
    wire [31:0] norm_div_mul_fp32;
    wire        group_scale_zero = (group_scale[6:0] == 7'd0);
    wire        x_zero = (x_fp32[30:0] == 31'd0);
    wire [31:0] norm_fp32_comb = group_scale_zero ?
                                  (x_zero ? 32'h00000000 : 32'h7f800000) :
                                  norm_div_mul_fp32;

    nvfp_fp32_mul U_DIV_GROUP_SCALE (
        .a(abs_x_fp32),
        .b(group_scale_inv_fp32),
        .z(norm_div_mul_fp32)
    );

    reg        v1;
    reg        sign_s1;
    reg [31:0] norm_fp32_s1;
    wire [2:0] fp4_mag_s2_comb = quant_mag_fp4_e2m1_fp32(norm_fp32_s1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            sign_s1 <= 1'b0;
            norm_fp32_s1 <= 32'h00000000;
        end else begin
            v1 <= in_valid;
            sign_s1 <= x_fp32[31];
            norm_fp32_s1 <= norm_fp32_comb;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            fp4 <= 4'd0;
        end else begin
            out_valid <= v1;
            fp4 <= (fp4_mag_s2_comb == 3'b000) ? 4'b0000 : {sign_s1, fp4_mag_s2_comb};
        end
    end
endmodule
