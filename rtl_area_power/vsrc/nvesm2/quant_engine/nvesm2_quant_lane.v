// =============================================================
// NVESM2 per-lane RQU.
// ES index maps to {1.00, 1.25, 1.50, 1.75}.
// FP4 format is E2M1: {sign, exp[1:0], mant[0]} with exponent bias 1.
// One instance evaluates one ES candidate at a time; the top level time-
// multiplexes four metadata candidates through the same per-lane hardware.
// =============================================================
module nvesm2_quant_lane (
    input             clk,
    input             rst_n,
    input             in_valid,
    input      [31:0] x_fp32,
    input      [7:0]  group_scale,
    input      [1:0]  es_idx,
    output reg        out_valid,
    output reg [1:0]  es_idx_out,
    output reg [3:0]  fp4,
    output reg [31:0] cost
);

    function [23:0] shr_round_u24;
        input [23:0] value;
        input integer sh;
        reg [24:0] addend;
        begin
            if (sh <= 0) begin
                shr_round_u24 = value;
            end else if (sh >= 24) begin
                shr_round_u24 = 24'd0;
            end else begin
                addend = {1'b0, value} + (25'd1 << (sh-1));
                shr_round_u24 = addend >> sh;
            end
        end
    endfunction

    function [12:0] e4m3_recip_q12;
        input [7:0] scale;
        reg [3:0] exp4;
        reg [2:0] mant3;
        begin
            exp4 = scale[6:3];
            mant3 = scale[2:0];
            if (scale[6:0] == 7'd0) begin
                e4m3_recip_q12 = 13'd8191;
            end else if (exp4 == 4'd0) begin
                case (mant3)
                    3'd1: e4m3_recip_q12 = 13'd4096; // 1 / 1
                    3'd2: e4m3_recip_q12 = 13'd2048; // 1 / 2
                    3'd3: e4m3_recip_q12 = 13'd1365; // 1 / 3
                    3'd4: e4m3_recip_q12 = 13'd1024; // 1 / 4
                    3'd5: e4m3_recip_q12 = 13'd819;  // 1 / 5
                    3'd6: e4m3_recip_q12 = 13'd683;  // 1 / 6
                    default: e4m3_recip_q12 = 13'd585; // 1 / 7
                endcase
            end else begin
                case (mant3)
                    3'd0: e4m3_recip_q12 = 13'd4096; // 8 / 8
                    3'd1: e4m3_recip_q12 = 13'd3641; // 8 / 9
                    3'd2: e4m3_recip_q12 = 13'd3277; // 8 / 10
                    3'd3: e4m3_recip_q12 = 13'd2979; // 8 / 11
                    3'd4: e4m3_recip_q12 = 13'd2731; // 8 / 12
                    3'd5: e4m3_recip_q12 = 13'd2521; // 8 / 13
                    3'd6: e4m3_recip_q12 = 13'd2341; // 8 / 14
                    default: e4m3_recip_q12 = 13'd2185; // 8 / 15
                endcase
            end
        end
    endfunction

    function signed [10:0] e4m3_scale_exp_unb;
        input [7:0] scale;
        reg [3:0] exp4;
        begin
            exp4 = scale[6:3];
            if (scale[6:0] == 7'd0)
                e4m3_scale_exp_unb = 11'sd7;
            else if (exp4 == 4'd0)
                e4m3_scale_exp_unb = -11'sd9;
            else
                e4m3_scale_exp_unb = $signed({1'b0, exp4}) - 11'sd7;
        end
    endfunction

    // abs(x_fp32 / group_scale) in Q8 fixed point. Division uses an E4M3 reciprocal LUT.
    function [23:0] fp32_abs_div_scale_to_q8;
        input [31:0] fp32;
        input [7:0] scale;
        reg [7:0] exp8;
        reg [22:0] frac23;
        reg [23:0] sig24;
        reg signed [10:0] e_unb;
        reg signed [10:0] shift;
        reg signed [10:0] scale_exp;
        reg [12:0] recip_q12;
        reg [36:0] prod_q12;
        reg [55:0] shifted;
        reg [55:0] rounded;
        integer rshift;
        begin
            exp8 = fp32[30:23];
            frac23 = fp32[22:0];
            scale_exp = e4m3_scale_exp_unb(scale);
            recip_q12 = e4m3_recip_q12(scale);
            fp32_abs_div_scale_to_q8 = 24'd0;

            if ((exp8 == 8'h00) && (frac23 == 23'd0)) begin
                fp32_abs_div_scale_to_q8 = 24'd0;
            end else if (scale[6:0] == 7'd0) begin
                fp32_abs_div_scale_to_q8 = 24'hffffff;
            end else if (exp8 == 8'h00) begin
                if (frac23 != 23'd0) begin
                    sig24 = {1'b0, frac23};
                    e_unb = -11'sd126;
                    shift = e_unb - scale_exp - 11'sd15;
                    prod_q12 = sig24 * recip_q12;
                    if (shift >= 11'sd24) begin
                        fp32_abs_div_scale_to_q8 = 24'hffffff;
                    end else if (shift >= 0) begin
                        shifted = ({19'd0, prod_q12} >> 12) << shift;
                        fp32_abs_div_scale_to_q8 = (|shifted[55:24]) ? 24'hffffff : shifted[23:0];
                    end else begin
                        rshift = 12 - shift;
                        if (rshift >= 56) begin
                            fp32_abs_div_scale_to_q8 = 24'd0;
                        end else begin
                            rounded = {19'd0, prod_q12} + (56'd1 << (rshift-1));
                            fp32_abs_div_scale_to_q8 = rounded >> rshift;
                        end
                    end
                end
            end else if (exp8 == 8'hff) begin
                fp32_abs_div_scale_to_q8 = 24'hffffff;
            end else begin
                sig24 = {1'b1, frac23};
                e_unb = $signed({1'b0, exp8}) - 11'sd127;
                shift = e_unb - scale_exp - 11'sd15;
                prod_q12 = sig24 * recip_q12;
                if (shift >= 11'sd24) begin
                    fp32_abs_div_scale_to_q8 = 24'hffffff;
                end else if (shift >= 0) begin
                    shifted = ({19'd0, prod_q12} >> 12) << shift;
                    fp32_abs_div_scale_to_q8 = (|shifted[55:24]) ? 24'hffffff : shifted[23:0];
                end else begin
                    rshift = 12 - shift;
                    if (rshift >= 56) begin
                        fp32_abs_div_scale_to_q8 = 24'd0;
                    end else begin
                        rounded = {19'd0, prod_q12} + (56'd1 << (rshift-1));
                        fp32_abs_div_scale_to_q8 = rounded >> rshift;
                    end
                end
            end
        end
    endfunction

    function [23:0] apply_inv_es;
        input [23:0] norm_q8;
        input [1:0] es_idx;
        reg [39:0] tmp;
        begin
            case (es_idx)
                2'd0: apply_inv_es = norm_q8;
                2'd1: begin
                    tmp = ({16'd0, norm_q8} * 8'd205) + 40'd128;
                    apply_inv_es = tmp[31:8];
                end
                2'd2: begin
                    tmp = ({16'd0, norm_q8} * 8'd171) + 40'd128;
                    apply_inv_es = tmp[31:8];
                end
                default: begin
                    tmp = ({16'd0, norm_q8} * 8'd146) + 40'd128;
                    apply_inv_es = tmp[31:8];
                end
            endcase
        end
    endfunction

    function [2:0] quant_mag_fp4_e2m1;
        input [23:0] mag_q8;
        begin
            if      (mag_q8 < 24'd96)   quant_mag_fp4_e2m1 = 3'b000; // 0
            else if (mag_q8 < 24'd224)  quant_mag_fp4_e2m1 = 3'b001; // 0.75
            else if (mag_q8 < 24'd320)  quant_mag_fp4_e2m1 = 3'b010; // 1.0
            else if (mag_q8 < 24'd448)  quant_mag_fp4_e2m1 = 3'b011; // 1.5
            else if (mag_q8 < 24'd640)  quant_mag_fp4_e2m1 = 3'b100; // 2.0
            else if (mag_q8 < 24'd896)  quant_mag_fp4_e2m1 = 3'b101; // 3.0
            else if (mag_q8 < 24'd1280) quant_mag_fp4_e2m1 = 3'b110; // 4.0
            else                         quant_mag_fp4_e2m1 = 3'b111; // 6.0
        end
    endfunction

    function [23:0] fp4_abs_to_q8;
        input [2:0] mag_code;
        begin
            case (mag_code)
                3'b000: fp4_abs_to_q8 = 24'd0;
                3'b001: fp4_abs_to_q8 = 24'd192;
                3'b010: fp4_abs_to_q8 = 24'd256;
                3'b011: fp4_abs_to_q8 = 24'd384;
                3'b100: fp4_abs_to_q8 = 24'd512;
                3'b101: fp4_abs_to_q8 = 24'd768;
                3'b110: fp4_abs_to_q8 = 24'd1024;
                default: fp4_abs_to_q8 = 24'd1536;
            endcase
        end
    endfunction

    function [4:0] abs_err_to_lut_idx;
        input [23:0] abs_err_q8;
        reg [19:0] err_q4;
        begin
            err_q4 = (abs_err_q8 + 24'd8) >> 4;
            abs_err_to_lut_idx = (err_q4 > 20'd16) ? 5'd16 : err_q4[4:0];
        end
    endfunction

    function [31:0] error_lut_4x17;
        input [1:0] es_sel;
        input [4:0] err_idx;
        begin
            case (es_sel)
                2'd0: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 32'd0;
                        5'd1:  error_lut_4x17 = 32'd16;
                        5'd2:  error_lut_4x17 = 32'd64;
                        5'd3:  error_lut_4x17 = 32'd144;
                        5'd4:  error_lut_4x17 = 32'd256;
                        5'd5:  error_lut_4x17 = 32'd400;
                        5'd6:  error_lut_4x17 = 32'd576;
                        5'd7:  error_lut_4x17 = 32'd784;
                        5'd8:  error_lut_4x17 = 32'd1024;
                        5'd9:  error_lut_4x17 = 32'd1296;
                        5'd10: error_lut_4x17 = 32'd1600;
                        5'd11: error_lut_4x17 = 32'd1936;
                        5'd12: error_lut_4x17 = 32'd2304;
                        5'd13: error_lut_4x17 = 32'd2704;
                        5'd14: error_lut_4x17 = 32'd3136;
                        5'd15: error_lut_4x17 = 32'd3600;
                        default: error_lut_4x17 = 32'd4096;
                    endcase
                end
                2'd1: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 32'd0;
                        5'd1:  error_lut_4x17 = 32'd25;
                        5'd2:  error_lut_4x17 = 32'd100;
                        5'd3:  error_lut_4x17 = 32'd225;
                        5'd4:  error_lut_4x17 = 32'd400;
                        5'd5:  error_lut_4x17 = 32'd625;
                        5'd6:  error_lut_4x17 = 32'd900;
                        5'd7:  error_lut_4x17 = 32'd1225;
                        5'd8:  error_lut_4x17 = 32'd1600;
                        5'd9:  error_lut_4x17 = 32'd2025;
                        5'd10: error_lut_4x17 = 32'd2500;
                        5'd11: error_lut_4x17 = 32'd3025;
                        5'd12: error_lut_4x17 = 32'd3600;
                        5'd13: error_lut_4x17 = 32'd4225;
                        5'd14: error_lut_4x17 = 32'd4900;
                        5'd15: error_lut_4x17 = 32'd5625;
                        default: error_lut_4x17 = 32'd6400;
                    endcase
                end
                2'd2: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 32'd0;
                        5'd1:  error_lut_4x17 = 32'd36;
                        5'd2:  error_lut_4x17 = 32'd144;
                        5'd3:  error_lut_4x17 = 32'd324;
                        5'd4:  error_lut_4x17 = 32'd576;
                        5'd5:  error_lut_4x17 = 32'd900;
                        5'd6:  error_lut_4x17 = 32'd1296;
                        5'd7:  error_lut_4x17 = 32'd1764;
                        5'd8:  error_lut_4x17 = 32'd2304;
                        5'd9:  error_lut_4x17 = 32'd2916;
                        5'd10: error_lut_4x17 = 32'd3600;
                        5'd11: error_lut_4x17 = 32'd4356;
                        5'd12: error_lut_4x17 = 32'd5184;
                        5'd13: error_lut_4x17 = 32'd6084;
                        5'd14: error_lut_4x17 = 32'd7056;
                        5'd15: error_lut_4x17 = 32'd8100;
                        default: error_lut_4x17 = 32'd9216;
                    endcase
                end
                default: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 32'd0;
                        5'd1:  error_lut_4x17 = 32'd49;
                        5'd2:  error_lut_4x17 = 32'd196;
                        5'd3:  error_lut_4x17 = 32'd441;
                        5'd4:  error_lut_4x17 = 32'd784;
                        5'd5:  error_lut_4x17 = 32'd1225;
                        5'd6:  error_lut_4x17 = 32'd1764;
                        5'd7:  error_lut_4x17 = 32'd2401;
                        5'd8:  error_lut_4x17 = 32'd3136;
                        5'd9:  error_lut_4x17 = 32'd3969;
                        5'd10: error_lut_4x17 = 32'd4900;
                        5'd11: error_lut_4x17 = 32'd5929;
                        5'd12: error_lut_4x17 = 32'd7056;
                        5'd13: error_lut_4x17 = 32'd8281;
                        5'd14: error_lut_4x17 = 32'd9604;
                        5'd15: error_lut_4x17 = 32'd11025;
                        default: error_lut_4x17 = 32'd12544;
                    endcase
                end
            endcase
        end
    endfunction

    // -------------------- S1: divide by group scale --------------------
    reg        v1;
    reg        sign_s1;
    reg [1:0]  es_s1;
    reg [23:0] norm_q8_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            sign_s1 <= 1'b0;
            es_s1 <= 2'd0;
            norm_q8_s1 <= 24'd0;
        end else begin
            v1 <= in_valid;
            sign_s1 <= x_fp32[31];
            es_s1 <= es_idx;
            norm_q8_s1 <= fp32_abs_div_scale_to_q8(x_fp32, group_scale);
        end
    end

    // -------------------- S2: apply inverse metadata scale --------------------
    reg        v2;
    reg        sign_s2;
    reg [1:0]  es_s2;
    reg [23:0] norm_es_q8_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0;
            sign_s2 <= 1'b0;
            es_s2 <= 2'd0;
            norm_es_q8_s2 <= 24'd0;
        end else begin
            v2 <= v1;
            sign_s2 <= sign_s1;
            es_s2 <= es_s1;
            norm_es_q8_s2 <= apply_inv_es(norm_q8_s1, es_s1);
        end
    end

    // -------------------- S3: quantize to FP4 magnitude --------------------
    reg        v3;
    reg        sign_s3;
    reg [1:0]  es_s3;
    reg [23:0] norm_es_q8_s3;
    reg [2:0]  fp4_mag_s3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v3 <= 1'b0;
            sign_s3 <= 1'b0;
            es_s3 <= 2'd0;
            norm_es_q8_s3 <= 24'd0;
            fp4_mag_s3 <= 3'd0;
        end else begin
            v3 <= v2;
            sign_s3 <= sign_s2;
            es_s3 <= es_s2;
            norm_es_q8_s3 <= norm_es_q8_s2;
            fp4_mag_s3 <= quant_mag_fp4_e2m1(norm_es_q8_s2);
        end
    end

    // -------------------- S4: subtract, abs error, and integer LUT index --------------------
    wire [23:0] fp4_q8_s3 = fp4_abs_to_q8(fp4_mag_s3);
    wire [23:0] abs_err_q8_s3 = (norm_es_q8_s3 >= fp4_q8_s3) ?
                                (norm_es_q8_s3 - fp4_q8_s3) :
                                (fp4_q8_s3 - norm_es_q8_s3);

    reg        v4;
    reg [1:0]  es_s4;
    reg [3:0]  fp4_s4;
    reg [4:0]  err_idx_s4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v4 <= 1'b0;
            es_s4 <= 2'd0;
            fp4_s4 <= 4'd0;
            err_idx_s4 <= 5'd0;
        end else begin
            v4 <= v3;
            es_s4 <= es_s3;
            fp4_s4 <= (fp4_mag_s3 == 3'b000) ? 4'b0000 : {sign_s3, fp4_mag_s3};
            err_idx_s4 <= abs_err_to_lut_idx(abs_err_q8_s3);
        end
    end

    // -------------------- S5: 4 x 17 LUT cost lookup --------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            es_idx_out <= 2'd0;
            fp4 <= 4'd0;
            cost <= 32'd0;
        end else begin
            out_valid <= v4;
            es_idx_out <= es_s4;
            fp4 <= fp4_s4;
            cost <= error_lut_4x17(es_s4, err_idx_s4);
        end
    end
endmodule

module nvesm2_subgroup_accum (
    input             clk,
    input             rst_n,
    input             in_valid,
    input      [1:0]  es_idx,
    input      [31:0] cost0,
    input      [31:0] cost1,
    input      [31:0] cost2,
    input      [31:0] cost3,
    input      [31:0] cost4,
    input      [31:0] cost5,
    input      [31:0] cost6,
    input      [31:0] cost7,
    output reg        out_valid,
    output reg [1:0]  es_idx_out,
    output reg [35:0] subgroup_cost
);
    reg        v1, v2;
    reg [1:0]  es1, es2;
    reg [32:0] s1_0, s1_1, s1_2, s1_3;
    reg [34:0] s2_0, s2_1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            es1 <= 2'd0;
            s1_0 <= 33'd0;
            s1_1 <= 33'd0;
            s1_2 <= 33'd0;
            s1_3 <= 33'd0;
        end else begin
            v1 <= in_valid;
            es1 <= es_idx;
            s1_0 <= {1'b0, cost0} + {1'b0, cost1};
            s1_1 <= {1'b0, cost2} + {1'b0, cost3};
            s1_2 <= {1'b0, cost4} + {1'b0, cost5};
            s1_3 <= {1'b0, cost6} + {1'b0, cost7};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0;
            es2 <= 2'd0;
            s2_0 <= 35'd0;
            s2_1 <= 35'd0;
        end else begin
            v2 <= v1;
            es2 <= es1;
            s2_0 <= {2'b00, s1_0} + {2'b00, s1_1};
            s2_1 <= {2'b00, s1_2} + {2'b00, s1_3};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            es_idx_out <= 2'd0;
            subgroup_cost <= 36'd0;
        end else begin
            out_valid <= v2;
            es_idx_out <= es2;
            subgroup_cost <= {1'b0, s2_0} + {1'b0, s2_1};
        end
    end
endmodule
