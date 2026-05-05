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

    function [31:0] inv_es_to_fp32;
        input [1:0] es_idx;
        begin
            case (es_idx)
                2'd0: inv_es_to_fp32 = 32'h3f800000; // 1 / 1.00
                2'd1: inv_es_to_fp32 = 32'h3f4ccccd; // 1 / 1.25
                2'd2: inv_es_to_fp32 = 32'h3f2aaaab; // 1 / 1.50
                default: inv_es_to_fp32 = 32'h3f124925; // 1 / 1.75
            endcase
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

    function [31:0] fp4_abs_to_fp32;
        input [2:0] mag_code;
        begin
            case (mag_code)
                3'b000: fp4_abs_to_fp32 = 32'h00000000;
                3'b001: fp4_abs_to_fp32 = 32'h3f400000;
                3'b010: fp4_abs_to_fp32 = 32'h3f800000;
                3'b011: fp4_abs_to_fp32 = 32'h3fc00000;
                3'b100: fp4_abs_to_fp32 = 32'h40000000;
                3'b101: fp4_abs_to_fp32 = 32'h40400000;
                3'b110: fp4_abs_to_fp32 = 32'h40800000;
                default: fp4_abs_to_fp32 = 32'h40c00000;
            endcase
        end
    endfunction

    function [4:0] abs_err_to_lut_idx_fp32;
        input [31:0] abs_err_fp32;
        begin
            if      (fp32_lt_pos(abs_err_fp32, 32'h3d000000)) abs_err_to_lut_idx_fp32 = 5'd0;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3dc00000)) abs_err_to_lut_idx_fp32 = 5'd1;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3e200000)) abs_err_to_lut_idx_fp32 = 5'd2;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3e600000)) abs_err_to_lut_idx_fp32 = 5'd3;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3e900000)) abs_err_to_lut_idx_fp32 = 5'd4;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3eb00000)) abs_err_to_lut_idx_fp32 = 5'd5;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3ed00000)) abs_err_to_lut_idx_fp32 = 5'd6;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3ef00000)) abs_err_to_lut_idx_fp32 = 5'd7;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f080000)) abs_err_to_lut_idx_fp32 = 5'd8;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f180000)) abs_err_to_lut_idx_fp32 = 5'd9;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f280000)) abs_err_to_lut_idx_fp32 = 5'd10;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f380000)) abs_err_to_lut_idx_fp32 = 5'd11;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f480000)) abs_err_to_lut_idx_fp32 = 5'd12;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f580000)) abs_err_to_lut_idx_fp32 = 5'd13;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f680000)) abs_err_to_lut_idx_fp32 = 5'd14;
            else if (fp32_lt_pos(abs_err_fp32, 32'h3f780000)) abs_err_to_lut_idx_fp32 = 5'd15;
            else                                              abs_err_to_lut_idx_fp32 = 5'd16;
        end
    endfunction

    // 4 ES candidates x 17 error buckets.  Values are squared-error weights
    // used only for relative comparison between ES candidates in a subgroup.
    function [31:0] error_lut_4x17;
        input [1:0] es_sel;  // selected ES candidate
        input [4:0] err_idx; // clipped error bucket, 0..16
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

    wire [31:0] abs_x_fp32 = {1'b0, x_fp32[30:0]};
    wire [31:0] group_scale_inv_fp32 = e4m3_inv_to_fp32(group_scale);
    wire [31:0] norm_div_mul_fp32;
    wire        group_scale_zero = (group_scale[6:0] == 7'd0);
    wire        x_zero = (x_fp32[30:0] == 31'd0);
    wire [31:0] norm_fp32_comb = group_scale_zero ?
                                  (x_zero ? 32'h00000000 : 32'h7f800000) :
                                  norm_div_mul_fp32;

    nvesm2_fp32_mul U_DIV_GROUP_SCALE (
        .a(abs_x_fp32),
        .b(group_scale_inv_fp32),
        .z(norm_div_mul_fp32)
    );

    // -------------------- S1: divide by group scale --------------------
    reg        v1;          // valid bit for stage S1 payload
    reg        sign_s1;     // original FP32 sign bit
    reg [1:0]  es_s1;       // ES candidate being evaluated for this lane
    reg [31:0] norm_fp32_s1; // abs(x_fp32 / group_scale), still FP32

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            sign_s1 <= 1'b0;
            es_s1 <= 2'd0;
            norm_fp32_s1 <= 32'h00000000;
        end else begin
            v1 <= in_valid;
            sign_s1 <= x_fp32[31];
            es_s1 <= es_idx;
            norm_fp32_s1 <= norm_fp32_comb;
        end
    end

    wire [31:0] inv_es_fp32_s1 = inv_es_to_fp32(es_s1);
    wire [31:0] norm_es_fp32_comb;

    nvesm2_fp32_mul U_APPLY_ES (
        .a(norm_fp32_s1),
        .b(inv_es_fp32_s1),
        .z(norm_es_fp32_comb)
    );

    // -------------------- S2: apply inverse metadata scale --------------------
    reg        v2;             // valid bit for stage S2 payload
    reg        sign_s2;        // sign delayed from S1
    reg [1:0]  es_s2;          // ES candidate delayed from S1
    reg [31:0] norm_es_fp32_s2; // norm * LUT[1 / ES], still FP32

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0;
            sign_s2 <= 1'b0;
            es_s2 <= 2'd0;
            norm_es_fp32_s2 <= 32'h00000000;
        end else begin
            v2 <= v1;
            sign_s2 <= sign_s1;
            es_s2 <= es_s1;
            norm_es_fp32_s2 <= norm_es_fp32_comb;
        end
    end

    // -------------------- S3: quantize to FP4 magnitude --------------------
    reg        v3;             // valid bit for stage S3 payload
    reg        sign_s3;        // sign delayed from S2
    reg [1:0]  es_s3;          // ES candidate delayed from S2
    reg [31:0] norm_es_fp32_s3; // pre-quantization magnitude kept for error calculation
    reg [2:0]  fp4_mag_s3;     // unsigned E2M1 magnitude code chosen by threshold compare

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v3 <= 1'b0;
            sign_s3 <= 1'b0;
            es_s3 <= 2'd0;
            norm_es_fp32_s3 <= 32'h00000000;
            fp4_mag_s3 <= 3'd0;
        end else begin
            v3 <= v2;
            sign_s3 <= sign_s2;
            es_s3 <= es_s2;
            norm_es_fp32_s3 <= norm_es_fp32_s2;
            fp4_mag_s3 <= quant_mag_fp4_e2m1_fp32(norm_es_fp32_s2);
        end
    end

    // -------------------- S4: Calculate quantization err and convert to 0~16 --------------------
    wire [31:0] fp4_fp32_s3 = fp4_abs_to_fp32(fp4_mag_s3);
    wire [31:0] abs_err_fp32_s3;

    nvesm2_fp32_abs_diff_pos U_ERR_DIFF (
        .a(norm_es_fp32_s3),
        .b(fp4_fp32_s3),
        .z(abs_err_fp32_s3)
    );

    reg        v4;         // valid bit for stage S4 payload
    reg [1:0]  es_s4;      // ES candidate delayed from S3
    reg [3:0]  fp4_s4;     // signed FP4 code; zero is forced positive
    reg [4:0]  err_idx_s4; // address into the squared-error LUT

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
            err_idx_s4 <= abs_err_to_lut_idx_fp32(abs_err_fp32_s3);
        end
    end

    // -------------------- S5: 4 x 17 LUT cost lookup --------------------
    // Output latency from in_valid is five cycles.  out_valid, es_idx_out,
    // fp4, and cost are aligned for one lane and one ES candidate.
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
