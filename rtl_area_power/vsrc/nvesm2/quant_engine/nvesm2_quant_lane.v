// =============================================================
// NVESM2 per-lane RQU.
// ES index maps to {1.00, 1.25, 1.50, 1.75}.
// FP4 format is E2M1: {sign, exp[1:0], mant[0]} with exponent bias 1.
// One instance evaluates one ES candidate at a time; the top level time-
// multiplexes four metadata candidates through the same per-lane hardware.
// ES scaling is folded into integer Q7 thresholds, so each lane only needs one
// FP32 multiply for abs(x) / group_scale.
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
    output reg [13:0] cost
);

    function [31:0] pack_pos_fp32;
        input signed [4:0] exp_unb;
        input [22:0] frac;
        reg signed [8:0] exp_biased;
        begin
            exp_biased = exp_unb + 9'sd127;
            if (exp_biased >= 9'sd255)
                pack_pos_fp32 = 32'h7f800000;
            else if (exp_biased <= 9'sd0)
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
                    3'd1: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd9,  23'h000000); // 512
                    3'd2: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd8,  23'h000000); // 256
                    3'd3: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd7,  23'h2aaaab); // 512/3
                    3'd4: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd7,  23'h000000); // 128
                    3'd5: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6,  23'h4ccccd); // 512/5
                    3'd6: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6,  23'h2aaaab); // 512/6
                    default: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6, 23'h124925); // 512/7
                endcase
            end else begin
                case (mant3)
                    3'd0: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd7 - $signed({1'b0, exp4}), 23'h000000);
                    3'd1: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6 - $signed({1'b0, exp4}), 23'h638e39);
                    3'd2: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6 - $signed({1'b0, exp4}), 23'h4ccccd);
                    3'd3: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6 - $signed({1'b0, exp4}), 23'h3a2e8c);
                    3'd4: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6 - $signed({1'b0, exp4}), 23'h2aaaab);
                    3'd5: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6 - $signed({1'b0, exp4}), 23'h1d89d9);
                    3'd6: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6 - $signed({1'b0, exp4}), 23'h124925);
                    default: e4m3_inv_to_fp32 = pack_pos_fp32(5'sd6 - $signed({1'b0, exp4}), 23'h088889);
                endcase
            end
        end
    endfunction

    function [12:0] fp32_to_q7_info;
        input [31:0] value;
        reg [7:0] exp_bits;
        reg [22:0] frac_bits;
        reg [23:0] sig;
        reg signed [8:0] exp_unb;
        reg signed [8:0] shift;
        reg [7:0] rshift;
        reg [23:0] scaled;
        reg exact;
        begin
            exp_bits = value[30:23];
            frac_bits = value[22:0];
            sig = (exp_bits == 8'd0) ? {1'b0, frac_bits} : {1'b1, frac_bits};
            exp_unb = (exp_bits == 8'd0) ? -9'sd126 :
                       ($signed({1'b0, exp_bits}) - 9'sd127);
            shift = exp_unb + 9'sd7 - 9'sd23;
            scaled = 24'd0;
            exact = 1'b0;

            if (value[30:0] == 31'd0) begin
                scaled = 24'd0;
                exact = 1'b1;
            end else if (exp_bits == 8'hff) begin
                scaled = 24'd4095;
            end else if (shift >= 9'sd0) begin
                scaled = 24'd4096;
            end else begin
                rshift = -shift;
                if (rshift >= 8'd24) begin
                    scaled = 24'd0;
                    exact = (sig == 24'd0);
                end else begin
                    scaled = sig >> rshift;
                    exact = ((sig & ((24'd1 << rshift) - 24'd1)) == 24'd0);
                end
            end

            if (scaled > 24'd4095)
                fp32_to_q7_info = {1'b0, 12'hfff};
            else
                fp32_to_q7_info = {exact, scaled[11:0]};
        end
    endfunction

    function [7:0] fp4_base_q5;
        input [2:0] mag_code;
        begin
            case (mag_code)
                3'b000: fp4_base_q5 = 8'd0;
                3'b001: fp4_base_q5 = 8'd24;  // 0.75 * 32
                3'b010: fp4_base_q5 = 8'd32;  // 1.00 * 32
                3'b011: fp4_base_q5 = 8'd48;  // 1.50 * 32
                3'b100: fp4_base_q5 = 8'd64;  // 2.00 * 32
                3'b101: fp4_base_q5 = 8'd96;  // 3.00 * 32
                3'b110: fp4_base_q5 = 8'd128; // 4.00 * 32
                default: fp4_base_q5 = 8'd192; // 6.00 * 32
            endcase
        end
    endfunction

    function [7:0] quant_bound_base_q5;
        input [2:0] bound_idx;
        begin
            case (bound_idx)
                3'd0: quant_bound_base_q5 = 8'd12;  // 0.375 * 32
                3'd1: quant_bound_base_q5 = 8'd28;  // 0.875 * 32
                3'd2: quant_bound_base_q5 = 8'd40;  // 1.250 * 32
                3'd3: quant_bound_base_q5 = 8'd56;  // 1.750 * 32
                3'd4: quant_bound_base_q5 = 8'd80;  // 2.500 * 32
                3'd5: quant_bound_base_q5 = 8'd112; // 3.500 * 32
                default: quant_bound_base_q5 = 8'd160; // 5.000 * 32
            endcase
        end
    endfunction

    function [11:0] scaled_q7;
        input [7:0] base_q5;
        input [1:0] es_idx;
        reg [11:0] base;
        begin
            base = {4'd0, base_q5};
            case (es_idx)
                2'd0: scaled_q7 = base << 2;
                2'd1: scaled_q7 = (base << 2) + base;
                2'd2: scaled_q7 = (base << 2) + (base << 1);
                default: scaled_q7 = (base << 3) - base;
            endcase
        end
    endfunction

    function [2:0] quant_mag_fp4_e2m1_q7;
        input [11:0] mag_q7;
        input [1:0]  es_idx;
        begin
            if      (mag_q7 < scaled_q7(quant_bound_base_q5(3'd0), es_idx)) quant_mag_fp4_e2m1_q7 = 3'b000;
            else if (mag_q7 < scaled_q7(quant_bound_base_q5(3'd1), es_idx)) quant_mag_fp4_e2m1_q7 = 3'b001;
            else if (mag_q7 < scaled_q7(quant_bound_base_q5(3'd2), es_idx)) quant_mag_fp4_e2m1_q7 = 3'b010;
            else if (mag_q7 < scaled_q7(quant_bound_base_q5(3'd3), es_idx)) quant_mag_fp4_e2m1_q7 = 3'b011;
            else if (mag_q7 < scaled_q7(quant_bound_base_q5(3'd4), es_idx)) quant_mag_fp4_e2m1_q7 = 3'b100;
            else if (mag_q7 < scaled_q7(quant_bound_base_q5(3'd5), es_idx)) quant_mag_fp4_e2m1_q7 = 3'b101;
            else if (mag_q7 < scaled_q7(quant_bound_base_q5(3'd6), es_idx)) quant_mag_fp4_e2m1_q7 = 3'b110;
            else                                                            quant_mag_fp4_e2m1_q7 = 3'b111;
        end
    endfunction

    function [11:0] fp4_es_q7;
        input [2:0] mag_code;
        input [1:0] es_idx;
        begin
            fp4_es_q7 = scaled_q7(fp4_base_q5(mag_code), es_idx);
        end
    endfunction

    function [4:0] abs_err_to_lut_idx_q7;
        input [11:0] mag_q7;
        input        mag_q7_exact;
        input [1:0]  es_idx;
        input [2:0]  mag_code;
        integer idx;
        reg [11:0] q_q7;
        reg [11:0] err_q7;
        reg [11:0] lo_q7;
        reg [11:0] hi_q7;
        reg lo_ok;
        reg hi_ok;
        reg found;
        begin
            q_q7 = fp4_es_q7(mag_code, es_idx);
            found = 1'b0;
            abs_err_to_lut_idx_q7 = 5'd16;

            for (idx=0; idx<16; idx=idx+1) begin
                err_q7 = scaled_q7((idx << 1) + 1, es_idx);
                lo_q7 = (q_q7 > err_q7) ? (q_q7 - err_q7) : 12'd0;
                hi_q7 = q_q7 + err_q7;
                lo_ok = (q_q7 <= err_q7) ||
                        (mag_q7 > lo_q7) ||
                        ((mag_q7 == lo_q7) && !mag_q7_exact);
                hi_ok = (mag_q7 < hi_q7);

                if (!found && lo_ok && hi_ok) begin
                    abs_err_to_lut_idx_q7 = idx[4:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    // 4 ES candidates x 17 error buckets.  Max value is 12544, so 14 bits
    // are enough for each per-lane cost.
    function [13:0] error_lut_4x17;
        input [1:0] es_sel;  // selected ES candidate
        input [4:0] err_idx; // clipped error bucket, 0..16
        begin
            case (es_sel)
                2'd0: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 14'd0;
                        5'd1:  error_lut_4x17 = 14'd16;
                        5'd2:  error_lut_4x17 = 14'd64;
                        5'd3:  error_lut_4x17 = 14'd144;
                        5'd4:  error_lut_4x17 = 14'd256;
                        5'd5:  error_lut_4x17 = 14'd400;
                        5'd6:  error_lut_4x17 = 14'd576;
                        5'd7:  error_lut_4x17 = 14'd784;
                        5'd8:  error_lut_4x17 = 14'd1024;
                        5'd9:  error_lut_4x17 = 14'd1296;
                        5'd10: error_lut_4x17 = 14'd1600;
                        5'd11: error_lut_4x17 = 14'd1936;
                        5'd12: error_lut_4x17 = 14'd2304;
                        5'd13: error_lut_4x17 = 14'd2704;
                        5'd14: error_lut_4x17 = 14'd3136;
                        5'd15: error_lut_4x17 = 14'd3600;
                        default: error_lut_4x17 = 14'd4096;
                    endcase
                end
                2'd1: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 14'd0;
                        5'd1:  error_lut_4x17 = 14'd25;
                        5'd2:  error_lut_4x17 = 14'd100;
                        5'd3:  error_lut_4x17 = 14'd225;
                        5'd4:  error_lut_4x17 = 14'd400;
                        5'd5:  error_lut_4x17 = 14'd625;
                        5'd6:  error_lut_4x17 = 14'd900;
                        5'd7:  error_lut_4x17 = 14'd1225;
                        5'd8:  error_lut_4x17 = 14'd1600;
                        5'd9:  error_lut_4x17 = 14'd2025;
                        5'd10: error_lut_4x17 = 14'd2500;
                        5'd11: error_lut_4x17 = 14'd3025;
                        5'd12: error_lut_4x17 = 14'd3600;
                        5'd13: error_lut_4x17 = 14'd4225;
                        5'd14: error_lut_4x17 = 14'd4900;
                        5'd15: error_lut_4x17 = 14'd5625;
                        default: error_lut_4x17 = 14'd6400;
                    endcase
                end
                2'd2: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 14'd0;
                        5'd1:  error_lut_4x17 = 14'd36;
                        5'd2:  error_lut_4x17 = 14'd144;
                        5'd3:  error_lut_4x17 = 14'd324;
                        5'd4:  error_lut_4x17 = 14'd576;
                        5'd5:  error_lut_4x17 = 14'd900;
                        5'd6:  error_lut_4x17 = 14'd1296;
                        5'd7:  error_lut_4x17 = 14'd1764;
                        5'd8:  error_lut_4x17 = 14'd2304;
                        5'd9:  error_lut_4x17 = 14'd2916;
                        5'd10: error_lut_4x17 = 14'd3600;
                        5'd11: error_lut_4x17 = 14'd4356;
                        5'd12: error_lut_4x17 = 14'd5184;
                        5'd13: error_lut_4x17 = 14'd6084;
                        5'd14: error_lut_4x17 = 14'd7056;
                        5'd15: error_lut_4x17 = 14'd8100;
                        default: error_lut_4x17 = 14'd9216;
                    endcase
                end
                default: begin
                    case (err_idx)
                        5'd0:  error_lut_4x17 = 14'd0;
                        5'd1:  error_lut_4x17 = 14'd49;
                        5'd2:  error_lut_4x17 = 14'd196;
                        5'd3:  error_lut_4x17 = 14'd441;
                        5'd4:  error_lut_4x17 = 14'd784;
                        5'd5:  error_lut_4x17 = 14'd1225;
                        5'd6:  error_lut_4x17 = 14'd1764;
                        5'd7:  error_lut_4x17 = 14'd2401;
                        5'd8:  error_lut_4x17 = 14'd3136;
                        5'd9:  error_lut_4x17 = 14'd3969;
                        5'd10: error_lut_4x17 = 14'd4900;
                        5'd11: error_lut_4x17 = 14'd5929;
                        5'd12: error_lut_4x17 = 14'd7056;
                        5'd13: error_lut_4x17 = 14'd8281;
                        5'd14: error_lut_4x17 = 14'd9604;
                        5'd15: error_lut_4x17 = 14'd11025;
                        default: error_lut_4x17 = 14'd12544;
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

    wire [12:0] norm_q7_info_s1 = fp32_to_q7_info(norm_fp32_s1);

    // -------------------- S2: convert norm to Q7 threshold domain --------------------
    reg        v2;
    reg        sign_s2;
    reg [1:0]  es_s2;
    reg [11:0] norm_q7_s2;
    reg        norm_q7_exact_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0;
            sign_s2 <= 1'b0;
            es_s2 <= 2'd0;
            norm_q7_s2 <= 12'd0;
            norm_q7_exact_s2 <= 1'b1;
        end else begin
            v2 <= v1;
            sign_s2 <= sign_s1;
            es_s2 <= es_s1;
            norm_q7_s2 <= norm_q7_info_s1[11:0];
            norm_q7_exact_s2 <= norm_q7_info_s1[12];
        end
    end

    // -------------------- S3: quantize to FP4 magnitude --------------------
    reg        v3;
    reg        sign_s3;
    reg [1:0]  es_s3;
    reg [11:0] norm_q7_s3;
    reg        norm_q7_exact_s3;
    reg [2:0]  fp4_mag_s3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v3 <= 1'b0;
            sign_s3 <= 1'b0;
            es_s3 <= 2'd0;
            norm_q7_s3 <= 12'd0;
            norm_q7_exact_s3 <= 1'b1;
            fp4_mag_s3 <= 3'd0;
        end else begin
            v3 <= v2;
            sign_s3 <= sign_s2;
            es_s3 <= es_s2;
            norm_q7_s3 <= norm_q7_s2;
            norm_q7_exact_s3 <= norm_q7_exact_s2;
            fp4_mag_s3 <= quant_mag_fp4_e2m1_q7(norm_q7_s2, es_s2);
        end
    end

    // -------------------- S4: bucket absolute error in Q7 threshold domain --------------------
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
            err_idx_s4 <= abs_err_to_lut_idx_q7(
                norm_q7_s3,
                norm_q7_exact_s3,
                es_s3,
                fp4_mag_s3
            );
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
            cost <= 14'd0;
        end else begin
            out_valid <= v4;
            es_idx_out <= es_s4;
            fp4 <= fp4_s4;
            cost <= error_lut_4x17(es_s4, err_idx_s4);
        end
    end
endmodule
