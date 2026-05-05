// =============================================================
// NVESM2 16-lane FP32 group scale generator.
// Computes group_scale(e4m3) = max(abs(x[15:0])) / E2M1_max.
// E2M1_max is 6.0; division is implemented as reciprocal multiply.
// The 16-lane max tree and scale conversion are pipelined.
// =============================================================
module nvesm2_group_scale (
    input              clk,
    input              rst_n,
    input              in_valid,
    input  [16*32-1:0] fp32_bus,
    output reg         out_valid,
    output reg [7:0]   group_scale_e4m3
);
    localparam [12:0] RECIP_E2M1_MAX_NORM_Q12 = 13'd5461; // round(2^12 * 4 / 3), with exponent -3

    genvar gi; // generate index for unpacking the 16 FP32 absolute values

    // Right-shift with round-to-nearest behavior for a 24-bit significand.
    // value is treated as unsigned; sh is the requested right-shift amount.
    function [23:0] rshift_round_u24;
        input [23:0] value;
        input integer sh;
        reg [24:0] sum; // one extra bit holds the rounding addend carry
        begin
            if (sh <= 0)
                rshift_round_u24 = value;
            else if (sh >= 24)
                rshift_round_u24 = 24'd0;
            else begin
                sum = {1'b0, value} + (25'd1 << (sh-1));
                rshift_round_u24 = sum >> sh;
            end
        end
    endfunction

    // Encode a positive normalized value into unsigned E4M3.
    // exp_unb is the unbiased exponent of sig24[23], and sig24 is the
    // 1.x-style significand used for both normal and subnormal rounding.
    function [7:0] sig_exp_to_e4m3;
        input signed [10:0] exp_unb; // unbiased FP exponent before E4M3 biasing
        input [23:0] sig24;          // 24-bit significand with binary point after bit 23
        integer sub_shift;           // shift amount used to create E4M3 subnormals
        reg [3:0] keep4;             // hidden bit plus 3 mantissa bits before rounding
        reg guard;                   // first discarded bit for round-to-nearest-even
        reg sticky;                  // OR of all lower discarded bits
        reg inc;                     // rounding increment bit
        reg [4:0] rounded4;          // keep4 plus possible carry from rounding
        reg [23:0] sub_mag;          // rounded subnormal magnitude before mantissa clamp
        reg [3:0] sub_mant;          // subnormal mantissa candidate, including overflow bit
        reg [3:0] norm_exp4;         // biased 4-bit E4M3 exponent
        reg signed [10:0] exp_r;     // exponent after possible rounding carry
        begin
            exp_r = exp_unb;
            if (sig24 == 24'd0 || exp_r < -11'sd10) begin
                sig_exp_to_e4m3 = 8'h00;
            end else if (exp_r > 11'sd7) begin
                sig_exp_to_e4m3 = 8'h77;
            end else if (exp_r >= -11'sd6) begin
                keep4 = sig24[23:20];
                guard = sig24[19];
                sticky = |sig24[18:0];
                inc = guard & (sticky | keep4[0]);
                rounded4 = {1'b0, keep4} + {4'd0, inc};
                if (rounded4[4]) begin
                    exp_r = exp_r + 11'sd1;
                    rounded4 = 5'b01000;
                end
                if (exp_r > 11'sd7)
                    sig_exp_to_e4m3 = 8'h77;
                else begin
                    norm_exp4 = exp_r + 11'sd7;
                    sig_exp_to_e4m3 = {1'b0, norm_exp4, rounded4[2:0]};
                end
            end else begin
                sub_shift = 14 - exp_r;
                sub_mag = rshift_round_u24(sig24, sub_shift);
                sub_mant = sub_mag[3:0];
                if (sub_mant > 4'd7)
                    sig_exp_to_e4m3 = 8'h08;
                else
                    sig_exp_to_e4m3 = {1'b0, 4'b0000, sub_mant[2:0]};
            end
        end
    endfunction

    // Convert max_abs * (1 / E2M1_max) into E4M3.  prod_q12 is the product
    // of the 24-bit FP32 significand and RECIP_E2M1_MAX_NORM_Q12.
    function [7:0] prod_q12_to_e4m3;
        input scale_zero;            // max_abs was zero
        input scale_inf;             // max_abs was NaN/Inf; saturate the scale
        input signed [10:0] exp_unb; // unbiased exponent associated with prod_q12
        input [36:0] prod_q12;       // Q12 reciprocal multiply result
        reg [23:0] div_sig24;        // normalized significand after dividing by 6
        begin
            if (scale_zero) begin
                prod_q12_to_e4m3 = 8'h00;
            end else if (scale_inf) begin
                prod_q12_to_e4m3 = 8'h77;
            end else begin
                if (prod_q12 >= 37'h1000000000) begin
                    div_sig24 = rshift_round_u24(prod_q12[36:13], 0);
                    prod_q12_to_e4m3 = sig_exp_to_e4m3(exp_unb - 11'sd2, div_sig24);
                end else begin
                    div_sig24 = rshift_round_u24(prod_q12[35:12], 0);
                    prod_q12_to_e4m3 = sig_exp_to_e4m3(exp_unb - 11'sd3, div_sig24);
                end
            end
        end
    endfunction

    wire [16*31-1:0] abs_bus; // packed FP32 magnitudes, sign bit removed from each lane
    wire [30:0] max_abs;      // maximum positive FP32 encoding across the 16 lanes
    wire        max_valid;    // valid delayed through max_abs_fp32_16_pipe
    wire [7:0]  max_exp8  = max_abs[30:23]; // FP32 exponent field of max_abs
    wire [22:0] max_frac23 = max_abs[22:0]; // FP32 fraction field of max_abs
    wire [23:0] max_sig24 = (max_exp8 == 8'h00) ? {1'b0, max_frac23} : {1'b1, max_frac23};
    wire signed [10:0] max_exp_unb = (max_exp8 == 8'h00) ? -11'sd126 : ($signed({1'b0, max_exp8}) - 11'sd127);
    wire [36:0] max_prod_q12 = max_sig24 * RECIP_E2M1_MAX_NORM_Q12; // max_abs / 6 significand product

    // Conversion stage register.  It separates the max tree from the E4M3
    // encoder so the input-to-scale path does not remain one long combo arc.
    reg        conv_v1;             // valid for conversion-stage payload below
    reg        conv_zero_v1;        // registered max_abs == 0 special case
    reg        conv_inf_v1;         // registered max_abs exponent == 255 special case
    reg signed [10:0] conv_exp_unb_v1; // registered unbiased exponent
    reg [36:0] conv_prod_q12_v1;    // registered reciprocal multiply product

    generate
      for (gi=0; gi<16; gi=gi+1) begin: GEN_ABS
        // Drop the FP32 sign bit.  For positive FP encodings, unsigned
        // compare on {exp, frac} preserves numeric ordering.
        assign abs_bus[gi*31 + 30 : gi*31] = fp32_bus[gi*32 + 30 : gi*32];
      end
    endgenerate

    max_abs_fp32_16_pipe U_MAX_ABS (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .abs_bus(abs_bus),
        .out_valid(max_valid),
        .max_abs(max_abs)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_v1 <= 1'b0;
            conv_zero_v1 <= 1'b0;
            conv_inf_v1 <= 1'b0;
            conv_exp_unb_v1 <= 11'sd0;
            conv_prod_q12_v1 <= 37'd0;
            out_valid <= 1'b0;
            group_scale_e4m3 <= 8'h00;
        end else begin
            conv_v1 <= max_valid;
            conv_zero_v1 <= (max_abs == 31'd0);
            conv_inf_v1 <= (max_exp8 == 8'hff);
            conv_exp_unb_v1 <= max_exp_unb;
            conv_prod_q12_v1 <= max_prod_q12;

            out_valid <= conv_v1;
            group_scale_e4m3 <= prod_q12_to_e4m3(
                conv_zero_v1,
                conv_inf_v1,
                conv_exp_unb_v1,
                conv_prod_q12_v1
            );
        end
    end
endmodule

module max_abs_fp32_16_pipe (
    input             clk,
    input             rst_n,
    input             in_valid,
    input  [16*31-1:0] abs_bus,
    output reg        out_valid,
    output reg [30:0] max_abs
);
    integer mi; // reset loop index for the registered max arrays

    reg        v1; // valid after pairwise compare stage
    reg        v2; // valid after 4-way reduce stage
    reg        v3; // valid after 8-way reduce stage
    reg [30:0] max_s1 [0:7]; // stage 1: max of lane pairs, 16 inputs -> 8 values
    reg [30:0] max_s2 [0:3]; // stage 2: max of max_s1 pairs, 8 values -> 4 values
    reg [30:0] max_s3 [0:1]; // stage 3: max of max_s2 pairs, 4 values -> 2 values

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            v2 <= 1'b0;
            v3 <= 1'b0;
            out_valid <= 1'b0;
            max_abs <= 31'd0;
            for (mi=0; mi<8; mi=mi+1)
                max_s1[mi] <= 31'd0;
            for (mi=0; mi<4; mi=mi+1)
                max_s2[mi] <= 31'd0;
            for (mi=0; mi<2; mi=mi+1)
                max_s3[mi] <= 31'd0;
        end else begin
            v1 <= in_valid;
            v2 <= v1;
            v3 <= v2;
            out_valid <= v3;

            max_s1[0] <= (abs_bus[0*31 + 30 : 0*31] > abs_bus[1*31 + 30 : 1*31]) ? abs_bus[0*31 + 30 : 0*31] : abs_bus[1*31 + 30 : 1*31];
            max_s1[1] <= (abs_bus[2*31 + 30 : 2*31] > abs_bus[3*31 + 30 : 3*31]) ? abs_bus[2*31 + 30 : 2*31] : abs_bus[3*31 + 30 : 3*31];
            max_s1[2] <= (abs_bus[4*31 + 30 : 4*31] > abs_bus[5*31 + 30 : 5*31]) ? abs_bus[4*31 + 30 : 4*31] : abs_bus[5*31 + 30 : 5*31];
            max_s1[3] <= (abs_bus[6*31 + 30 : 6*31] > abs_bus[7*31 + 30 : 7*31]) ? abs_bus[6*31 + 30 : 6*31] : abs_bus[7*31 + 30 : 7*31];
            max_s1[4] <= (abs_bus[8*31 + 30 : 8*31] > abs_bus[9*31 + 30 : 9*31]) ? abs_bus[8*31 + 30 : 8*31] : abs_bus[9*31 + 30 : 9*31];
            max_s1[5] <= (abs_bus[10*31 + 30 : 10*31] > abs_bus[11*31 + 30 : 11*31]) ? abs_bus[10*31 + 30 : 10*31] : abs_bus[11*31 + 30 : 11*31];
            max_s1[6] <= (abs_bus[12*31 + 30 : 12*31] > abs_bus[13*31 + 30 : 13*31]) ? abs_bus[12*31 + 30 : 12*31] : abs_bus[13*31 + 30 : 13*31];
            max_s1[7] <= (abs_bus[14*31 + 30 : 14*31] > abs_bus[15*31 + 30 : 15*31]) ? abs_bus[14*31 + 30 : 14*31] : abs_bus[15*31 + 30 : 15*31];

            max_s2[0] <= (max_s1[0] > max_s1[1]) ? max_s1[0] : max_s1[1];
            max_s2[1] <= (max_s1[2] > max_s1[3]) ? max_s1[2] : max_s1[3];
            max_s2[2] <= (max_s1[4] > max_s1[5]) ? max_s1[4] : max_s1[5];
            max_s2[3] <= (max_s1[6] > max_s1[7]) ? max_s1[6] : max_s1[7];

            max_s3[0] <= (max_s2[0] > max_s2[1]) ? max_s2[0] : max_s2[1];
            max_s3[1] <= (max_s2[2] > max_s2[3]) ? max_s2[2] : max_s2[3];

            max_abs <= (max_s3[0] > max_s3[1]) ? max_s3[0] : max_s3[1];
        end
    end
endmodule
