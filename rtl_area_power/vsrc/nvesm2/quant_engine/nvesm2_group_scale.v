// =============================================================
// NVESM2 16-lane FP32 group scale generator, simple pipeline.
//
// Operation:
//   1. group_max   = max(abs(fp32_bus[15:0]))
//   2. raw_scale   = group_max * LUT[1 / E2M1_max]
//                  = group_max * (1 / 6)
//   3. group_scale = quantize_unsigned_e4m3(raw_scale)
//
// =============================================================
module nvesm2_group_scale (
    input              clk,
    input              rst_n,
    input              in_valid,
    input  [16*32-1:0] fp32_bus,
    output reg         out_valid,
    output reg [7:0]   group_scale_e4m3
);
    // E2M1 max value is 6.0, so 1 / E2M1_max = 1/6.
    // Store it as normalized significand 4/3 in Q12 with binary exponent -3.
    localparam [12:0] INV_E2M1_MAX_SIG_Q12 = 13'd5461;

    integer ii;
    genvar gi;

    function [30:0] max_abs2;
        input [30:0] a;
        input [30:0] b;
        begin
            max_abs2 = (a > b) ? a : b;
        end
    endfunction

    function [23:0] rshift_round_u24;
        input [23:0] value;
        input [4:0] sh;
        reg [24:0] rounded;
        begin
            if (sh == 5'd0) begin
                rshift_round_u24 = value;
            end else if (sh >= 5'd24) begin
                rshift_round_u24 = 24'd0;
            end else begin
                rounded = {1'b0, value} + (25'd1 << (sh - 5'd1));
                rshift_round_u24 = rounded >> sh;
            end
        end
    endfunction

    function [7:0] encode_unsigned_e4m3;
        input              is_zero;
        input              is_inf;
        input signed [8:0] exp_unb;
        input [23:0]       sig24;
        reg [4:0] sub_shift;
        reg signed [8:0] exp_r;
        reg [3:0] keep4;
        reg guard;
        reg sticky;
        reg round_up;
        reg [4:0] rounded4;
        reg [3:0] exp4;
        reg [23:0] sub_mag;
        reg [3:0] sub_mant;
        begin
            exp_r = exp_unb;

            if (is_zero || (sig24 == 24'd0) || (exp_r < -9'sd10)) begin
                encode_unsigned_e4m3 = 8'h00;
            end else if (is_inf || (exp_r > 9'sd7)) begin
                encode_unsigned_e4m3 = 8'h77;
            end else if (exp_r >= -9'sd6) begin
                keep4 = sig24[23:20];
                guard = sig24[19];
                sticky = |sig24[18:0];
                round_up = guard & (sticky | keep4[0]);
                rounded4 = {1'b0, keep4} + {4'd0, round_up};

                if (rounded4[4]) begin
                    exp_r = exp_r + 9'sd1;
                    rounded4 = 5'b01000;
                end

                if (exp_r > 9'sd7)
                    encode_unsigned_e4m3 = 8'h77;
                else begin
                    exp4 = exp_r + 9'sd7;
                    encode_unsigned_e4m3 = {1'b0, exp4, rounded4[2:0]};
                end
            end else begin
                case (exp_r)
                    -9'sd10: sub_shift = 5'd24;
                    -9'sd9:  sub_shift = 5'd23;
                    -9'sd8:  sub_shift = 5'd22;
                    default: sub_shift = 5'd21;
                endcase
                sub_mag = rshift_round_u24(sig24, sub_shift);
                sub_mant = sub_mag[3:0];

                if (sub_mant > 4'd7)
                    encode_unsigned_e4m3 = 8'h08;
                else
                    encode_unsigned_e4m3 = {1'b0, 4'b0000, sub_mant[2:0]};
            end
        end
    endfunction

    // -------------------- S0: unpack absolute FP32 magnitudes --------------------
    wire [30:0] abs_lane [0:15];

    generate
      for (gi=0; gi<16; gi=gi+1) begin: GEN_ABS_LANE
        assign abs_lane[gi] = fp32_bus[gi*32 + 30 : gi*32];
      end
    endgenerate

    // -------------------- S1-S4: pipelined 16-way max tree --------------------
    reg        v1;
    reg [30:0] max_s1 [0:7];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            for (ii=0; ii<8; ii=ii+1)
                max_s1[ii] <= 31'd0;
        end else begin
            v1 <= in_valid;
            for (ii=0; ii<8; ii=ii+1)
                max_s1[ii] <= max_abs2(abs_lane[2*ii], abs_lane[2*ii+1]);
        end
    end

    reg        v2;
    reg [30:0] max_s2 [0:3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0;
            for (ii=0; ii<4; ii=ii+1)
                max_s2[ii] <= 31'd0;
        end else begin
            v2 <= v1;
            for (ii=0; ii<4; ii=ii+1)
                max_s2[ii] <= max_abs2(max_s1[2*ii], max_s1[2*ii+1]);
        end
    end

    reg        v3;
    reg [30:0] max_s3 [0:1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v3 <= 1'b0;
            max_s3[0] <= 31'd0;
            max_s3[1] <= 31'd0;
        end else begin
            v3 <= v2;
            max_s3[0] <= max_abs2(max_s2[0], max_s2[1]);
            max_s3[1] <= max_abs2(max_s2[2], max_s2[3]);
        end
    end

    reg        v4;
    reg [30:0] group_max_s4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v4 <= 1'b0;
            group_max_s4 <= 31'd0;
        end else begin
            v4 <= v3;
            group_max_s4 <= max_abs2(max_s3[0], max_s3[1]);
        end
    end

    // -------------------- S5: group_max * [1 / E2M1_max] --------------------
    wire [7:0]  group_max_exp_s4  = group_max_s4[30:23];
    wire [22:0] group_max_frac_s4 = group_max_s4[22:0];
    wire [23:0] group_max_sig_s4  = (group_max_exp_s4 == 8'h00) ?
                                    {1'b0, group_max_frac_s4} :
                                    {1'b1, group_max_frac_s4};
    wire signed [8:0] group_max_exp_unb_s4 = (group_max_exp_s4 == 8'h00) ?
                                             -9'sd126 :
                                             ($signed({1'b0, group_max_exp_s4}) - 9'sd127);
    wire [36:0] scale_prod_s4 = group_max_sig_s4 * INV_E2M1_MAX_SIG_Q12;

    reg        v5;
    reg        scale_zero_s5;
    reg        scale_inf_s5;
    reg signed [8:0] max_exp_unb_s5;
    reg [36:0] scale_prod_s5;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v5 <= 1'b0;
            scale_zero_s5 <= 1'b0;
            scale_inf_s5 <= 1'b0;
            max_exp_unb_s5 <= 9'sd0;
            scale_prod_s5 <= 37'd0;
        end else begin
            v5 <= v4;
            scale_zero_s5 <= (group_max_s4 == 31'd0);
            scale_inf_s5 <= (group_max_exp_s4 == 8'hff);
            max_exp_unb_s5 <= group_max_exp_unb_s4;
            scale_prod_s5 <= scale_prod_s4;
        end
    end

    // -------------------- S6: normalize the scaled FP32 significand --------------------
    wire scale_prod_ge_2_s5 = scale_prod_s5[36];
    wire [23:0] scale_sig_s5 = scale_prod_ge_2_s5 ?
                               scale_prod_s5[36:13] :
                               scale_prod_s5[35:12];
    wire signed [8:0] scale_exp_unb_s5 = scale_prod_ge_2_s5 ?
                                         (max_exp_unb_s5 - 9'sd2) :
                                         (max_exp_unb_s5 - 9'sd3);

    reg        v6;
    reg        scale_zero_s6;
    reg        scale_inf_s6;
    reg signed [8:0] scale_exp_unb_s6;
    reg [23:0] scale_sig_s6;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v6 <= 1'b0;
            scale_zero_s6 <= 1'b0;
            scale_inf_s6 <= 1'b0;
            scale_exp_unb_s6 <= 9'sd0;
            scale_sig_s6 <= 24'd0;
        end else begin
            v6 <= v5;
            scale_zero_s6 <= scale_zero_s5;
            scale_inf_s6 <= scale_inf_s5;
            scale_exp_unb_s6 <= scale_exp_unb_s5;
            scale_sig_s6 <= scale_sig_s5;
        end
    end

    // -------------------- S7: quantize raw_scale to unsigned E4M3 --------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            group_scale_e4m3 <= 8'h00;
        end else begin
            out_valid <= v6;
            group_scale_e4m3 <= encode_unsigned_e4m3(
                scale_zero_s6,
                scale_inf_s6,
                scale_exp_unb_s6,
                scale_sig_s6
            );
        end
    end
endmodule
