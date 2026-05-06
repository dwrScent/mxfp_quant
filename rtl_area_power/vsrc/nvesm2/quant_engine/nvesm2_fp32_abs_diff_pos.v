module nvesm2_fp32_abs_diff_pos (
    input  [31:0] a,
    input  [31:0] b,
    output reg [31:0] z
);
    reg [31:0] big;
    reg [31:0] small;
    reg [7:0]  e_big;
    reg [7:0]  e_small;
    reg [23:0] m_big;
    reg [23:0] m_small;
    reg [23:0] m_small_shifted;
    reg [24:0] diff;
    reg signed [10:0] exp_unb;
    reg signed [10:0] exp_out;
    reg [7:0]  exp_bits;
    reg [23:0] norm_m;
    reg [4:0]  lead_shift;

    function [4:0] leading_shift24;
        input [23:0] value;
        begin
            casez (value)
                24'b1???????????????????????: leading_shift24 = 5'd0;
                24'b01??????????????????????: leading_shift24 = 5'd1;
                24'b001?????????????????????: leading_shift24 = 5'd2;
                24'b0001????????????????????: leading_shift24 = 5'd3;
                24'b00001???????????????????: leading_shift24 = 5'd4;
                24'b000001??????????????????: leading_shift24 = 5'd5;
                24'b0000001?????????????????: leading_shift24 = 5'd6;
                24'b00000001????????????????: leading_shift24 = 5'd7;
                24'b000000001???????????????: leading_shift24 = 5'd8;
                24'b0000000001??????????????: leading_shift24 = 5'd9;
                24'b00000000001?????????????: leading_shift24 = 5'd10;
                24'b000000000001????????????: leading_shift24 = 5'd11;
                24'b0000000000001???????????: leading_shift24 = 5'd12;
                24'b00000000000001??????????: leading_shift24 = 5'd13;
                24'b000000000000001?????????: leading_shift24 = 5'd14;
                24'b0000000000000001????????: leading_shift24 = 5'd15;
                24'b00000000000000001???????: leading_shift24 = 5'd16;
                24'b000000000000000001??????: leading_shift24 = 5'd17;
                24'b0000000000000000001?????: leading_shift24 = 5'd18;
                24'b00000000000000000001????: leading_shift24 = 5'd19;
                24'b000000000000000000001???: leading_shift24 = 5'd20;
                24'b0000000000000000000001??: leading_shift24 = 5'd21;
                24'b00000000000000000000001?: leading_shift24 = 5'd22;
                24'b000000000000000000000001: leading_shift24 = 5'd23;
                default:                       leading_shift24 = 5'd24;
            endcase
        end
    endfunction

    always @(*) begin
        m_small_shifted = 24'd0;
        diff = 25'd0;
        exp_unb = 11'sd0;
        exp_out = 11'sd0;
        exp_bits = 8'd0;
        norm_m = 24'd0;
        lead_shift = 5'd0;
        z = 32'h00000000;

        if (a[30:0] >= b[30:0]) begin
            big = a;
            small = b;
        end else begin
            big = b;
            small = a;
        end

        e_big = big[30:23];
        e_small = small[30:23];
        m_big = {((e_big != 8'd0) ? 1'b1 : 1'b0), big[22:0]};
        m_small = {((e_small != 8'd0) ? 1'b1 : 1'b0), small[22:0]};

        if (e_big == 8'hff) begin
            z = 32'h7f800000;
        end else if (big[30:0] == small[30:0]) begin
            z = 32'h00000000;
        end else begin
            if ((e_big - e_small) >= 8'd24)
                m_small_shifted = 24'd0;
            else
                m_small_shifted = m_small >> (e_big - e_small);

            diff = {1'b0, m_big} - {1'b0, m_small_shifted};
            lead_shift = leading_shift24(diff[23:0]);

            exp_unb = (e_big == 8'd0) ? -11'sd126 : ($signed({1'b0, e_big}) - 11'sd127);
            exp_out = exp_unb - $signed({6'd0, lead_shift});
            exp_bits = exp_out + 11'sd127;
            norm_m = diff[23:0] << lead_shift;

            if (diff == 25'd0)
                z = 32'h00000000;
            else if (exp_out < -11'sd126)
                z = 32'h00000000;
            else
                z = {1'b0, exp_bits, norm_m[22:0]};
        end
    end
endmodule
