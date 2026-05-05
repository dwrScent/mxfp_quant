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
    integer jj;

    always @(*) begin
        if (a[30:0] >= b[30:0]) begin
            big = a;
            small = b;
        end else begin
            big = b;
            small = a;
        end

        e_big = big[30:23];
        e_small = small[30:23];
        m_big = {e_big != 8'd0, big[22:0]};
        m_small = {e_small != 8'd0, small[22:0]};

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
            lead_shift = 5'd24;
            for (jj=23; jj>=0; jj=jj-1)
                if ((lead_shift == 5'd24) && diff[jj])
                    lead_shift = 23 - jj;

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
