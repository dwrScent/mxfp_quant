// Signed fixed-point (W bits, FRAC fractional) -> IEEE-754 single (RNE)
// Verilog-2001 friendly: no SV, no inline decls, no slice-on-expression.
module fxp_to_fp32
#( parameter W=32, parameter FRAC=2 )
(
    input  signed [W-1:0] din,  // value = din / 2^FRAC
    output [31:0]         dout
);
    localparam EB=8, MB=23, BIAS=127;

    // ---- signals / temps (声明放模块作用域) ----
    wire        s;
    wire [W-1:0] abs_in;
    wire        is_zero;

    integer     k;
    integer     msb;

    integer     exp_unb;
    reg  [7:0]  E;
    reg  [22:0] F;
    reg  [31:0] dout_r;

    integer     shift;
    reg  [W-1:0] tmp_shifted;
    reg  [22:0] frac_pre;
    reg         guard, roundb, sticky;
    reg  [24:0] m_norm;
    reg  [7:0]  e_norm;
    reg         add_round;
    reg         carry_norm;

    integer     rshift;
    reg  [W+24:0] tmp_den;

    // ---- 基本量 ----
    assign s      = din[W-1];
    assign abs_in = s ? (~din + 1'b1) : din;
    assign is_zero= (abs_in == {W{1'b0}});

    // ---- 找 MSB ----
    always @(*) begin : FIND_MSB
        msb = -1;
        for (k=W-1; k>=0; k=k-1) begin
            if (abs_in[k]) begin
                msb = k;
                disable FIND_MSB;
            end
        end
    end

    // ---- 非偏指数 ----
    always @(*) begin
        if (is_zero) exp_unb = 0;
        else         exp_unb = msb - FRAC;
    end

    // ---- 主逻辑 ----
    always @(*) begin
        if (is_zero) begin
            dout_r = 32'h00000000;
        end else if (exp_unb > 127) begin
            // 溢出 -> Inf
            dout_r = {s, 8'hFF, 23'h0};
        end else if (exp_unb >= -126) begin
            // 正规数路径
            shift = msb - 23;             // 右移得到 [1].[22:0]
            if (shift >= 0) begin
                tmp_shifted = abs_in >> shift;
                frac_pre    = tmp_shifted[22:0];
                guard       = (shift>0) ? abs_in[shift-1] : 1'b0;
                roundb      = (shift>1) ? abs_in[shift-2] : 1'b0;
                sticky      = (shift>2) ? |abs_in[shift-3:0] : 1'b0;
            end else begin
                tmp_shifted = abs_in << (-shift);
                frac_pre    = tmp_shifted[22:0];
                guard=1'b0; roundb=1'b0; sticky=1'b0;
            end

            add_round = guard & (roundb | sticky | frac_pre[0]);
            {carry_norm, m_norm} = {1'b0, 1'b1, frac_pre} + add_round; // 1.int + frac

            e_norm = exp_unb + BIAS + carry_norm;
            if (e_norm==8'hFF) begin
                dout_r = {s, 8'hFF, 23'h0};
            end else begin
                E = e_norm;
                F = m_norm[22:0];
                dout_r = {s, E, F};
            end
        end else begin
            // 次正规（简化）
            rshift = (-126 - exp_unb) + (msb - 23);
            if (rshift < 0) rshift = 0;
            tmp_den = {abs_in, 25'b0} >> rshift;  // 把点移到 E=0 区
            frac_pre = tmp_den[24:2];
            guard    = tmp_den[1];
            roundb   = tmp_den[0];
            sticky   = 1'b0;

            add_round = guard & (roundb | sticky | frac_pre[0]);
            {carry_norm, m_norm} = {1'b0, 1'b0, frac_pre} + add_round; // 无隐藏位
            dout_r = {s, 8'h00, m_norm[22:0]};
        end
    end

    assign dout = dout_r;
endmodule
