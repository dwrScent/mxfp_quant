module fxp_to_fp32
#( parameter W=32, parameter FRAC=2 )
(
    input  signed [W-1:0] din,  // value = din / 2^FRAC
    output [31:0]         dout
);
    localparam EB=8, MB=23, BIAS=127;

    // -------- helpers --------
    // 低 n 位掩码（n 可变，Verilog-2001 可综合）
    function [W-1:0] low_mask;
        input integer n;
        integer j;
        begin
            low_mask = {W{1'b0}};
            for (j=0; j<W; j=j+1)
                if (j < n) low_mask[j] = 1'b1;
        end
    endfunction
    // -------------------------

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

    assign s      = din[W-1];
    assign abs_in = s ? (~din + 1'b1) : din;
    assign is_zero= (abs_in == {W{1'b0}});

    // 找到 MSB（去掉 disable，用优先赋值方式）
    always @(*) begin
        msb = -1;
        for (k=W-1; k>=0; k=k-1) begin
            if ((msb == -1) && abs_in[k])
                msb = k;
        end
    end

    always @(*) begin
        if (is_zero) exp_unb = 0;
        else         exp_unb = msb - FRAC;
    end

    always @(*) begin
        if (is_zero) begin
            dout_r = 32'h00000000;

        end else if (exp_unb > 127) begin
            // +Inf / -Inf
            dout_r = {s, 8'hFF, 23'h0};

        end else if (exp_unb >= -126) begin
            // 规格化数
            shift = msb - 23;

            if (shift >= 0) begin
                // 右移对齐到 1.xxx * 2^(exp_unb)
                tmp_shifted = abs_in >> shift;
                frac_pre    = tmp_shifted[22:0];

                // 下面三项不用可变切片：
                // guard  = bit (shift-1)
                // roundb = bit (shift-2)
                // sticky = OR of bits [shift-3:0]
                guard  = (shift>0) ? (((abs_in >> (shift-1)) & {{(W-1){1'b0}},1'b1}) != {W{1'b0}}) : 1'b0;
                roundb = (shift>1) ? (((abs_in >> (shift-2)) & {{(W-1){1'b0}},1'b1}) != {W{1'b0}}) : 1'b0;
                sticky = (shift>2) ? (|(abs_in & low_mask(shift-2))) : 1'b0;

            end else begin
                // 左移（无需取舍，GRS=0）
                tmp_shifted = abs_in << (-shift);
                frac_pre    = tmp_shifted[22:0];
                guard=1'b0; roundb=1'b0; sticky=1'b0;
            end

            add_round = guard & (roundb | sticky | frac_pre[0]);
            {carry_norm, m_norm} = {1'b0, 1'b1, frac_pre} + add_round;

            e_norm = exp_unb + BIAS + carry_norm;
            if (e_norm==8'hFF) begin
                dout_r = {s, 8'hFF, 23'h0};
            end else begin
                E = e_norm;
                F = m_norm[22:0];
                dout_r = {s, E, F};
            end

        end else begin
            // 非规格化数（exp = 0，暗位为 0）
            // 先将 abs_in<<25 与对齐右移 rshift，得到 [0].[23位分数]+GR
            rshift = (-126 - exp_unb) + (msb - 23);
            if (rshift < 0) rshift = 0;

            tmp_den  = {abs_in, 25'b0} >> rshift;
            frac_pre = tmp_den[24:2];
            guard    = tmp_den[1];
            roundb   = tmp_den[0];
            sticky   = 1'b0;

            add_round = guard & (roundb | sticky | frac_pre[0]);
            {carry_norm, m_norm} = {1'b0, 1'b0, frac_pre} + add_round; // 非规无隐藏位
            dout_r = {s, 8'h00, m_norm[22:0]};
        end
    end

    assign dout = dout_r;
endmodule
