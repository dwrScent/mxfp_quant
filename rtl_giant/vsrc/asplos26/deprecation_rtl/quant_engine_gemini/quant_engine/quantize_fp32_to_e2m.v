// =============================================================
// FP32 -> E2M* Quantizer (Verilog-2001 Compliant)
// Parameters:
//   M = mantissa bits (1 for FP4(E2M1), 3 for FP6(E2M3))
// Code layout (width = 1+2+M): {sign, e2[1:0], m[M-1:0]}
// =============================================================
module quantize_fp32_to_e2m
#( parameter M = 1 )
(
    input      [31:0]        x_fp32,   // IEEE754 single
    input signed [7:0]       k,        // shared_scale
    output reg [ (1+2+M)-1:0 ] y_code  // packed code
);

    // 解析 FP32
    wire s          = x_fp32[31];
    wire [7:0] e8   = x_fp32[30:23];
    wire [22:0] f23 = x_fp32[22:0];

    // 特殊值
    wire is_zero_sub = (e8 == 8'h00);       // 含次正规
    wire is_inf_nan  = (e8 == 8'hFF);

    // 目标：e_adj_unb = (e_unb - k)
    // e_unb = e8 - 127
    wire signed [9:0] e_unb       = $signed({1'b0,e8}) - 127;
    wire signed [9:0] e_adj_unb   = e_unb - k; // 可能为负

    // 取 M 位mantissa + RNE
    // [SV_FIX] Changed SystemVerilog part-select `[22 -: M]` to Verilog-2001 standard `[22 : 22-M+1]`
    wire [M-1:0] mant_M_pre = (M>0) ? f23[22 : 22-M+1] : 1'b0;
    wire         guard      = (M<=22) ? f23[22-M] : 1'b0;
    wire         sticky     = (M<22)  ? (|f23[22-M-1:0]) : 1'b0;

    wire         lsb        = (M>0) ? mant_M_pre[0] : 1'b0;
    wire         rnd_inc    = guard & (sticky | lsb);

    wire [M:0]   mant_round_sum = {1'b0, mant_M_pre} + { {M{1'b0}}, rnd_inc };
    wire         mant_carry     = (M>0) ? mant_round_sum[M] : rnd_inc;
    wire [M-1:0] mant_M_rnd     = (M>0) ? mant_round_sum[M-1:0] : {M{1'b0}};

    // 组合主逻辑
    reg [1:0] e2_st;
    reg [M-1:0] m_out;
    
    // [FIX] Moved variable declaration from inside always block to the module scope
    reg signed [9:0] e2_tmp; // Note: increased width to signed to handle intermediate values correctly

    always @(*) begin
        // 默认：±0
        y_code = {(1+2+M){1'b0}};
        y_code[(1+2+M)-1] = s; // 保留符号

        if (is_inf_nan) begin
            // 饱和到 max：exp=2(->st=3), mant 全 1
            e2_st = 2'd3;
            m_out = {M{1'b1}};
            y_code = {s, e2_st, m_out};
        end
        else if (is_zero_sub) begin
            // ±0
            e2_st = 2'd0; m_out = {M{1'b0}};
            y_code = {s, e2_st, m_out};
        end
        else if (e_adj_unb < -1) begin
            // 太小 -> ±0
            // Note: RNE rounding might round up a value like -1.4 to -1.0, making it representable.
            // For simplicity, we keep the original logic of clamping anything less than -1.
            e2_st = 2'd0; m_out = {M{1'b0}};
            y_code = {s, e2_st, m_out};
        end
        else if (e_adj_unb > 2) begin
            // 太大 -> 饱和
            e2_st = 2'd3; m_out = {M{1'b1}};
            y_code = {s, e2_st, m_out};
        end
        else begin
            // 正常范围：e2_st = e_adj_unb + bias(=1)
            e2_tmp = e_adj_unb + 1; // Range: 0..3

            // mant rounding 溢出会进位到指数
            if (mant_carry) begin
                // 1.111.. round -> 10.000.. => exp+1, mant=0
                if (e2_tmp == 3) begin
                    // 再进位会超范围 -> 饱和
                    e2_st = 2'd3;
                    m_out = {M{1'b1}};
                end else begin
                    e2_st = e2_tmp[1:0] + 1;
                    m_out = {M{1'b0}};
                end
            end else begin
                e2_st = e2_tmp[1:0];
                m_out = mant_M_rnd;
            end
            y_code = {s, e2_st, m_out};
        end
    end
endmodule
