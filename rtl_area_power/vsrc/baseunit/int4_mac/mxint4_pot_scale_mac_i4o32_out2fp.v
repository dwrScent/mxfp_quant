module mxint4_mac (
    input wire        clk,       // 时钟
    input wire        rst,       // 复位
    input wire [3:0]  a,         // INT4 输入 a
    input wire [3:0]  b,         // INT4 输入 b
    input wire [7:0]  scale_a,   // Scaling factor A (E8M0)
    input wire [7:0]  scale_b,   // Scaling factor B (E8M0)
    output reg [31:0] c_out,     // 最终结果 (FP32)
    output wire       error      // 错误标志
);

    reg signed [31:0] acc;          // INT32 累加器
    wire signed [7:0] mul_result;   // INT4 乘法结果
    wire [31:0] acc_fp32;           // INT32 转 FP32 结果
    wire [31:0] scaled_result;      // 缩放后的 FP32 结果
    wire        mul_error;          // 错误标志
    wire        scale_overflow;     // Scaling 溢出标志

    // **INT4 乘法结果**
    assign mul_result = a * b;

    // **INT4 MAC 操作**
    always @(posedge clk or posedge rst) begin
        if (rst)
            acc <= 32'b0; // 复位累加器
        else
            acc <= acc + mul_result;
    end

    // **INT32 转 FP32 转换**
    int32_to_fp32 u_int32_to_fp32 (
        .int_in(acc),
        .fp32_out(acc_fp32)
    );

    // **Scaling Factor 应用**
    wire [7:0] exponent_in  = acc_fp32[30:23];   // FP32 exponent
    wire [22:0] mantissa_in = acc_fp32[22:0];    // FP32 mantissa
    wire [7:0] exponent_sum = exponent_in + scale_a + scale_b;
    wire [7:0] exponent_out = (exponent_sum > 8'b11111111) ? 8'b11111111 : exponent_sum; // Clamp 到 FP32 范围

    assign scaled_result = {acc_fp32[31], exponent_out, mantissa_in};

    // **检测 Scaling Factor 溢出**
    assign scale_overflow = (exponent_sum > 8'b11111111);

    // **最终结果输出**
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_out <= 32'b0; // 复位
        else
            c_out <= scale_overflow ? 32'b0 : scaled_result;
    end

    // **错误标志**
    assign error = mul_error | scale_overflow;

endmodule

module int32_to_fp32 (
    input wire signed [31:0] int_in,
    output reg [31:0] fp32_out
);

    reg [7:0] exponent;
    reg [22:0] mantissa;
    reg sign;
    reg [31:0] abs_value;
    integer shift;

    always @(*) begin
        // 处理符号
        sign = int_in[31];
        abs_value = sign ? -int_in : int_in;

        // 找到最高位的 1
        shift = 0;
        while (abs_value[31] == 0 && shift < 32) begin
            abs_value = abs_value << 1;
            shift = shift + 1;
        end

        // 生成 FP32 格式的 exponent 和 mantissa
        exponent = 127 + (31 - shift);        // 偏置 127
        mantissa = abs_value[30:8];           // 取前 23 位

        // 组合 FP32 结果
        fp32_out = {sign, exponent, mantissa};
    end

endmodule