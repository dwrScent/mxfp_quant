module vsqint4_mac (
    input wire        clk,       // 时钟
    input wire        rst,       // 复位
    input wire [3:0]  a,         // INT4 输入 a
    input wire [3:0]  b,         // INT4 输入 b
    input wire [3:0]  scale_a,   // Scaling factor A (E0M4)
    input wire [3:0]  scale_b,   // Scaling factor B (E0M4)
    output reg [31:0] c_out,     // 最终结果 (FP32)
    output wire       error      // 错误标志
);

    reg signed [31:0] acc;          // INT32 累加器
    wire signed [7:0] mul_result;   // INT4 乘法结果
    wire signed [31:0] scaled_acc;  // 应用 Scaling Factor 后的累加器结果
    wire [31:0] fp32_out;           // INT32 转 FP32 结果
    wire scale_overflow;            // Scaling 溢出标志

    // **INT4 乘法结果**
    assign mul_result = $signed(a) * $signed(b);

    // **INT4 MAC 操作**
    always @(posedge clk or posedge rst) begin
        if (rst)
            acc <= 32'b0; // 复位累加器
        else
            acc <= acc + mul_result;
    end

    // **Scaling Factor 应用**
    // 定点数 Scaling: INT32 * INT4 * INT4
    assign scaled_acc = acc * $signed(scale_a) * $signed(scale_b);

    // **INT32 转 FP32 转换**
    int32_to_fp32 u_int32_to_fp32 (
        .int_in(scaled_acc),        // 经过 Scaling 的结果
        .fp32_out(fp32_out)         // 转换后的 FP32 结果
    );

    // **最终结果输出**
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_out <= 32'b0; // 复位
        else
            c_out <= scale_overflow ? 32'b0 : fp32_out;
    end

    // **错误标志**
    assign scale_overflow = 1'b0;  // 对于定点数 scaling，目前没有额外溢出检测
    assign error = scale_overflow;

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