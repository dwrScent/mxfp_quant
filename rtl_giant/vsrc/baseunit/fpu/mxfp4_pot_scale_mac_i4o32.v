module mxfp4_mac (
    input  wire        clk,        // 时钟
    input  wire        rst,        // 复位
    input  wire [3:0]  a,          // FP4 输入 a
    input  wire [3:0]  b,          // FP4 输入 b
    input  wire [7:0]  scale_a,    // Scaling factor A (FP8)
    input  wire [7:0]  scale_b,    // Scaling factor B (FP8)
    output reg  [31:0] c_out,      // 最终结果 (FP32)
    output wire        error       // 错误标志
);

    reg  [31:0] c_acc;             // 累加器状态 (FP32)
    wire [31:0] mul_result;        // FP4 -> FP32 乘法结果
    wire [31:0] c_tmp;             // 中间累加结果
    wire [31:0] scaled_result;     // 缩放后的结果
    wire        mul_error;         // 乘法错误标志
    wire        scale_overflow;    // Scaling Factor 溢出标志

    // **累加器逻辑**
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_acc <= 32'b0; // 复位累加器
        else
            c_acc <= c_tmp; // 更新累加结果
    end

    // **FP4 乘法器实例化**
    fp4_mul u_mul (
        .a(a),
        .b(b),
        .c(mul_result),
        .error(mul_error)
    );

    // **FP32 加法器实例化**
    fp32_adder u_adder (
        .a(c_acc),
        .b(mul_result),
        .c(c_tmp)
    );

    // **Scaling Factor 应用**
    wire [7:0] exponent_in  = c_tmp[30:23];
    wire [22:0] mantissa_in = c_tmp[22:0];
    wire [7:0] exponent_sum = exponent_in + scale_a[7:0] + scale_b[7:0];
    wire [7:0] exponent_out = (exponent_sum > 8'b11111111) ? 8'b11111111 : exponent_sum;
    assign scaled_result    = {c_tmp[31], exponent_out, mantissa_in};

    // **检测 Scaling Factor 溢出**
    assign scale_overflow = (exponent_sum > 8'b11111111);

    // **最终结果输出**
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_out <= 32'b0;
        else
            c_out <= scale_overflow ? 32'b0 : scaled_result;
    end

    // **错误标志**
    assign error = mul_error | scale_overflow;

endmodule

module fp4_mul (
    input  wire [3:0]  a,          // FP4 输入 a
    input  wire [3:0]  b,          // FP4 输入 b
    output wire [31:0] c,          // FP32 乘法结果
    output wire        error       // 溢出/下溢标志
);

    wire a_sign, b_sign, c_sign;
    wire [1:0] a_exp, b_exp;
    wire [1:0] a_mant, b_mant;      // FP4 尾数，包含隐含位
    wire [3:0] mant_product;        // 尾数乘积
    wire [9:0] exp_sum;
    wire underflow, overflow;

    // 提取 FP4 的符号位、指数位、尾数位
    assign a_sign = a[3];
    assign b_sign = b[3];
    assign a_exp  = a[2:1];
    assign b_exp  = b[2:1];
    assign a_mant = {1'b1, a[0]}; // 扩展尾数，隐含位为 1
    assign b_mant = {1'b1, b[0]}; // 扩展尾数，隐含位为 1

    // 计算结果的符号位和尾数位
    assign c_sign = a_sign ^ b_sign;
    assign mant_product = a_mant * b_mant; // 尾数乘积为 4 位

    // 指数相加，考虑偏置（Bias = 127 for FP32, Bias = 1 for FP4）
    assign exp_sum = a_exp + b_exp - 10'd1 + 10'd127; // Adjust bias

    // 检测溢出和下溢
    assign overflow = (exp_sum > 10'd254);  // 超出 FP32 指数范围
    assign underflow = (exp_sum < 10'd0);   // 小于 FP32 最小指数

    // 规范化输出，扩展尾数为 23 位
    assign c = overflow ? {c_sign, 8'b11111111, 23'b0} :             // 溢出
               underflow ? {c_sign, 8'b00000000, 23'b0} :            // 下溢
               {c_sign, exp_sum[7:0], {mant_product, 19'b0}};        // 正常输出

    assign error = overflow | underflow;

endmodule

module fp32_adder (
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] c
);

    wire [22:0] adder_input_1, adder_input_2, aligned_small, adder_output;
    wire if_sub, a_sign, b_sign, c_sign, c1;
    wire [31:0] result;

    reg [30:0] bigger, smaller;
    reg a_larger_b;

    assign a_sign = a[31];
    assign b_sign = b[31];
    assign if_sub = (a_sign == b_sign) ? 1'b0 : 1'b1;
    assign c_sign = a_larger_b ? a_sign : b_sign;

    assign adder_input_1 = {1'b1, bigger[22:0]};
    assign adder_input_2 = (if_sub ? ~aligned_small + 1'b1 : aligned_small);
    assign c = result;

    // 比较两数大小（忽略符号位）
    always @(*) begin
        if (a[30:0] > b[30:0]) begin
            bigger = a[30:0];
            smaller = b[30:0];
            a_larger_b = 1'b1;
        end else begin
            bigger = b[30:0];
            smaller = a[30:0];
            a_larger_b = 1'b0;
        end
    end

    // 对齐较小数的尾数
    alignment u_align(bigger, smaller, aligned_small);

    assign {c1, adder_output} = adder_input_1 + adder_input_2;

    // 规范化加法结果
    add_normalizer u_norm(c_sign, bigger[30:23], adder_output, result, c1, if_sub);

endmodule

module alignment (
    input  [30:0] bigger,
    input  [30:0] smaller,
    output [22:0] aligned_small
);

    wire c1;
    wire [7:0] bigger_exponent, smaller_exponent, shift_bits;

    assign bigger_exponent  = bigger[30:23];
    assign smaller_exponent = smaller[30:23];
    assign aligned_small    = ({1'b1, smaller[22:0]} >> shift_bits);

    assign {c1, shift_bits} = bigger_exponent - smaller_exponent;

endmodule

module add_normalizer (
    input             sign,
    input      [7:0]  exponent,
    input      [22:0] mantissa_add,
    output reg [31:0] result,
    input             if_carry,
    input             if_sub
);

    reg [7:0] number_of_zero_lead;
    reg [22:0] norm_mantissa_add;
    wire [7:0] shift_left_exp;
    wire c1;

    always @(*) begin
        if (mantissa_add[22:18] == 5'b00001) begin
            number_of_zero_lead = 8'd5;
            norm_mantissa_add   = mantissa_add << 5;
        end else if (mantissa_add[22:19] == 4'b0001) begin
            number_of_zero_lead = 8'd4;
            norm_mantissa_add   = mantissa_add << 4;
        end else if (mantissa_add[22:20] == 3'b001) begin
            number_of_zero_lead = 8'd3;
            norm_mantissa_add   = mantissa_add << 3;
        end else begin
            number_of_zero_lead = 8'd0;
            norm_mantissa_add   = mantissa_add;
        end
    end

    always @(*) begin
        result[31]       = sign;
        if (!if_sub) begin
            result[30:23] = if_carry ? exponent + 1 : exponent;
            result[22:0]  = if_carry ? mantissa_add[22:1] : mantissa_add;
        end else begin
            result[30:23] = shift_left_exp;
            result[22:0]  = norm_mantissa_add;
        end
    end

    assign {c1, shift_left_exp} = exponent - number_of_zero_lead;

endmodule
