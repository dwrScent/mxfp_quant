module mxint4_mac (
    input wire        clk,       // 时钟
    input wire        rst,       // 复位
    input wire [3:0]  a,         // INT4 输入 a
    input wire [3:0]  b,         // INT4 输入 b
    input wire [7:0]  scale_a,   // Scaling factor A (E8M0)
    input wire [7:0]  scale_b,   // Scaling factor B (E8M0)
    output reg [31:0] c_out,     // 最终结果 (INT32)
    output wire       error      // 错误标志
);

    reg signed [31:0] acc;          // INT32 累加器
    wire signed [7:0] mul_result;   // INT4 乘法结果
    wire signed [31:0] scaled_result; // 缩放后的 INT32 结果
    wire overflow;            // 溢出标志

    // **INT4 乘法结果**
    assign mul_result = $signed(a) * $signed(b); // INT4 乘法结果

    // **INT4 MAC 操作**
    always @(posedge clk or posedge rst) begin
        if (rst)
            acc <= 32'b0; // 复位累加器
        else
            acc <= acc + mul_result; // 累加操作
    end

    // **Scaling Factor 应用**
    // 定点数 Scaling: INT32 * INT8 * INT8
    assign scaled_result = acc * $signed(scale_a) * $signed(scale_b);

    // **检测 Scaling Factor 溢出**
    assign overflow = (scaled_result > 32'sh7FFFFFFF || scaled_result < -32'sh80000000);

    // **最终结果输出**
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_out <= 32'b0; // 复位
        else
            c_out <= overflow ? 32'b0 : scaled_result; // 如果溢出，输出清零
    end

    // **错误标志**
    assign error = overflow;

endmodule