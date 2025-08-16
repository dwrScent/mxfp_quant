module mxint4_mac (
    input wire        clk,       // 时钟
    input wire        rst,       // 复位
    input wire [3:0]  a,         // INT4 输入 a
    input wire [3:0]  b,         // INT4 输入 b
    input wire [7:0]  scale_a,   // Scaling factor A (E8M0)
    input wire [7:0]  scale_b,   // Scaling factor B (E8M0)
    output reg [31:0] c_out      // 最终结果 (INT32)
);

    reg signed [31:0] acc;          // INT32 累加器
    wire signed [3:0] a_signed = a; // 手动声明为有符号的 INT4 输入 a
    wire signed [3:0] b_signed = b; // 手动声明为有符号的 INT4 输入 b
    wire signed [7:0] scale_a_signed = scale_a; // 手动声明为有符号的 INT8 输入 scale_a
    wire signed [7:0] scale_b_signed = scale_b; // 手动声明为有符号的 INT8 输入 scale_b
    wire signed [7:0] mul_result;   // INT4 乘法结果
    wire signed [31:0] scaled_result; // 缩放后的 INT32 结果

    // **INT4 乘法结果**
    assign mul_result = a_signed * b_signed; // INT4 乘法结果

    // **INT4 MAC 操作**
    always @(posedge clk or posedge rst) begin
        if (rst)
            acc <= 32'b0; // 复位累加器
        else
            acc <= acc + mul_result; // 累加操作
    end

    // **Scaling Factor 应用**
    // 定点数 Scaling: INT32 * INT8 * INT8
    assign scaled_result = acc * scale_a_signed * scale_b_signed;

    // **最终结果输出**
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_out <= 32'b0; // 复位
        else
            c_out <= scaled_result; // 输出缩放后的结果
    end

endmodule