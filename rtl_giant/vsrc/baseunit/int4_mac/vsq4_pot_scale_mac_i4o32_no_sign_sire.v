module vsqint4_mac (
    input wire        clk,       // 时钟
    input wire        rst,       // 复位
    input wire [3:0]  a,         // INT4 输入 a
    input wire [3:0]  b,         // INT4 输入 b
    input wire [3:0]  scale_a,   // Scaling factor A (E0M4)
    input wire [3:0]  scale_b,   // Scaling factor B (E0M4)
    output reg [31:0] c_out     // 最终结果 (INT32)
);

    reg signed [31:0] acc;          // INT32 累加器
    wire signed [7:0] mul_result;   // INT4 乘法结果
    wire signed [31:0] scaled_acc;  // 应用 Scaling Factor 后的累加器结果

    // **INT4 乘法结果**
    assign mul_result = a * b; // INT4 乘法结果

    // **INT4 MAC 操作**
    always @(posedge clk or posedge rst) begin
        if (rst)
            acc <= 32'b0; // 复位累加器
        else
            acc <= acc + mul_result; // 累加操作
    end

    // **Scaling Factor 应用**
    // 定点数 Scaling: INT32 * INT4 * INT4
    assign scaled_acc = acc * scale_a * scale_b;

    // **最终结果输出**
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_out <= 32'b0; // 复位
        else
            c_out <= scaled_acc; // 如果溢出，输出清零
    end

endmodule