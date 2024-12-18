module int4_mac_acc32bit (
    input        clk,       // 时钟
    input        rst,       // 复位
    input  [3:0] a,         // 带符号的 INT4 输入 a
    input  [3:0] b,         // 带符号的 INT4 输入 b
    output reg [31:0] c     // 32位累加结果
);

    // 将 4位带符号的 a 和 b 转换为 32位带符号整数
    wire signed [3:0] a_signed = a;
    wire signed [3:0] b_signed = b;
    wire signed [31:0] product = a_signed * b_signed; // 乘法结果（32位带符号）

    // 累加逻辑
    always @(posedge clk or posedge rst) begin
        if (rst)
            c <= 32'b0; // 复位累加器
        else
            c <= c + product; // 累加带符号的乘法结果
    end

endmodule