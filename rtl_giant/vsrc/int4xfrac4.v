module fixed_point_multiplier(
    input [3:0] a,
    input [3:0] b,
    output reg [3:0] c
);

    // 临时变量，用于存储乘积结果，因为需要处理8位的结果（4位整数*4位小数）
    reg [7:0] temp_product;

    always @(a or b) begin
        // a和b相乘的结果，需要将b看作是[0,1)区间内的数，扩大16倍处理
        temp_product = a * b;
        
        // 判断小数部分最高位（即temp_product的第四位，从0开始计数）
        // 如果该位为1，则结果需要加1进行舍入
        if (temp_product[3]) begin
            c = temp_product[7:4] + 1;
        end else begin
            c = temp_product[7:4]; // 直接取整数部分
        end
    end

endmodule