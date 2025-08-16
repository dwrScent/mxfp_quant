// 仅比较 exponent(8b, biased) 的最大值；忽略符号/尾数
module max_exp (
    input  [7:0] exp_bv [0:31],
    output reg [7:0] max_exp_biased
);
    integer i;
    always @* begin
        max_exp_biased = 8'd0;
        for (i=0; i<32; i=i+1) begin
            if (exp_bv[i] > max_exp_biased)
                max_exp_biased = exp_bv[i];
        end
    end
endmodule
