module int_fp_mul (
    input clk,
    input rst,

    input  [15:0] a,
    input  [15:0] b,
    output reg [15:0] c_reg,
    output        error // valid in fp16 mode 
    );

    wire [15:0] c;
    always @(posedge clk) begin
        if(rst)
            c_reg <= 0;
        else begin
            c_reg <= c;
        end
    end

    wire [15:0] c_tmp;
    wire        c_sign,a_zero,b_zero;
    wire [4:0] sum_exponent, biased_sum_exponent;
    wire [15:0] multiplier_input1,multiplier_input2;

    wire [31:0] multiplier_output;
    wire [14:0] normalized_out;
    wire [21:0] mantissa_prod;
    wire c1,c2,underflow,overflow;

    assign overflow = (c1 && c2 && ~biased_sum_exponent[4]) ? 1'b1 :1'b0;
    assign underflow = (~c1 && ~c2 && biased_sum_exponent[4]) ? 1'b1:1'b0;

    assign a_zero = ~(|a);
    assign b_zero = ~(|b);
    assign c_sign = a[15] ^ b[15];
    assign multiplier_input1 = {5'b0,1'b1,a[9:0]};
    assign multiplier_input2 = {5'b0,1'b1,b[9:0]};
    
    assign c = ((a_zero | b_zero) ? 16'b0 : c_tmp);
    //error detect
    assign c_tmp = (~error) ? {c_sign,normalized_out} : (underflow ? {c_sign,15'b0000_0000_0000_000} : {c_sign,5'b1111_1,10'b0000_0000_00});
    
    assign error = overflow | underflow; 

    assign mantissa_prod = multiplier_output[21:0];
    assign multiplier_output = multiplier_input1 * multiplier_input2;
    
    assign {c1, sum_exponent} = a[14:10] + b[14:10];
    assign {c2, biased_sum_exponent} = sum_exponent + 5'b10001; //minux bias

    mul_normalizer u4(biased_sum_exponent,mantissa_prod,normalized_out);

endmodule
module mul_normalizer (
	input  [ 4:0] exponent,
	input  [21:0] mantissa_prod,
	output [14:0] result
);

	wire [4:0] result_exponent;
	wire [9:0] result_mantissa;

	assign result_exponent = (mantissa_prod[21]) ? (exponent + 1'b1): exponent;
	assign result_mantissa = (mantissa_prod[21]) ? mantissa_prod[20:11]:mantissa_prod[19:10];
	assign result 		   = {result_exponent,result_mantissa};

// No rounding and No overflow/underflow detection

endmodule


