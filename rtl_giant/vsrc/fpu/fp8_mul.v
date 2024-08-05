module fp8_mul (
    input clk,
    input rst,
    input  [7:0] a,
    input  [7:0] b,
    output [7:0] c_reg,
    output        error // valid in fp16 mode 
    );

    wire [7:0] c;
    always @(a or b) begin
        if (rst)
            c_reg <= 8'b0;
        else
            c_reg <= c;
    end

    wire [7:0] c_tmp;
    wire        c_sign,a_zero,b_zero;
    wire [4:0] sum_exponent, biased_sum_exponent; 
    wire [3:0] multiplier_input1,multiplier_input2; 

    wire [7:0] multiplier_output; 
    wire [6:0] normalized_out; 
    wire [5:0] mantissa_prod; 
    wire c1,c2,underflow,overflow;

    assign overflow = (c1 && c2 && ~biased_sum_exponent[4]) ? 1'b1 :1'b0; 
    assign underflow = (~c1 && ~c2 && biased_sum_exponent[4]) ? 1'b1:1'b0; 

    assign a_zero = ~(|a);
    assign b_zero = ~(|b);
    assign c_sign = a[7] ^ b[7];
    assign multiplier_input1 = {1'b0,1'b1,a[1:0]};
    assign multiplier_input2 = {1'b0,1'b1,b[1:0]};
    
    assign c = ((a_zero | b_zero) ? 8'b0 : c_tmp);
    //error detect
    assign c_tmp = (~error) ? {c_sign,normalized_out} : (underflow ? {c_sign,7'b0000000} : {c_sign,5'b1111_1,2'b00});
    
    assign error = overflow | underflow; 

    assign mantissa_prod = multiplier_output[5:0];
    assign multiplier_output = multiplier_input1 * multiplier_input2;
    
    assign {c1,sum_exponent} = a[6:2] + b[6:2];
    assign {c2,biased_sum_exponent} = sum_exponent + 5'b10001; //minux bias

    mul_normalizer u4(biased_sum_exponent,mantissa_prod,normalized_out);
endmodule

module mul_normalizer (
	input  [4:0] exponent,
	input  [5:0] mantissa_prod,
	output [6:0] result
);

	wire [4:0] result_exponent;
	wire [1:0] result_mantissa;

	assign result_exponent = (mantissa_prod[5]) ? (exponent + 1'b1): exponent;
	assign result_mantissa = (mantissa_prod[5]) ? mantissa_prod[4:3]:mantissa_prod[3:2];
	assign result 		   = {result_exponent,result_mantissa};

// No rounding and No overflow/underflow detection

endmodule