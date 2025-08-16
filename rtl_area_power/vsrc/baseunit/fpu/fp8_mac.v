module fp8_mac (
	input clk,
	input rst,
	input  [7:0] a,
    input  [7:0] b,
    output reg [7:0] c,
	output error
);
	
	wire [7:0] c_tmp;
	wire [7:0] c_inner;
	always @(posedge clk) begin
		if(rst)
			c <= 0;
		else begin
			c <= c_tmp;
		end
	end

	fp8_mul u1(a,b,c_inner,error);
	fp8_adder u2(c_inner, c, c_tmp);
endmodule

module fp8_mul (
    input  [7:0] a,
    input  [7:0] b,
    output [7:0] c,
    output        error // valid in fp16 mode 
    );

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

module fp8_adder (
    input  [7:0] a,
    input  [7:0] b,
    output [7:0] c
    );

    wire [2:0] adder_input_1,adder_input_2,aligned_small,adder_output;
    wire if_sub,a_sign, b_sign, c_sign,c1;
    wire [7:0] normalized_out;

    wire [7:0] result;

    reg [6:0] bigger, smaller;
    reg a_larger_b;

    assign a_sign        = a[7];
    assign b_sign        = b[7];
    assign if_sub        = (a_sign == b_sign) ? 1'b0 : 1'b1;
    assign c_sign        = a_larger_b ? a_sign : b_sign;
    assign adder_input_1 = {1'b1,bigger[1:0]};
    assign adder_input_2 = (if_sub ? ~aligned_small + 1'b1 : aligned_small);
    assign c             = result; 

    //compare two number regardless sign
    always @(*) begin
        if (a[6:0] > b[6:0]) begin
            bigger = a[6:0];
            smaller = b[6:0];
            a_larger_b = 1'b1;
        end else begin 
            bigger = b[6:0];
            smaller = a[6:0];
            a_larger_b = 1'b0;
        end 
    end

    alignment u1(bigger,smaller,aligned_small); // TODO:

	assign {c1, adder_output} = adder_input_1 + adder_input_2;

    add_normalizer u4(c_sign,bigger[6:2],adder_output,result,c1,if_sub);

endmodule

module alignment (
	input  [6:0] bigger, 
	input  [6:0] smaller,
	output [2:0] aligned_small
	);

	wire c1;
	wire [4:0] bigger_exponent, smaller_exponent,shift_bits;

	assign bigger_exponent  = bigger  [6:2];
	assign smaller_exponent = smaller [6:2];
	assign aligned_small    = ({1'b1,smaller[1:0]} >> shift_bits);

	assign {c1, shift_bits} = bigger_exponent - smaller_exponent;
endmodule

module add_normalizer (
    input             sign,
    input      [ 4:0] exponent,
    input      [ 2:0] mantissa_add,
    output reg [ 7:0] result,
    input             if_carray,
    input             if_sub
    );

    reg [4:0] number_of_zero_lead;
    reg [2:0] norm_mantissa_add;

    wire [4:0] shift_left_exp; 
    wire c1;

    always @ (*) begin
        if (mantissa_add[2:0] == 3'b001) begin
            number_of_zero_lead = 5'd2;
            norm_mantissa_add   = (mantissa_add << 4'd2);
        end else if (mantissa_add[2:1] == 2'b01) begin
            number_of_zero_lead = 5'd1;
            norm_mantissa_add   = (mantissa_add << 4'd1);
        end else begin 
            number_of_zero_lead = 5'd0;
            norm_mantissa_add   = mantissa_add[2:0];
        end 
    end

    always @(*) begin
        result[7]        = sign;
        if (!if_sub) begin 
            result[6:2] = if_carray ? exponent + 1'b1 : exponent;
            result[1:0]   = if_carray ? mantissa_add[2:1] : mantissa_add[1:0];
        end else begin 
            result[6:2] = shift_left_exp;
            result[1:0]   = norm_mantissa_add[1:0];
        end 
    end

	assign {c1, shift_left_exp} = exponent - number_of_zero_lead;
endmodule
