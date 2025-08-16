module fp16_adder (
	input clk,
	input rst,
    input  [15:0] a,
    input  [15:0] b,
    output reg [15:0] c_reg
);
	wire [15:0] c;
	always @(posedge clk) begin
		if (rst)
			c_reg <= 0;
		else
			c_reg <= c;
	end

    wire [10:0] adder_input_1,adder_input_2,aligned_small,adder_output;
    wire if_sub,a_sign, b_sign, c_sign,c1;
    wire [15:0] normalized_out;

    wire [15:0] result;
    reg [14:0] bigger, smaller;
    reg a_larger_b;

    assign a_sign        = a[15];
    assign b_sign        = b[15];
    assign if_sub        = (a_sign == b_sign) ? 1'b0 : 1'b1;
    assign c_sign        = a_larger_b ? a_sign : b_sign;
    assign adder_input_1 = {1'b1,bigger[9:0]};
    assign adder_input_2 = (if_sub ? ~aligned_small + 1'b1 : aligned_small);
    assign c             = result;

    //compare two number regardless sign
    always @(*) begin
        if (a[14:0] > b[14:0]) begin
            bigger = a[14:0];
            smaller = b[14:0];
            a_larger_b = 1'b1;
        end else begin 
            bigger = b[14:0];
            smaller = a[14:0];
            a_larger_b = 1'b0;
        end 
    end

    // align small number
    alignment u1(bigger,smaller,aligned_small);

    assign {c1, adder_output} = adder_input_1 + adder_input_2;

    add_normalizer u4(c_sign,bigger[14:10],adder_output,result,c1,if_sub);

endmodule

module alignment (
	input  [14:0] bigger, 
	input  [14:0] smaller,
	output [10:0] aligned_small
	);

	wire c1;
	wire [4:0] bigger_exponent, smaller_exponent,shift_bits;

	assign bigger_exponent  = bigger  [14:10];
	assign smaller_exponent = smaller [14:10];
	assign aligned_small    = ({1'b1,smaller[9:0]} >> shift_bits);

    assign {c1, shift_bits} = bigger_exponent - smaller_exponent;
endmodule

module add_normalizer (
    input             sign,
    input      [ 4:0] exponent,
    input      [10:0] mantissa_add,
    output reg [15:0] result,
    input             if_carray,
    input             if_sub
    );

    reg [4:0] number_of_zero_lead;
    reg [10:0] norm_mantissa_add;
    reg [9:0] mantissa_tmp;

    wire [4:0] shift_left_exp;
    wire c1;

    always @ (*) begin
        if (mantissa_add[10:4] == 7'b0000_001) begin
            number_of_zero_lead = 5'd6;
            norm_mantissa_add   = (mantissa_add << 4'd6);
        end else if (mantissa_add[10:5] == 6'b0000_01) begin 
            number_of_zero_lead = 5'd5;
            norm_mantissa_add   = (mantissa_add << 4'd5);
        end else if (mantissa_add[10:6] == 5'b0000_1) begin
            number_of_zero_lead = 5'd4;
            norm_mantissa_add   = (mantissa_add << 4'd4);
        end else if (mantissa_add[10:7] == 4'b0001) begin
            number_of_zero_lead = 5'd3;
            norm_mantissa_add   = (mantissa_add << 4'd3);
        end else if (mantissa_add[10:8] == 3'b001) begin
            number_of_zero_lead = 5'd2;
            norm_mantissa_add   = (mantissa_add << 4'd2);
        end else if (mantissa_add[10:9] == 2'b01) begin
            number_of_zero_lead = 5'd1;
            norm_mantissa_add   = (mantissa_add << 4'd1);
        end else begin 
            number_of_zero_lead = 5'd0;
            norm_mantissa_add   = mantissa_add[10:0];
        end 
    end

    always @(*) begin
        result[15]        = sign;
        if (!if_sub) begin 
            result[14:10] = if_carray ? exponent + 1'b1 : exponent;
            result[9:0]   = if_carray ? mantissa_add[10:1] : mantissa_add[9:0];
        end else begin 
            result[14:10] = shift_left_exp;
            result[9:0]   = norm_mantissa_add[9:0];
        end 
    end

	assign {c1, shift_left_exp} = exponent - number_of_zero_lead;
endmodule
