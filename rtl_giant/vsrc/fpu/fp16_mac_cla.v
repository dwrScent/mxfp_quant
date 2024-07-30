module fp16_mac_cla (
	input clk,
	input rst,
	input  [15:0] a,
    input  [15:0] b,
    output reg [15:0] c_reg,
	output error
);
	
	wire [15:0] c_tmp;
	wire [15:0] c_inner;
	always @(posedge clk) begin
		if(rst)
			c_reg <= 0;
		else begin
			c_reg <= c_tmp;
		end
	end

	fp16_mul u1(a,b,c_inner,error);
	fp16_adder u2(c_inner, c_reg, c_tmp);
endmodule

module fp16_adder (
    input  [15:0] a,
    input  [15:0] b,
    output [15:0] c
    );

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

    cla_nbit #(.n(11)) u2(adder_input_1,adder_input_2,1'b0,adder_output,c1);

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

	cla_nbit #(.n(5)) u1(bigger_exponent,~smaller_exponent+1'b1,1'b0,shift_bits,c1);

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

    cla_nbit #(.n(5)) u1(exponent,~number_of_zero_lead+1'b1,1'b0,shift_left_exp,c1);

endmodule

module fp16_mul (
    input  [15:0] a,
    input  [15:0] b,
    output [15:0] c,
    output        error // valid in fp16 mode 
    );

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
    mul16x16 u1(multiplier_input1,multiplier_input2,multiplier_output);
    
    cla_nbit #(.n(5)) u2(a[14:10],b[14:10],1'b0,sum_exponent,c1); // add exponent
    cla_nbit #(.n(5)) u3(sum_exponent, 5'b10001,1'b0,biased_sum_exponent,c2); // minus bias
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

module mul16x16(
    input  [15:0] a,
    input  [15:0] b,
    output [31:0] c);

    wire [63:0] tmp1,tmp2;
    wire [23:0] result1;
    wire [23:0] result2;
    wire co1,co2,co3;

	assign tmp2 = tmp1;

    mul8x8 u1(a[15:8],b[15:8],tmp1[63:48]);
    mul8x8 u2(a[7:0] ,b[15:8],tmp1[47:32]);
    mul8x8 u3(a[15:8],b[ 7:0],tmp1[31:16]);
    mul8x8 u4(a[7:0] ,b[ 7:0],tmp1[15:0]);

    cla_nbit #(.n(24)) u5({tmp2[63:48],8'b0} ,{8'b0,tmp2[47:32]} ,1'b0 ,result1 ,co1);
    cla_nbit #(.n(24)) u6({8'b0,tmp2[31:16]} ,{16'b0,tmp2[15:8]} ,co1  ,result2 ,co2);
    cla_nbit #(.n(24)) u7(result1            ,result2            ,co2  ,c[31:8] ,co3);

    assign c[7:0] = tmp2[7:0];

endmodule

module mul8x8(
	input  [ 7:0] a,
	input  [ 7:0] b,
	output [15:0] c
);

	wire [31:0] tmp1;
	wire [11:0] result1;
	wire [11:0] result2;
	wire co1,co2,co3;

	mul4x4 u1(a[7:4],b[7:4],tmp1[31:24]);
	mul4x4 u2(a[3:0],b[7:4],tmp1[23:16]);
	mul4x4 u3(a[7:4],b[3:0],tmp1[15:8]);
	mul4x4 u4(a[3:0],b[3:0],tmp1[7:0]);

	cla_nbit #(.n(12)) u5({tmp1[31:24],4'b0} ,{4'b0,tmp1[23:16]} ,1'b0 ,result1 ,co1);
	cla_nbit #(.n(12)) u6({4'b0,tmp1[15:8]}  ,{8'b0,tmp1[7:4]}   ,co1  ,result2 ,co2);
	cla_nbit #(.n(12)) u7(result1			 ,result2			 ,co2  ,c[15:4] ,co3);

	assign c[3:0] = tmp1[3:0];

endmodule

module mul4x4(
	input  [3:0] a,
	input  [3:0] b,
	output [7:0] c
	);

	wire [15:0] tmp1;
	wire [ 5:0] result1;
	wire [ 5:0] result2;
	wire 		co1,co2,co3;

	mul2x2 u1(a[3:2],b[3:2],tmp1[15:12]);
	mul2x2 u2(a[1:0],b[3:2],tmp1[11:8]);
	mul2x2 u3(a[3:2],b[1:0],tmp1[7:4]);
	mul2x2 u4(a[1:0],b[1:0],tmp1[3:0]);

	cla_nbit #(.n(6)) u5({tmp1[15:12],2'b0},{2'b0,tmp1[11:8]},1'b0	,result1	,co1);
	cla_nbit #(.n(6)) u6({2'b0,tmp1[7:4]}  ,{4'b0,tmp1[3:2]} ,co1 	,result2	,co2);
	cla_nbit #(.n(6)) u7(result1		   ,result2			 ,co2 	,c[7:2] 	,co3);

	assign c[1:0] = tmp1[1:0];

endmodule

module mul2x2(
	input  [1:0] a,
	input  [1:0] b,
	output [3:0] c
	);

	wire [3:0] tmp;
	
	assign tmp[0] = a[0] & b[0];
	assign tmp[1] = (a[1]&b[0]) ^ (a[0]&b[1]);
	assign tmp[2] = (a[0]&b[1]) & (a[1]&b[0]) ^ (a[1]&b[1]);
	assign tmp[3] = (a[0]&b[1]) & (a[1]&b[0]) & (a[1]&b[1]);
	assign c 	  = {tmp[3],tmp[2],tmp[1],tmp[0]};

endmodule


module cla_nbit #(
    parameter n = 4
) (
  input   [n-1:0] a,
  input   [n-1:0] b,
  input           ci,
  output  [n-1:0] s,
  output          co
  );

  wire [n-1:0] g;
  wire [n-1:0] p;
  wire [  n:0] c;

  assign c[0] = ci;
  assign co   = c[n];

  genvar i;  /* i - generate index variable */

  generate
    for (i = 0; i < n; i = i + 1) begin : addbit
      assign s[i] = a[i] ^ b[i] ^ c[i];
      assign g[i] = a[i] & b[i];
      assign p[i] = a[i] | b[i];
      assign c[i + 1] = g[i] | (p[i] & c[i]);
    end
  endgenerate
  
endmodule
