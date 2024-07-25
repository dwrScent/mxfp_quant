module fp8mul (
  input clk,
  input rst,

  input [7:0] in1,
  input [7:0] in2,

  output reg [7:0] out
);

	wire sign1 = in1[7];
	wire [3:0] exp1 = in1[6:3];
	wire [2:0] mant1 = in1[2:0];

	wire sign2 = in2[7];
	wire [3:0] exp2 = in2[6:3];
	wire [2:0] mant2 = in2[2:0];

	wire sign_out;
	wire [3:0] exp_out;
	wire [2:0] mant_out;

	always @(posedge clk) begin
		if(rst) begin
			out <= 8'b0;
		end else begin
			out <= {sign_out, exp_out, mant_out};
		end
	end

    parameter EXP_BIAS = 7;
    wire isnan = (sign1 == 1 && exp1 == 0 && mant1 == 0) || (sign2 == 1 && exp2 == 0 && mant2 == 0);
    wire [7:0] full_mant = ({exp1 != 0, mant1} * {exp2 != 0, mant2});
    wire overflow_mant = full_mant[7];
    wire [6:0] shifted_mant = overflow_mant ? full_mant[6:0] : {full_mant[5:0], 1'b0};
    // is the mantissa overflowing up to the next exponent?
    wire roundup = (exp1 + exp2 + overflow_mant < 1 + EXP_BIAS) && (shifted_mant[6:0] != 0)
                   || (shifted_mant[6:4] == 3'b111 && shifted_mant[3]);
    wire underflow = (exp1 + exp2 + overflow_mant) < 1 - roundup + EXP_BIAS;
    wire is_zero = exp1 == 0 || exp2 == 0 || isnan || underflow;
    // note: you can't use negative numbers reliably. just keep things positive during compares.
    wire [4:0] exp_out_tmp = (exp1 + exp2 + overflow_mant + roundup) < EXP_BIAS ? 0 : (exp1 + exp2 + overflow_mant + roundup - EXP_BIAS);
    assign exp_out = exp_out_tmp > 15 ? 4'b1111 : (is_zero) ? 0 : exp_out_tmp[3:0];  // Exponent bias is 7
    assign mant_out = exp_out_tmp > 15 ? 3'b111 : (is_zero || roundup) ? 0 : (shifted_mant[6:4] + (shifted_mant[3:0] > 8 || (shifted_mant[3:0] == 8 && shifted_mant[4])));
    assign sign_out = ((sign1 ^ sign2) && !(is_zero)) || isnan;
endmodule
