module tender_Pe(
	input clk,
	input rst,
	input [3:0] weight,
	input [3:0] activation,
	input rescale,

	output [3:0] forwarded_input,
	output forwarded_rescale,
	output [3:0] forwarded_weight,

	output reg [31:0] acc
);
	assign forwarded_input = activation;
	assign forwarded_rescale = rescale;
	assign forwarded_weight = weight;

	always @(posedge clk) begin
		if(rst)
			acc <= 0;
		else begin
			if (rescale)
				// left shift acc 1 bit
				acc <= acc << 1;
			else
				acc <= acc + activation * weight;
		end
	end
endmodule
