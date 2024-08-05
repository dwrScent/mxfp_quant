module tender_Pe_withFusion(
	input clk,
	input rst,
	input [7:0] weight,
	input [7:0] activation,
	input precision, // 0 for int4 and 1 for int8
	input rescale_low,

	output [7:0] forwarded_input,
	output forwarded_rescale_low,
	output [7:0] forwarded_weight,

	output reg [31:0] acc
);

	assign forwarded_input = activation;
	assign forwarded_rescale_low = rescale_low;
	assign forwarded_weight = weight;

	wire [3:0] weight_low = weight[3:0];
	wire [3:0] weight_high = weight[7:4];
	wire [3:0] activation_low = activation[3:0];
	wire [3:0] activation_high = activation[7:4];

	always @(posedge clk) begin
		if(rst)
			acc <= 0;
		else begin
				// left shift acc 1 bit
				case(precision)
					0: begin
						if (rescale_low) begin
							acc[15:0] <= acc[15:0] << 1;
							acc[31:16] <= acc[31:16] << 1;
						end
						else begin
							acc[15:0] <= acc[15:0] + (activation_low * weight_low);	
							acc[31:16] <= acc[31:16] + (activation_high * weight_high);
						end
					end
					1: begin
						if (rescale_low)
							acc <= acc << 1;
						else
							acc <= acc + (activation * weight);
					end
				endcase
		end
	end
endmodule

