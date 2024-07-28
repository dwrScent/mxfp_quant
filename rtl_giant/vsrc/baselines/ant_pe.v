module ant_pe(
	input clk,
	input rst,
	input [7:0] weight,
	input [7:0] activation,

	output [7:0] forwarded_input,
	output [7:0] forwarded_weight,

	output reg [15:0] acc
);
	assign forwarded_input = activation;
	assign forwarded_weight = weight;

	wire [3:0] exp_weight = weight[7:4];
	wire [3:0] exp_activation = activation[7:4];
	wire [3:0] base_int_weight = weight[3:0];
	wire [3:0] base_int_activation = activation[3:0];
	
	wire [3:0] exp_sum = exp_weight + exp_activation;

	always @(posedge clk) begin
		if(rst)
			acc <= 0;
		else begin
			acc <= acc + ((base_int_weight * base_int_activation) << (exp_weight + exp_activation));
		end
	end
endmodule

