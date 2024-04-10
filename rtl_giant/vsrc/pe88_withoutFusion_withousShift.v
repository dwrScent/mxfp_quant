module pe88_withFusion_withoutShift (
	input clk,
    input rst,
    input [7:0] weight_init,
	input [7:0] in_activation,
	input [15:0] in_psum_1,
	output reg [7:0] out_activation,
	output reg [15:0] out_psum_1
);
    reg [7:0] weight;

    always @(posedge clk) begin
        if(rst) begin
            weight <= weight_init;
		end else begin
            out_activation <= in_activation;
            out_psum_1 <= out_activation * weight + in_psum_1;  
		end
    end 
endmodule
