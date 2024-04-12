module pe88_no_fusion (
	input clk,
    input rst,
    input [7:0] weight_init,
	input [7:0] in_activation,
	input [15:0] in_psum_1,
	input [15:0] in_psum_2,
	output reg [7:0] out_activation,
	output reg [15:0] out_psum_1,
	output reg [15:0] out_psum_2
);
    reg [7:0] weight;

    wire [7:0] unsigned_shifted = out_activation << weight;
    wire [7:0] signed_shifted = weight[7]==1'b0 ? unsigned_shifted : -unsigned_shifted;
    always @(posedge clk) begin
        if(rst) begin
            weight <= weight_init;
        end else begin
            out_activation <= in_activation;
            out_psum_1 <= out_activation * weight + in_psum_1;  
            out_psum_2 <= {8'd0, signed_shifted} + in_psum_2;
        end
    end 
endmodule
