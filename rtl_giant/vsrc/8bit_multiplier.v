module eight_bit_multiplier(
    input clk,
    input rst,
    input [7:0] a,
    input [7:0] b,
    output reg [7:0] c
);

    reg [15:0] temp_product;
	reg [7:0] final_result;
    // combinational
    always @(a or b) begin
        temp_product = a * b;
        if (temp_product[7]) begin
            final_result = temp_product[15:8] + 1;
        end else begin
            final_result = temp_product[15:8];
        end
    end
    
    // sequential
    always @(posedge clk) begin
        if(!rst) 
            c <= final_result;
    end

endmodule