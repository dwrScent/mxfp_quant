module fixed_point_multiplier(
    input clk,
    input rst,
    input [3:0] a,
    input [3:0] b,
    output reg [3:0] c
);

    reg [7:0] temp_product;
    reg [3:0] final_result;

    // combinational
    always @(a or b) begin
        temp_product = a * b;
        if (temp_product[3]) begin
            final_result = temp_product[7:4] + 1;
        end else begin
            final_result = temp_product[7:4];
        end
    end
    
    // sequential
    always @(posedge clk) begin
        if(!rst) 
            c <= final_result;
    end

endmodule