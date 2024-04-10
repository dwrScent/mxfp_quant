module fixed_point_multiplier(
    input [3:0] a,
    input [3:0] b,
    output reg [3:0] c
);

    reg [7:0] temp_product;

    always @(a or b) begin
        temp_product = a * b;
        
        if (temp_product[3]) begin
            c = temp_product[7:4] + 1;
        end else begin
            c = temp_product[7:4];
        end
    end

endmodule