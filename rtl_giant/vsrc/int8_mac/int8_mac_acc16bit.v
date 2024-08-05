module int8_mac_acc16bit(
    input clk,
    input rst,
    input [7:0] a,
    input [7:0] b,
    output reg [15:0] c
);
    always @(posedge clk) begin
        if(rst) 
            c <= 0;
        else     
            c <= a * b + c;
    end
endmodule