module unsigned_int16_comparator(
    input clk,
    input rst,
    input mode, // 0 for up mode and 1 for left mode
    input [15:0] left,
    input [15:0] up,
    output [15:0] down,
    output [15:0] right
    );
    wire [15:0] in_data = (mode==0) ? left : up;
    reg [15:0] result;
    assign down = result;
    assign right = result;
    
    always @(posedge clk) begin
        if(rst) begin
            result <= 0;
        end else begin
            if(in_data>result) begin
                result <= in_data;
            end
        end
    end
endmodule