module pe88 (
    input clk,
    input rst,
    input [1:0] precision, // 00: 2bit, 01: 4bit, 10: 8bit
    input [7:0] init_weight, // for initialization
    input [7:0] in_activation,
    input [15:0] in_psum_1,
    input [15:0] in_psum_2,
    output reg [7:0] out_activation,
    output reg [15:0] out_psum_1,
    output reg [15:0] out_psum_2
);
	reg [7:0] weight;

    reg [15:0] unsigned_shifted_0; // used for w[1:0] or w[3:0] or w[7:0]
    reg [15:0] unsigned_shifted_1; // used for w[3:2] or w[7:4]
    reg [15:0] unsigned_shifted_2; // used for w[5:4]
    reg [15:0] unsigned_shifted_3; // used for w[7:6]
    always @(*) begin
        case (precision)
            2'b00: begin
                unsigned_shifted_0 = {8'd0, out_activation} << weight[0];
                unsigned_shifted_1 = {8'd0, out_activation} << weight[2];
                unsigned_shifted_2 = {8'd0, out_activation} << weight[4];
                unsigned_shifted_3 = {8'd0, out_activation} << weight[6];
            end
            2'b01: begin
                unsigned_shifted_0 = {8'd0, out_activation} << weight[2:0];
                unsigned_shifted_1 = {8'd0, out_activation} << weight[6:4];
                unsigned_shifted_2 = 0;
                unsigned_shifted_3 = 0;
            end
            2'b10: begin
                unsigned_shifted_0 = {8'd0, out_activation} << weight[6:0];
                unsigned_shifted_1 = 0;
                unsigned_shifted_2 = 0;
                unsigned_shifted_3 = 0;
            end
            default: begin
                unsigned_shifted_0 = 0;
                unsigned_shifted_1 = 0;
                unsigned_shifted_2 = 0;
                unsigned_shifted_3 = 0;
            end
        endcase
    end
    always @(posedge clk) begin
    // receive activation
        if(rst) begin
            weight <= init_weight;
        end else begin
	        out_activation <= in_activation;
            case (precision)
                2'b00: begin
                    out_psum_1 <= out_activation * weight[7:6] 
                                + out_activation * weight[5:4] 
                                + out_activation * weight[3:2] 
                                + out_activation * weight[1:0] 
                                + in_psum_1;
                    out_psum_2 <= ((weight[7]==1'b0) ? unsigned_shifted_0 : -unsigned_shifted_0) 
                                + ((weight[5]==1'b0) ? unsigned_shifted_1 : -unsigned_shifted_1) 
                                + ((weight[3]==1'b0) ? unsigned_shifted_2 : -unsigned_shifted_2) 
                                + ((weight[1]==1'b0) ? unsigned_shifted_3 : -unsigned_shifted_3) 
                                + in_psum_2;
                end
                2'b01: begin
                    out_psum_1 <= out_activation * weight[7:4] 
                                + out_activation * weight[3:0] 
                                + in_psum_1;
                    out_psum_2 <= ((weight[7]==1'b0) ? unsigned_shifted_0 : -unsigned_shifted_0)
                                + ((weight[3]==1'b0) ? unsigned_shifted_1 : -unsigned_shifted_1)
                                + in_psum_2;
                end
                2'b10: begin
                    out_psum_1 <= out_activation * weight + in_psum_1;
                    out_psum_2 <= (weight[7]==1'b0 ? unsigned_shifted_0 : -unsigned_shifted_0) + in_psum_2;
                end
                default: begin
                    out_psum_1 <= out_psum_1;
                    out_psum_2 <= out_psum_2;
                end
            endcase
        end
    end 
endmodule
