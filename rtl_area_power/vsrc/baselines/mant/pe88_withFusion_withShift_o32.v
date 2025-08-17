// =============================================================
// MANT PE with 32-bit psum and simplified precision (W4/W8, A8)
// Verilog-2001 Compliant.
// =============================================================
module pe_mant_32b (
    input clk,
    input rst,
    input mode_is_w8a8,         // [MODIFIED] Mode select: 1 for W8A8, 0 for W4A8
    input [7:0] init_weight,
    input [7:0] in_activation,
    input [31:0] in_psum_mac,   // [MODIFIED] Port width changed to 32
    input [31:0] in_psum_sac,   // [MODIFIED] Port width changed to 32
    output reg [7:0] out_activation,
    output reg [31:0] out_psum_mac,  // [MODIFIED] Port width changed to 32
    output reg [31:0] out_psum_sac   // [MODIFIED] Port width changed to 32
);
	reg [7:0] weight;

    // Internal registers for the SAC (Shift-Accumulate) path
    reg [31:0] unsigned_shifted_0; // [MODIFIED] Widened to 32 bits
    reg [31:0] unsigned_shifted_1; // [MODIFIED] Widened to 32 bits

    // Combinational logic for SAC path's shift amount
    always @(*) begin
        if (mode_is_w8a8) begin // W8A8 Mode
            unsigned_shifted_0 = {24'd0, out_activation} << weight[6:0];
            unsigned_shifted_1 = 32'b0;
        end else begin // W4A8 Mode
            unsigned_shifted_0 = {24'd0, out_activation} << weight[2:0];
            unsigned_shifted_1 = {24'd0, out_activation} << weight[6:4];
        end
    end

    // Main sequential logic for both MAC and SAC paths
    always @(posedge clk) begin
        if(rst) begin
            weight <= init_weight;
        end else begin
	        out_activation <= in_activation;
	        
            if (mode_is_w8a8) begin
                // W8A8 Mode
                // MAC Path
                out_psum_mac <= out_activation * weight + in_psum_mac;
                // SAC Path
                out_psum_sac <= (weight[7]==1'b0 ? unsigned_shifted_0 : -unsigned_shifted_0) 
                              + in_psum_sac;
            end else begin
                // W4A8 Mode
                // MAC Path
                out_psum_mac <= out_activation * weight[7:4] 
                              + out_activation * weight[3:0] 
                              + in_psum_mac;
                // SAC Path
                out_psum_sac <= ((weight[7]==1'b0) ? unsigned_shifted_0 : -unsigned_shifted_0)
                              + ((weight[3]==1'b0) ? unsigned_shifted_1 : -unsigned_shifted_1)
                              + in_psum_sac;
            end
        end
    end 
endmodule