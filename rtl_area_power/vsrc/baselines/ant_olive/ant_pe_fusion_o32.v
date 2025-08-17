// =============================================================
// Baseline PE for MX-ANT, MX-OLiVE, MX-BitFusion (32-bit psum)
// Function: Mixed-Precision MAC supporting W8A8 and W4A4
// Verilog-2001 Compliant.
// =============================================================
module pe_mac_baseline_32b (
    input clk,
    input rst,
    input mode_is_w8a8,     // Mode select: 1 for W8A8, 0 for W4A4
    input [7:0] init_weight,    // for initialization
    input [7:0] in_activation,
    input [31:0] in_psum,       // [MODIFIED] Port width changed to 32
    output reg [7:0] out_activation,
    output reg [31:0] out_psum        // [MODIFIED] Port width changed to 32
);

    reg [7:0] weight;

    always @(posedge clk) begin
        if (rst) begin
            weight <= init_weight;
        end else begin
            out_activation <= in_activation;

            if (mode_is_w8a8) begin
                // W8A8 Mode: Perform one 8x8 MAC operation.
                out_psum <= in_activation * weight + in_psum;
            end else begin
                // W4A4 Mode: Perform two parallel 4x4 MAC operations.
                out_psum <= (in_activation[7:4] * weight[7:4])
                          + (in_activation[3:0] * weight[3:0])
                          + in_psum;
            end
        end
    end
endmodule