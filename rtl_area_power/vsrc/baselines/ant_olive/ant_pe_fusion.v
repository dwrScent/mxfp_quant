// =============================================================
// Baseline PE for MX-ANT, MX-OLiVE, MX-BitFusion
// Function: Mixed-Precision MAC supporting W8A8 and W4A4
// Verilog-2001 Compliant.
// =============================================================
module pe_mac_baseline (
    input clk,
    input rst,
    input mode_is_w8a8,     // Mode select: 1 for W8A8, 0 for W4A4
    input [7:0] init_weight,    // for initialization
    input [7:0] in_activation,
    input [15:0] in_psum,
    output reg [7:0] out_activation,
    output reg [15:0] out_psum
);

    reg [7:0] weight;

    always @(posedge clk) begin
        // The weight register is loaded once at reset.
        if (rst) begin
            weight <= init_weight;
        end else begin
            // Activation from the previous PE is passed through.
            out_activation <= in_activation;

            // Select MAC operation based on the mode.
            if (mode_is_w8a8) begin
                // W8A8 Mode: Perform one 8x8 MAC operation.
                out_psum <= in_activation * weight + in_psum;
            end else begin
                // W4A4 Mode: Perform two parallel 4x4 MAC operations.
                // The 8-bit datapath is split into two 4-bit lanes.
                out_psum <= (in_activation[7:4] * weight[7:4])
                          + (in_activation[3:0] * weight[3:0])
                          + in_psum;
            end
        end
    end
endmodule