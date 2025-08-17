// ====================================================================
// Tile for ANT/OLIVE/BitFusion Architectures
// Structure: 1x4 array of PEs, providing 8 parallel 4x4 MACs.
// Verilog-2001 Compliant.
// ====================================================================
module tile_1x4_ant (
    input clk,
    input rst,
    input mode_is_w8a8,              // Mode select, broadcast to all PEs

    // Inputs for 4 PEs (packed into buses)
    input [4*8-1:0]  in_activations_bus,  // 4x 8-bit activations
    input [4*8-1:0]  init_weights_bus,    // 4x 8-bit weights
    input [4*32-1:0] in_psums_bus,        // 4x 32-bit initial psums

    // Outputs for 4 PEs (packed into buses)
    output [4*8-1:0]  out_activations_bus,
    output [4*32-1:0] out_psums_bus
);

    parameter NUM_PES = 4;
    genvar i;

    // Generate 4 parallel PE instances.
    // There are no direct connections between these PEs in this simple tile structure.
    generate
        for (i = 0; i < NUM_PES; i = i + 1) begin: PE_GEN
            
            pe_mac_baseline_32b u_pe (
                .clk            (clk),
                .rst            (rst),
                .mode_is_w8a8   (mode_is_w8a8),
                
                // Slice the buses to feed each PE
                .init_weight    (init_weights_bus[i*8 + 7 : i*8]),
                .in_activation  (in_activations_bus[i*8 + 7 : i*8]),
                .in_psum        (in_psums_bus[i*32 + 31 : i*32]),

                // Connect outputs to the corresponding slice of the output bus
                .out_activation (out_activations_bus[i*8 + 7 : i*8]),
                .out_psum       (out_psums_bus[i*32 + 31 : i*32])
            );
            
        end
    endgenerate

endmodule

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