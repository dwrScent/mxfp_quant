// =============================================================
// PE for MicroScopiQ Baseline
// Function: Mixed-Precision (W={2,4}b, A={4,8}b) Multiply-Accumulate
// Style based on previous baseline PE modules.
// Verilog-2001 Compliant.
// =============================================================
module pe_microscopiq (
    input clk,
    input rst,
    input precision_w,      // Weight precision: 0 for 2-bit, 1 for 4-bit
    input precision_a,      // Activation precision: 0 for 4-bit, 1 for 8-bit
    input [7:0] init_weight,    // for initialization
    input [7:0] in_activation,
    input [15:0] in_psum,
    output reg [7:0] out_activation,
    output reg [15:0] out_psum
);

    reg [7:0] weight;

    // A 2-bit signal to control the case statement for all 4 modes
    wire [1:0] mode = {precision_a, precision_w};

    always @(posedge clk) begin
        if (rst) begin
            weight <= init_weight;
        end else begin
            // Pass through the activation from the previous PE
            out_activation <= in_activation;

            case (mode)
                // Mode 0: Activation 4-bit, Weight 2-bit
                // Performs four 4x2 MAC operations and sums them up.
                2'b00: begin
                    out_psum <= in_activation[3:0] * weight[7:6] 
                              + in_activation[3:0] * weight[5:4] 
                              + in_activation[3:0] * weight[3:2] 
                              + in_activation[3:0] * weight[1:0] 
                              + in_psum;
                end

                // Mode 1: Activation 4-bit, Weight 4-bit
                // Performs one 4x4 MAC. Uses lower 4 bits of weight.
                2'b01: begin
                    out_psum <= in_activation[3:0] * weight[3:0] 
                              + in_psum;
                end

                // Mode 2: Activation 8-bit, Weight 2-bit
                // Performs four 8x2 MAC operations and sums them up.
                2'b10: begin
                    out_psum <= in_activation * weight[7:6] 
                              + in_activation * weight[5:4] 
                              + in_activation * weight[3:2] 
                              + in_activation * weight[1:0] 
                              + in_psum;
                end

                // Mode 3: Activation 8-bit, Weight 4-bit
                // Performs one 8x4 MAC. Uses lower 4 bits of weight.
                2'b11: begin
                    out_psum <= in_activation * weight[3:0] 
                              + in_psum;
                end
                
                default: begin
                    out_psum <= out_psum; // Keep previous value
                end
            endcase
        end
    end 
endmodule