// =============================================================
// PE for MicroScopiQ Baseline (32-bit psum)
// Function: Mixed-Precision (W={2,4}b, A={4,8}b) MAC
// Verilog-2001 Compliant.
// =============================================================
module pe_microscopiq_32b (
    input clk,
    input rst,
    input precision_w,      // Weight precision: 0 for 2-bit, 1 for 4-bit
    input precision_a,      // Activation precision: 0 for 4-bit, 1 for 8-bit
    input [7:0] init_weight,
    input [7:0] in_activation,
    input [31:0] in_psum,       // [MODIFIED] Port width changed to 32
    output reg [7:0] out_activation,
    output reg [31:0] out_psum        // [MODIFIED] Port width changed to 32
);

    reg [7:0] weight;
    wire [1:0] mode = {precision_a, precision_w};

    always @(posedge clk) begin
        if (rst) begin
            weight <= init_weight;
        end else begin
            out_activation <= in_activation;

            case (mode)
                // Mode 0: A=4b, W=2b
                // One 8-bit activation bus carries two 4-bit inputs.
                // Performs two sets of (four 4x2 MACs).
                2'b00: begin
                    out_psum <= (in_activation[3:0] * weight[3:2] + in_activation[3:0] * weight[1:0])
                              + (in_activation[7:4] * weight[7:6] + in_activation[7:4] * weight[5:4])
                              + in_psum;
                end

                // Mode 1: A=4b, W=4b
                // [MODIFIED] Performs two parallel 4x4 MACs for fair comparison.
                2'b01: begin
                    out_psum <= (in_activation[7:4] * weight[7:4])
                              + (in_activation[3:0] * weight[3:0])
                              + in_psum;
                end

                // Mode 2: A=8b, W=2b
                // Performs four 8x2 MACs and sums them up.
                2'b10: begin
                    out_psum <= in_activation * weight[7:6] 
                              + in_activation * weight[5:4] 
                              + in_activation * weight[3:2] 
                              + in_activation * weight[1:0] 
                              + in_psum;
                end

                // Mode 3: A=8b, W=4b
                // Performs two parallel 8x4 MACs.
                2'b11: begin
                    out_psum <= in_activation * weight[7:4] 
                              + in_activation * weight[3:0]
                              + in_psum;
                end
                
                default: begin
                    out_psum <= out_psum; // Keep previous value
                end
            endcase
        end
    end 
endmodule