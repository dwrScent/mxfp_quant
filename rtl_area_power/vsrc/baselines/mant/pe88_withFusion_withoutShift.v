// =============================================================
// Baseline PE for MX-ANT, MX-OLiVE, MX-BitFusion
// Function: 8-bit Mixed-Precision (2/4/8b) Multiply-Accumulate (MAC)
// Style based on the provided MANT PE module.
// Verilog-2001 Compliant.
// =============================================================
module pe_mac_baseline (
    input clk,
    input rst,
    input [1:0] precision,      // 00: 2-bit, 01: 4-bit, 10: 8-bit
    input [7:0] init_weight,    // for initialization
    input [7:0] in_activation,
    input [15:0] in_psum,       // Renamed from in_psum_1
    output reg [7:0] out_activation,
    output reg [15:0] out_psum        // Renamed from out_psum_1
);

    reg [7:0] weight;

    // The entire always@(*) block for shift calculations has been removed.
    // All `unsigned_shifted_*` registers have been removed.

    always @(posedge clk) begin
        // The weight register is loaded once at reset.
        if (rst) begin
            weight <= init_weight;
        end else begin
            // Activation from the previous PE is passed through.
	        out_activation <= in_activation;

            // MAC logic remains, supporting 2/4/8-bit precision.
            case (precision)
                2'b00: begin // 4x 2-bit MAC operations
                    out_psum <= out_activation * weight[7:6] 
                              + out_activation * weight[5:4] 
                              + out_activation * weight[3:2] 
                              + out_activation * weight[1:0] 
                              + in_psum;
                end
                2'b01: begin // 2x 4-bit MAC operations
                    out_psum <= out_activation * weight[7:4] 
                              + out_activation * weight[3:0] 
                              + in_psum;
                end
                2'b10: begin // 1x 8-bit MAC operation
                    out_psum <= out_activation * weight 
                              + in_psum;
                end
                default: begin
                    out_psum <= out_psum; // Keep previous value
                end
            endcase
            // All logic related to out_psum_2 has been removed.
        end
    end 
endmodule