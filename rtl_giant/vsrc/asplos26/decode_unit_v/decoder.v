// ============================
// FP4 to UINT Lookup Table
// ============================
module fp4_to_uint_lut (
    input  [3:0] fp4_in,
    output reg [3:0] val_out
);
    always @(*) begin
        case (fp4_in)
            // Value : FP4_Binary -> UINT_Binary
            4'b0111: val_out = 4'b1111; // +6.0
            4'b1111: val_out = 4'b1110; // -6.0
            4'b0110: val_out = 4'b1101; // +4.0
            4'b1110: val_out = 4'b1100; // -4.0
            4'b0101: val_out = 4'b1011; // +3.0
            4'b1101: val_out = 4'b1010; // -3.0
            4'b0100: val_out = 4'b1001; // +2.0
            4'b1100: val_out = 4'b1000; // -2.0
            4'b0011: val_out = 4'b0111; // +1.5
            4'b1011: val_out = 4'b0110; // -1.5
            4'b0010: val_out = 4'b0101; // +1.0
            4'b1010: val_out = 4'b0100; // -1.0
            4'b0001: val_out = 4'b0011; // +0.5
            4'b1001: val_out = 4'b0010; // -0.5
            4'b0000: val_out = 4'b0001; // +0.0
            4'b1000: val_out = 4'b0000; // -0.0
            default: val_out = 4'b0000; // Default to smallest value for safety
        endcase
    end
endmodule

// ============================
// Comparator with tie-break by idx
// ============================
module comp (
    input  [3:0] val_a,
    input  [2:0] idx_a,
    input  [3:0] val_b,
    input  [2:0] idx_b,
    output reg [3:0] val_out,
    output reg [2:0] idx_out
);
    always @(*) begin
        if (val_a > val_b) begin
            val_out = val_a;
            idx_out = idx_a;
        end else if (val_b > val_a) begin
            val_out = val_b;
            idx_out = idx_b;
        end else begin
            if (idx_a < idx_b) begin
                val_out = val_a;
                idx_out = idx_a;
            end else begin
                val_out = val_b;
                idx_out = idx_b;
            end
        end
    end
endmodule

// ============================
// Top-1 Detection Unit
// ============================
module top1_detection_unit (
    input  [3:0] fp4_in0, fp4_in1, fp4_in2, fp4_in3,
    input  [3:0] fp4_in4, fp4_in5, fp4_in6, fp4_in7,
    output [2:0] idx_max
);
    // LUT outputs
    wire [3:0] val0, val1, val2, val3, val4, val5, val6, val7;

    fp4_to_uint_lut lut0 (.fp4_in(fp4_in0), .val_out(val0));
    fp4_to_uint_lut lut1 (.fp4_in(fp4_in1), .val_out(val1));
    fp4_to_uint_lut lut2 (.fp4_in(fp4_in2), .val_out(val2));
    fp4_to_uint_lut lut3 (.fp4_in(fp4_in3), .val_out(val3));
    fp4_to_uint_lut lut4 (.fp4_in(fp4_in4), .val_out(val4));
    fp4_to_uint_lut lut5 (.fp4_in(fp4_in5), .val_out(val5));
    fp4_to_uint_lut lut6 (.fp4_in(fp4_in6), .val_out(val6));
    fp4_to_uint_lut lut7 (.fp4_in(fp4_in7), .val_out(val7));

    // First stage
    wire [3:0] val_s1_0, val_s1_1, val_s1_2, val_s1_3;
    wire [2:0] idx_s1_0, idx_s1_1, idx_s1_2, idx_s1_3;

    comp c0 (.val_a(val0), .idx_a(3'd0), .val_b(val1), .idx_b(3'd1), .val_out(val_s1_0), .idx_out(idx_s1_0));
    comp c1 (.val_a(val2), .idx_a(3'd2), .val_b(val3), .idx_b(3'd3), .val_out(val_s1_1), .idx_out(idx_s1_1));
    comp c2 (.val_a(val4), .idx_a(3'd4), .val_b(val5), .idx_b(3'd5), .val_out(val_s1_2), .idx_out(idx_s1_2));
    comp c3 (.val_a(val6), .idx_a(3'd6), .val_b(val7), .idx_b(3'd7), .val_out(val_s1_3), .idx_out(idx_s1_3));

    // Second stage
    wire [3:0] val_s2_0, val_s2_1;
    wire [2:0] idx_s2_0, idx_s2_1;

    comp c4 (.val_a(val_s1_0), .idx_a(idx_s1_0), .val_b(val_s1_1), .idx_b(idx_s1_1), .val_out(val_s2_0), .idx_out(idx_s2_0));
    comp c5 (.val_a(val_s1_2), .idx_a(idx_s1_2), .val_b(val_s1_3), .idx_b(idx_s1_3), .val_out(val_s2_1), .idx_out(idx_s2_1));

    // Third stage
    wire [3:0] val_final;
    comp c6 (.val_a(val_s2_0), .idx_a(idx_s2_0), .val_b(val_s2_1), .idx_b(idx_s2_1), .val_out(val_final), .idx_out(idx_max));

endmodule
