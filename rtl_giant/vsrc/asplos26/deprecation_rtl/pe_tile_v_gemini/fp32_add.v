// IEEE-754 single-precision adder (Synthesizable, modularized)
module fp32_add(
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] z
);
    // 1. Deconstruct inputs
    wire sa = a[31]; wire [7:0] ea = a[30:23]; wire [22:0] fa = a[22:0];
    wire sb = b[31]; wire [7:0] eb = b[30:23]; wire [22:0] fb = b[22:0];

    // 2. Special value detection
    wire a_nan = (ea == 8'hFF) && (fa != 0);
    wire b_nan = (eb == 8'hFF) && (fb != 0);
    wire a_inf = (ea == 8'hFF) && (fa == 0);
    wire b_inf = (eb == 8'hFF) && (fb == 0);
    wire a_zero= (ea == 0)     && (fa == 0);
    wire b_zero= (eb == 0)     && (fb == 0);
    
    // 3. Select larger and smaller operands
    wire a_mag_gt_b = (ea > eb) || ((ea == eb) && (fa > fb));
    
    wire        s_big, s_small;
    wire [7:0]  e_big, e_small;
    wire [23:0] m_big, m_small; // Mantissa with hidden bit

    assign s_big   = a_mag_gt_b ? sa : sb;
    assign e_big   = a_mag_gt_b ? ea : eb;
    assign m_big   = a_mag_gt_b ? {(ea!=0), fa} : {(eb!=0), fb}; // Add hidden bit
    
    assign s_small = a_mag_gt_b ? sb : sa;
    assign e_small = a_mag_gt_b ? eb : ea;
    assign m_small = a_mag_gt_b ? {(eb!=0), fb} : {(ea!=0), fa}; // Add hidden bit

    // 4. Align smaller mantissa
    wire [24:0] m_small_aligned; // 24-bit mantissa + 1 sticky bit
    
    fp32_align u_align (
        .e_big   (e_big),
        .e_small (e_small),
        .m_small (m_small),
        .m_small_aligned(m_small_aligned)
    );

    // 5. Perform Add/Subtract
    wire is_sub = (sa != sb);
    wire [25:0] sum;
    
    // Add an extra bit for potential carry
    wire [24:0] m_big_ext = {1'b0, m_big}; 
    wire [24:0] m_small_eff = is_sub ? ~m_small_aligned : m_small_aligned;
    
    assign sum = m_big_ext + m_small_eff + (is_sub ? 1'b1 : 1'b0);

    // 6. Normalize the result
    wire [31:0] z_norm;
    fp32_normalize u_normalize (
        .s_in     (s_big),
        .e_in     (e_big),
        .sum_in   (sum),
        .is_sub   (is_sub),
        .z_out    (z_norm)
    );

    // 7. Final Output Selection
    assign z = a_nan ? 32'h7FC00001 : // Return a quiet NaN
               b_nan ? 32'h7FC00001 :
               (a_inf && b_inf && is_sub) ? 32'h7FC00000 : // Inf - Inf = NaN
               a_inf ? a :
               b_inf ? b :
               (a_zero && b_zero) ? (sa & sb ? 32'h80000000 : 32'h0) :
               (sum[24:0] == 0) ? 32'h0 : // Result is zero
               z_norm;

endmodule

module fp32_align (
    input  [7:0]  e_big,
    input  [7:0]  e_small,
    input  [23:0] m_small,
    output [24:0] m_small_aligned // 24 bits for mantissa, 1 bit for sticky
);
    wire [7:0] shift_amount_raw = e_big - e_small;
    
    // Cap shift amount to prevent excessive shifting
    wire [5:0] shift_amount = (shift_amount_raw > 25) ? 26 : shift_amount_raw;

    // Extend mantissa for shifting to capture sticky bit
    wire [48:0] m_extended = {m_small, 25'b0};
    wire [48:0] m_shifted = m_extended >> shift_amount;
    
    wire sticky = |m_shifted[23:0];
    
    assign m_small_aligned = {m_shifted[48:25], sticky};

endmodule

module fp32_normalize (
    input         s_in,
    input  [7:0]  e_in,
    input  [25:0] sum_in,
    input         is_sub,
    output [31:0] z_out
);
    reg [7:0] e_out;
    reg [22:0] f_out;

    // For addition
    wire add_carry = !is_sub && sum_in[25];
    wire [24:0] add_sum_shifted = {sum_in[25], sum_in[24:1]}; // Right shift with rounding bit
    wire [7:0] e_add = e_in + 1;

    // For subtraction - Leading Zero Counter (Priority Encoder)
    reg [4:0] lead_zeros;
    always @(*) begin
        if      (sum_in[24]) lead_zeros = 0;
        else if (sum_in[23]) lead_zeros = 1;
        else if (sum_in[22]) lead_zeros = 2;
        else if (sum_in[21]) lead_zeros = 3;
        else if (sum_in[20]) lead_zeros = 4;
        else if (sum_in[19]) lead_zeros = 5;
        else if (sum_in[18]) lead_zeros = 6;
        else if (sum_in[17]) lead_zeros = 7;
        else if (sum_in[16]) lead_zeros = 8;
        else if (sum_in[15]) lead_zeros = 9;
        else if (sum_in[14]) lead_zeros = 10;
        else if (sum_in[13]) lead_zeros = 11;
        else if (sum_in[12]) lead_zeros = 12;
        else if (sum_in[11]) lead_zeros = 13;
        else if (sum_in[10]) lead_zeros = 14;
        else if (sum_in[9])  lead_zeros = 15;
        else if (sum_in[8])  lead_zeros = 16;
        else if (sum_in[7])  lead_zeros = 17;
        else if (sum_in[6])  lead_zeros = 18;
        else if (sum_in[5])  lead_zeros = 19;
        else if (sum_in[4])  lead_zeros = 20;
        else if (sum_in[3])  lead_zeros = 21;
        else if (sum_in[2])  lead_zeros = 22;
        else if (sum_in[1])  lead_zeros = 23;
        else                 lead_zeros = 24;
    end
    
    wire [24:0] sub_sum_shifted = sum_in[24:0] << lead_zeros;
    wire [7:0]  e_sub = e_in - lead_zeros;

    // Final selection
    always @(*) begin
        if (add_carry) begin // Addition overflow
            e_out = e_add;
            // Note: Simple truncation for rounding. For RNE, more logic is needed.
            f_out = add_sum_shifted[23:1];
        end else begin // Normal addition or subtraction
            e_out = is_sub ? e_sub : e_in;
            f_out = is_sub ? sub_sum_shifted[23:1] : sum_in[23:1];
        end
    end
    
    // Handle overflow/underflow to Inf/Zero
    assign z_out = (e_out >= 8'hFF) ? {s_in, 8'hFF, 23'h0} : // Overflow to Infinity
                   (e_out <= 8'h00)  ? {s_in, 8'h00, 23'h0} : // Underflow to Zero (simplified)
                   {s_in, e_out, f_out};

endmodule