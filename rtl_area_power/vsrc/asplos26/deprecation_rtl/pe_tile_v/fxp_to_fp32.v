// =============================================================
// Signed fixed-point -> IEEE-754 single (Verilog-2001 compatible FIX v3)
// =============================================================
module fxp_to_fp32
#( parameter W=32, parameter FRAC=2 )
(
    input  signed [W-1:0] din,
    output       [31:0]   dout
);
    localparam EB=8, MB=23, BIAS=127;

    wire s = din[W-1];
    wire [W-1:0] abs_in = s ? (~din + 1'b1) : din;
    wire is_zero = (abs_in == {W{1'b0}});

    // --- Internal variables ---
    integer k;
    integer msb;
    integer exp_unb;
    reg [EB-1:0] E;
    reg [MB-1:0] F;
    reg [31:0] dout_r;
    
    // For normalization path
    reg [22:0] frac_pre;
    reg guard, roundb, sticky;
    reg [24:0] m_norm;
    reg [7:0]  e_norm;
    reg add_round;
    reg carry_norm;
    integer shift;
    
    // [FIX] Added a temporary register for shift results
    reg [W-1:0] temp_shifted_val;

    // For subnormal path
    integer rshift;
    reg [W+24:0] tmp;

    // MSB index finder
    always @(*) begin
        msb = -1;
        for (k=W-1; k>=0; k=k-1) begin
            if (abs_in[k]) begin 
                msb = k; 
            end
        end
    end

    // Unbiased exponent calculation
    always @(*) begin
        exp_unb = is_zero ? 0 : (msb - FRAC);
    end

    // Main conversion logic
    always @(*) begin
        shift = msb - 23; 

        if (is_zero) begin
            dout_r = 32'h00000000;
        end else if (exp_unb > 127) begin
            dout_r = {s, 8'hFF, 23'h0}; // Overflow -> +/-Inf
        end else if (exp_unb >= -126) begin
            // Normal numbers
            if (shift >= 0) begin
                // [FIX] Perform shift and select in two steps
                temp_shifted_val = abs_in >> shift;
                frac_pre = temp_shifted_val[22:0];
                
                guard    = (shift>0) ? abs_in[shift-1] : 1'b0;
                roundb   = (shift>1) ? abs_in[shift-2] : 1'b0;
                sticky   = (shift>2) ? |abs_in[shift-3:0] : 1'b0;
            end else begin
                // [FIX] Perform shift and select in two steps
                temp_shifted_val = abs_in << (-shift);
                frac_pre = temp_shifted_val[22:0];

                guard=1'b0; roundb=1'b0; sticky=1'b0;
            end
            
            add_round = guard & (roundb | sticky | frac_pre[0]);
            {carry_norm, m_norm} = {1'b0,1'b1,frac_pre} + add_round;

            if (carry_norm) begin
                e_norm = exp_unb + BIAS + 1;
            end else begin
                e_norm = exp_unb + BIAS;
            end

            if (e_norm==8'hFF) begin
                dout_r = {s, 8'hFF, 23'h0}; // Rounded up to Infinity
            end else begin
                F = m_norm[22:0];
                E = e_norm[7:0];
                dout_r = {s,E,F};
            end
        end else begin
            // Subnormal numbers path
            rshift = (-126 - exp_unb);
            tmp = {abs_in, 25'b0} >> rshift;
            frac_pre = tmp[W+24 : W+2];
            guard    = tmp[W+1];
            roundb   = tmp[W+0];
            sticky   = |tmp[W-1:0];

            add_round = guard & (roundb | sticky | frac_pre[0]);
            {carry_norm, m_norm} = {1'b0,1'b0,frac_pre} + add_round;
            if (carry_norm) begin
                 dout_r = {s, 8'h01, m_norm[22:0]}; // Promoted to normal number
            end else begin
                 dout_r = {s, 8'h00, m_norm[22:0]};
            end
        end
    end

    assign dout = dout_r;
endmodule