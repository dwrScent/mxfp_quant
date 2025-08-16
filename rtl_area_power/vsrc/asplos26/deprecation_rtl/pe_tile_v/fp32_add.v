// =============================================================
// IEEE-754 single-precision adder (Verilog-2001 compatible FIX v2)
// =============================================================
module fp32_add(
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] z
);
    wire sa = a[31]; wire [7:0] ea = a[30:23]; wire [22:0] fa = a[22:0];
    wire sb = b[31]; wire [7:0] eb = b[30:23]; wire [22:0] fb = b[22:0];

    wire a_nan = (ea==8'hFF) && (fa!=0);
    wire b_nan = (eb==8'hFF) && (fb!=0);
    wire a_inf = (ea==8'hFF) && (fa==0);
    wire b_inf = (eb==8'hFF) && (fb==0);
    wire a_zero= (ea==0) && (fa==0);
    wire b_zero= (eb==0) && (fb==0);

    reg [31:0] z_r;

    // --- Internal variables for the always block ---
    reg [24:0] ma, mb;
    reg [7:0]  e_big, e_small;
    reg        s_big;
    reg [24:0] m_big, m_small;
    reg [7:0]  de;
    reg [50:0] m_small_shifted;
    reg [24:0] m_small_r;
    reg        sticky;
    reg [25:0] sum;
    reg [7:0]  e_res;
    reg [24:0] m_res;
    reg        s_res;
    integer    sh;
    integer    k;

    always @(*) begin
        if (a_nan) z_r = a;
        else if (b_nan) z_r = b;
        else if (a_inf && b_inf && (sa^sb)) z_r = 32'h7FC00000; // NaN
        else if (a_inf) z_r = a;
        else if (b_inf) z_r = b;
        else if (a_zero && b_zero) z_r = (sa & sb) ? 32'h80000000 : 32'h0;
        else if (a_zero) z_r = b;
        else if (b_zero) z_r = a;
        else begin
            // Step 1: Initialization & Unpacking
            ma = {1'b1, fa}; // Normal numbers have a hidden 1
            mb = {1'b1, fb};

            // Step 2: Swap to find larger exponent
            if (ea > eb) begin
                e_big   = ea; e_small = eb;
                m_big   = ma; m_small = mb;
                s_big   = sa;
            end else if (eb > ea) begin
                e_big   = eb; e_small = ea;
                m_big   = mb; m_small = ma;
                s_big   = sb;
            end else begin // Exponents are equal, check mantissa
                e_big = ea; e_small = eb;
                if (fa >= fb) begin
                    m_big   = ma; m_small = mb;
                    s_big   = sa;
                end else begin
                    m_big   = mb; m_small = ma;
                    s_big   = sb;
                end
            end

            // Step 3: Align mantissas
            de = e_big - e_small;
            m_small_shifted = {m_small, 26'b0};
            if (de > 0) m_small_shifted = m_small_shifted >> de;
            
            m_small_r = m_small_shifted[50:26];
            sticky    = |m_small_shifted[25:0];

            // Step 4: Add or Subtract
            s_res = s_big;
            if (sa == sb) begin // Addition
                sum = {1'b0, m_big} + {1'b0, m_small_r};
                e_res = e_big;
            end else begin // Subtraction
                sum = {1'b0, m_big} - {1'b0, m_small_r};
                e_res = e_big;
            end
            
            // Step 5: Normalize result
            if (sum == 0) begin
                z_r = 32'h0;
            end else begin
                if (sum[25]) begin // Overflow (for addition)
                    m_res = sum[25:1];
                    e_res = e_res + 1;
                end else begin // Potential underflow (for subtraction)
                    sh = 0;
                    for(k=24; k>=0; k=k-1) begin
                        if(sum[k]) begin
                           sh = 24-k;
                           k = -1; // break
                        end
                    end
                    m_res = sum << sh;
                    e_res = e_res - sh;
                end
                
                // Final Packing
                if (e_res >= 255) z_r = {s_res, 8'hFF, 23'h0}; // To Infinity
                else if (e_res <= 0) z_r = {s_res, 8'h00, 23'h0}; // To Zero (simplified)
                else z_r = {s_res, e_res, m_res[22:0]};
            end
        end
    end

    assign z = z_r;
endmodule