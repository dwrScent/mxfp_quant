module nvesm2_fp32_mul (
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] z
);
    wire        sign = a[31] ^ b[31];
    wire [7:0]  ea = a[30:23];
    wire [7:0]  eb = b[30:23];
    wire [22:0] fa = a[22:0];
    wire [22:0] fb = b[22:0];

    wire a_zero = (ea == 8'd0) && (fa == 23'd0);
    wire b_zero = (eb == 8'd0) && (fb == 23'd0);
    wire a_inf  = (ea == 8'hff) && (fa == 23'd0);
    wire b_inf  = (eb == 8'hff) && (fb == 23'd0);
    wire a_nan  = (ea == 8'hff) && (fa != 23'd0);
    wire b_nan  = (eb == 8'hff) && (fb != 23'd0);

    wire [23:0] ma = {ea != 8'd0, fa};
    wire [23:0] mb = {eb != 8'd0, fb};
    wire signed [10:0] exp_a = (ea == 8'd0) ? -11'sd126 : ($signed({1'b0, ea}) - 11'sd127);
    wire signed [10:0] exp_b = (eb == 8'd0) ? -11'sd126 : ($signed({1'b0, eb}) - 11'sd127);
    wire [47:0] prod = ma * mb;

    wire prod_ge_2 = prod[47];
    wire signed [11:0] exp_norm = exp_a + exp_b + (prod_ge_2 ? 12'sd1 : 12'sd0);
    wire [23:0] sig_pre = prod_ge_2 ? prod[47:24] : prod[46:23];
    wire guard = prod_ge_2 ? prod[23] : prod[22];
    wire sticky = prod_ge_2 ? (|prod[22:0]) : (|prod[21:0]);
    wire round_up = guard & (sticky | sig_pre[0]);
    wire [24:0] sig_round = {1'b0, sig_pre} + {24'd0, round_up};

    wire sig_carry = sig_round[24];
    wire signed [11:0] exp_round = exp_norm + (sig_carry ? 12'sd1 : 12'sd0);
    wire [22:0] frac_round = sig_carry ? sig_round[23:1] : sig_round[22:0];
    wire signed [11:0] exp_biased = exp_round + 12'sd127;

    assign z = (a_nan || b_nan || (a_zero && b_inf) || (b_zero && a_inf)) ? 32'h7fc00000 :
               (a_inf || b_inf) ? {sign, 8'hff, 23'd0} :
               (a_zero || b_zero || (prod == 48'd0)) ? {sign, 31'd0} :
               (exp_biased >= 12'sd255) ? {sign, 8'hff, 23'd0} :
               (exp_biased <= 12'sd0) ? {sign, 31'd0} :
               {sign, exp_biased[7:0], frac_round};
endmodule
