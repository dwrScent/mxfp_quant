// =============================================================
// IEEE-754 single-precision adder (combinational, RNE, simplified specials)
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

    always @(*) begin
        if (a_nan) z_r = a;
        else if (b_nan) z_r = b;
        else if (a_inf && b_inf && (sa^sb)) z_r = 32'h7FC00000; // Inf + (-Inf) -> NaN
        else begin
            // build mantissa with hidden-1 if normal
            reg [24:0] ma, mb;
            reg [7:0]  e_big, e_small;
            reg        s_big, s_small;
            reg [24:0] m_big, m_small;
            reg [7:0]  de;
            reg [49:0] tmp_small;
            reg [24:0] m_small_r;
            reg sticky;
            reg [25:0] sum;
            reg [7:0]  e_norm;
            reg [24:0] m_norm;
            reg guard, roundb;
            reg add_r;

            ma = (ea==0) ? {1'b0,1'b0,fa} : {1'b0,1'b1,fa};
            mb = (eb==0) ? {1'b0,1'b0,fb} : {1'b0,1'b1,fb};

            if (ea >= eb) begin
                e_big=ea; m_big=ma; s_big=sa;
                e_small=eb; m_small=mb; s_small=sb;
            end else begin
                e_big=eb; m_big=mb; s_big=sb;
                e_small=ea; m_small=ma; s_small=sa;
            end
            de = e_big - e_small;

            tmp_small = {m_small,25'b0} >> de;
            m_small_r = tmp_small[49:25];
            sticky    = |tmp_small[24:0];

            if (s_big==s_small) sum = {1'b0,m_big} + {1'b0,m_small_r};
            else                sum = {1'b0,m_big} - {1'b0,m_small_r};

            if (sum[25]) begin
                m_norm = sum[25:1];
                guard  = sum[0];
                roundb = sticky;
                add_r  = guard & (roundb | m_norm[0]);
                { /*carry*/ , m_norm } = {1'b0,m_norm} + add_r;
                e_norm = e_big + (add_r && m_norm[24]);
            end else begin
                integer sh, kk;
                sh = 0;
                for (kk=24; kk>=0; kk=kk-1) begin
                    if (sum[kk]) begin sh = 24-kk; kk=-1; end
                end
                m_norm = sum[24:0] << sh;
                e_norm = (e_big > sh) ? (e_big - sh) : 8'd0;
                guard  = 1'b0; roundb = sticky; add_r = guard & (roundb | m_norm[0]);
                { /*carry*/ , m_norm } = {1'b0,m_norm} + add_r;
            end

            if (a_inf) z_r = {sa,8'hFF,23'h0};
            else if (b_inf) z_r = {sb,8'hFF,23'h0};
            else if (a_zero && b_zero) z_r = {sa&sb,8'h00,23'h0};
            else if (e_norm==8'hFF) z_r = {s_big,8'hFF,23'h0};
            else z_r = {s_big, e_norm, m_norm[22:0]};
        end
    end

    assign z = z_r;
endmodule
