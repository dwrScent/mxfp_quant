module mul_base_q2_comb
#(
    parameter Q_FRAC = 2,
    parameter QW     = 20
)
(
    input      [3:0]      w,
    input      [3:0]      x,
    output signed [QW-1:0] q_out
);

    function [QW-1:0] fp4_to_q2;
      input [3:0] a;
      reg sign;
      integer e_unb;
      reg mant;
      reg is_zero;
      reg signed [QW-1:0] base;
      begin
        sign    = a[3];
        e_unb   = {1'b0, a[2:1]} - 1;
        mant    = a[0];
        is_zero = (a[2:0] == 3'b000);
        if (is_zero) begin
          fp4_to_q2 = {QW{1'b0}};
        end else begin
          base = {{(QW-1){1'b0}}, 1'b1};
          base = base <<< Q_FRAC;
          if (mant) base = base + (base >>> 1);
          if (e_unb >= 0) base = base <<< e_unb;
          else            base = base >>> (-e_unb);
          if (sign) base = -base;
          fp4_to_q2 = base;
        end
      end
    endfunction

    function [QW-1:0] mul_base_q2;
      input [3:0] w_in;
      input [3:0] x_in;
      reg xs;
      integer xe_unb;
      reg xm;
      reg xzero;
      reg signed [QW-1:0] wq;
      reg signed [QW-1:0] t;
      begin
        xs     = x_in[3];
        xe_unb = {1'b0, x_in[2:1]} - 1;
        xm     = x_in[0];
        xzero  = (x_in[2:0] == 3'b000);
        if (xzero) begin
          mul_base_q2 = {QW{1'b0}};
        end else begin
          wq = fp4_to_q2(w_in);
          if (xe_unb >= 0) t = wq <<< xe_unb;
          else             t = wq >>> (-xe_unb);
          if (xm) t = t + (t >>> 1);
          if (xs) t = -t;
          mul_base_q2 = t;
        end
      end
    endfunction

    assign q_out = mul_base_q2(w, x);

endmodule
