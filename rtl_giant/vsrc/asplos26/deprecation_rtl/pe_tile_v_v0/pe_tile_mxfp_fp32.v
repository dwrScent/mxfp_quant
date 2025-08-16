// =============================================================
// PE Tile for MXFP (Verilog-2001)
//   - 8× FP4(E2M1) MAC  + extra mantissa  + subgroup scale
//   - internal fixed-point: Q2 (exact for your value set)
//   - final output: IEEE-754 single (FP32)
// =============================================================

//`define USE_DW  // 若使用 DesignWare 的 DW_fp_add，请打开

module pe_tile_mxfp_fp32
#(
    parameter Q_FRAC = 2,         // Q2
    parameter QW     = 20,        // internal signed width
    parameter WW     = QW + 16    // wide width for shared-scale shifting
)
(
    input              clk,
    input              rst_n,
    input              in_valid,

    // 8×FP4(E2M1)（打包）：lane i 在 [4*i+3 : 4*i]
    input      [31:0]  x_fp4,
    input      [31:0]  w_fp4,

    input      [2:0]   idx_top,        // 0..7
    input      [1:0]   meta_extra,     // 00/01/10/11 -> 0/0.25/0.5/0.75
    input      [1:0]   sg_em,          // {+0.5*q, +0.25*q}

    input signed [7:0] shared_scale_e8m0, // ×2^k

    // 外部部分和（FP32）
    input              psum_in_valid,
    input      [31:0]  psum_in_fp32,

    output             out_valid,
    output     [31:0]  psum_out_fp32
);

    // --------- 把打包总线解成数组（仅模块内部使用）---------
    wire [3:0] x_bus [0:7];
    wire [3:0] w_bus [0:7];
    genvar gi0;
    generate
      for (gi0=0; gi0<8; gi0=gi0+1) begin: UNPACK
        assign x_bus[gi0] = x_fp4[gi0*4+3 -: 4];
        assign w_bus[gi0] = w_fp4[gi0*4+3 -: 4];
      end
    endgenerate

    // ====== FP4(E2M1) -> Q2（函数，移位+加法）=====
    function [QW-1:0] fp4_to_q2;
      input [3:0] a;
      reg sign;
      reg [1:0] e_st;
      integer e_unb;   // -1..2
      reg mant;
      reg is_zero;
      reg signed [QW-1:0] base;
      begin
        sign    = a[3];
        e_st    = a[2:1];
        e_unb   = {1'b0,a[2:1]} - 1; // 无偏指数
        mant    = a[0];
        is_zero = (a[2:0]==3'b000);

        if (is_zero) begin
          fp4_to_q2 = {QW{1'b0}};
        end else begin
          base = (1'b1 << Q_FRAC);      // 1.00 in Q2 = 4
          if (mant) base = base + (base >>> 1); // +0.5
          if (e_unb >= 0) base = base <<< e_unb;
          else            base = base >>> (-e_unb);
          if (sign) base = -base;
          fp4_to_q2 = base;
        end
      end
    endfunction

    // w * x_base（与 x 指数对齐，再乘 (1 or 1.5)）
    function [QW-1:0] mul_base_q2;
      input [3:0] w;
      input [3:0] x;
      reg xs; reg [1:0] xe_st; integer xe_unb; reg xm; reg xzero;
      reg signed [QW-1:0] wq, t;
      begin
        xs    = x[3];
        xe_st = x[2:1];
        xe_unb= {1'b0,x[2:1]} - 1;
        xm    = x[0];
        xzero = (x[2:0]==3'b000);

        if (xzero) begin
          mul_base_q2 = {QW{1'b0}};
        end else begin
          wq = fp4_to_q2(w);
          if (xe_unb >= 0) t = wq <<< xe_unb; else t = wq >>> (-xe_unb);
          if (xm) t = t + (t >>> 1);
          if (xs) t = -t;
          mul_base_q2 = t;
        end
      end
    endfunction

    // w * Δx（与 x 同符号、同指数，乘 {0.25,0.5,0.75}）
    function [QW-1:0] mul_extra_q2;
      input [3:0] w;
      input [3:0] x;
      input [1:0] m2;
      reg xs; reg [1:0] xe_st; integer xe_unb;
      reg signed [QW-1:0] wq, t;
      begin
        if (m2==2'b00) begin
          mul_extra_q2 = {QW{1'b0}};
        end else begin
          xs    = x[3];
          xe_st = x[2:1];
          xe_unb= {1'b0,x[2:1]} - 1;

          wq = fp4_to_q2(w);
          if (xe_unb >= 0) t = wq <<< xe_unb; else t = wq >>> (-xe_unb);

          case (m2)
            2'b01: t = t >>> 2;                 // *0.25
            2'b10: t = t >>> 1;                 // *0.5
            2'b11: t = (t >>> 1) + (t >>> 2);   // *0.75
            default: t = {QW{1'b0}};
          endcase

          if (xs) t = -t;
          mul_extra_q2 = t;
        end
      end
    endfunction

    // -------------------- S0：寄存输入 --------------------
    reg              v0;
    reg [3:0]        x0 [0:7];
    reg [3:0]        w0 [0:7];
    reg [2:0]        idx0;
    reg [1:0]        m2_0, sg0;
    reg signed [7:0] k0;

    integer si;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) v0 <= 1'b0;
      else        v0 <= in_valid;

      for (si=0; si<8; si=si+1) begin
        x0[si] <= x_bus[si];
        w0[si] <= w_bus[si];
      end
      idx0  <= idx_top;
      m2_0  <= meta_extra;
      sg0   <= sg_em;
      k0    <= shared_scale_e8m0;
    end

    // -------------------- S1：8 路 w*x_base --------------------
    reg              v1;
    reg signed [QW-1:0] p1 [0:7];
    always @(posedge clk or negedge rst_n) begin
      integer i;
      if (!rst_n) begin
        v1 <= 1'b0;
        for (i=0;i<8;i=i+1) p1[i] <= {QW{1'b0}};
      end else begin
        v1 <= v0;
        for (i=0;i<8;i=i+1) p1[i] <= mul_base_q2(w0[i], x0[i]);
      end
    end

    reg [3:0] x1 [0:7];
    reg [3:0] w1 [0:7];
    reg [2:0] idx1; reg [1:0] m2_1, sg1; reg signed [7:0] k1;
    always @(posedge clk or negedge rst_n) begin
      integer j;
      if (!rst_n) begin
        idx1<=3'd0; m2_1<=2'b0; sg1<=2'b0; k1<=8'sd0;
      end else begin
        for (j=0;j<8;j=j+1) begin x1[j]<=x0[j]; w1[j]<=w0[j]; end
        idx1<=idx0; m2_1<=m2_0; sg1<=sg0; k1<=k0;
      end
    end

    // -------------------- S2/S3/S4：加法树 -> q_base --------------------
    reg              v2; reg signed [QW-1:0] s2 [0:3];
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin v2<=1'b0; s2[0]<={QW{1'b0}}; s2[1]<={QW{1'b0}}; s2[2]<={QW{1'b0}}; s2[3]<={QW{1'b0}}; end
      else begin
        v2<=v1;
        s2[0] <= p1[0]+p1[1]; s2[1] <= p1[2]+p1[3];
        s2[2] <= p1[4]+p1[5]; s2[3] <= p1[6]+p1[7];
      end
    end

    reg              v3; reg signed [QW-1:0] s3 [0:1];
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin v3<=1'b0; s3[0]<={QW{1'b0}}; s3[1]<={QW{1'b0}}; end
      else begin v3<=v2; s3[0]<=s2[0]+s2[1]; s3[1]<=s2[2]+s2[3]; end
    end

    reg              v4; reg signed [QW-1:0] q_base;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin v4<=1'b0; q_base<={QW{1'b0}}; end
      else begin v4<=v3; q_base<=s3[0]+s3[1]; end
    end

    // -------------------- S4：extra 路 + 合并 --------------------
    reg signed [QW-1:0] q_extra, q4;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin q_extra<={QW{1'b0}}; q4<={QW{1'b0}}; end
      else begin
        q_extra <= mul_extra_q2(w1[idx1], x1[idx1], m2_1);
        q4      <= q_base + q_extra;
      end
    end
    reg [1:0] sg4; reg signed [7:0] k4; reg v4p;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin v4p<=1'b0; sg4<=2'b0; k4<=8'sd0; end
      else begin v4p<=v4; sg4<=sg1; k4<=k1; end
    end

    // -------------------- S5：subgroup scale --------------------
    reg              v5;
    reg signed [QW-1:0] q5;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin v5<=1'b0; q5<={QW{1'b0}}; end
      else begin
        v5 <= v4p;
        q5 <= q4 + (sg4[1] ? (q4 >>> 1) : {QW{1'b0}})  // +0.5*q
                + (sg4[0] ? (q4 >>> 2) : {QW{1'b0}});  // +0.25*q
      end
    end
    reg signed [7:0] k5;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) k5<=8'sd0; else k5<=k4;
    end

    // -------------------- S6：shared_scale (=2^k)（宽位宽）--------------------
    reg              v6;
    reg signed [WW-1:0] q6_wide;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin v6<=1'b0; q6_wide<={WW{1'b0}}; end
      else begin
        v6 <= v5;
        if (k5 >= 0) q6_wide <= {{(WW-QW){q5[QW-1]}}, q5} <<< k5;
        else         q6_wide <= {{(WW-QW){q5[QW-1]}}, q5} >>> (-k5);
      end
    end

    // -------------------- S7：Q2 -> FP32 --------------------
    reg        v7;
    wire [31:0] self_fp32;
    fxp_to_fp32 #(.W(WW), .FRAC(Q_FRAC)) U_Q2_TO_FP32 (.din(q6_wide), .dout(self_fp32));
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) v7 <= 1'b0; else v7 <= v6;
    end

    // -------------------- S8：FP32 累加 --------------------
`ifdef USE_DW
    wire [31:0] b_in = psum_in_valid ? psum_in_fp32 : 32'h00000000;
    wire [7:0]  dw_status;
    DW_fp_add #(23,8,0) U_DW_ADD (.a(self_fp32), .b(b_in), .rnd(3'b000), .z(psum_out_fp32), .status(dw_status));
    assign out_valid = v7;
`else
    reg [31:0] b_in_s8;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) b_in_s8 <= 32'h00000000;
      else        b_in_s8 <= psum_in_valid ? psum_in_fp32 : 32'h00000000;
    end
    wire [31:0] sum_s8;
    fp32_add U_SOFT_ADD (.a(self_fp32), .b(b_in_s8), .z(sum_s8));
    assign psum_out_fp32 = sum_s8;
    assign out_valid     = v7;
`endif

endmodule
