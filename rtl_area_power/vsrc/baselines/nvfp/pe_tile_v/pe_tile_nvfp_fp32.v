// PE Tile for baseline NVFP (Verilog-2001, synthesizable).
// FP4 operands are E2M1. The tile has no subgroup scale path; it applies only
// the shared E4M3 group scale after the eight-way FP4 dot product.
module pe_tile_nvfp_fp32
#(
    parameter Q_FRAC = 2,
    parameter QW     = 20,
    parameter WW     = QW + 16
)
(
    input              clk,
    input              rst_n,
    input              in_valid,
    input      [31:0]  x_fp4,
    input      [31:0]  w_fp4,
    input      [7:0]   shared_scale_e4m3,
    input              psum_in_valid,
    input      [31:0]  psum_in_fp32,
    output             out_valid,
    output     [31:0]  psum_out_fp32
);

    integer si;
    genvar gi0;

    wire [3:0] x_bus [0:7];
    wire [3:0] w_bus [0:7];

    generate
      for (gi0=0; gi0<8; gi0=gi0+1) begin: UNPACK
        assign x_bus[gi0] = x_fp4[gi0*4+3 -: 4];
        assign w_bus[gi0] = w_fp4[gi0*4+3 -: 4];
      end
    endgenerate

    // -------------------- S0: input registers --------------------
    reg              v0;
    reg [3:0]        x0 [0:7];
    reg [3:0]        w0 [0:7];
    reg signed [7:0] k0;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v0 <= 1'b0;
        for (si=0; si<8; si=si+1) begin
          x0[si] <= 4'b0;
          w0[si] <= 4'b0;
        end
        k0 <= 8'b0;
      end else begin
        v0 <= in_valid;
        for (si=0; si<8; si=si+1) begin
          x0[si] <= x_bus[si];
          w0[si] <= w_bus[si];
        end
        k0 <= shared_scale_e4m3;
      end
    end

    // -------------------- S1: 8-way FP4 base products --------------------
    wire signed [QW-1:0] p1_comb [0:7];

    generate
      for (gi0=0; gi0<8; gi0=gi0+1) begin: GEN_MUL_BASE
        mul_base_q2_comb #(.Q_FRAC(Q_FRAC), .QW(QW)) U_MUL_BASE (
            .w(w0[gi0]),
            .x(x0[gi0]),
            .q_out(p1_comb[gi0])
        );
      end
    endgenerate

    reg v1;
    reg signed [QW-1:0] p1 [0:7];
    reg signed [7:0] k1;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v1 <= 1'b0;
        for (si=0; si<8; si=si+1)
          p1[si] <= {QW{1'b0}};
        k1 <= 8'b0;
      end else begin
        v1 <= v0;
        for (si=0; si<8; si=si+1)
          p1[si] <= p1_comb[si];
        k1 <= k0;
      end
    end

    // -------------------- S2-S4: adder tree --------------------
    reg v2, v3, v4;
    reg signed [QW-1:0] s2 [0:3];
    reg signed [QW-1:0] s3 [0:1];
    reg signed [QW-1:0] q_base;
    reg signed [7:0] k2, k3, k4;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v2 <= 1'b0;
        for (si=0; si<4; si=si+1)
          s2[si] <= {QW{1'b0}};
        k2 <= 8'b0;
      end else begin
        v2 <= v1;
        s2[0] <= p1[0] + p1[1];
        s2[1] <= p1[2] + p1[3];
        s2[2] <= p1[4] + p1[5];
        s2[3] <= p1[6] + p1[7];
        k2 <= k1;
      end
    end

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v3 <= 1'b0;
        s3[0] <= {QW{1'b0}};
        s3[1] <= {QW{1'b0}};
        k3 <= 8'b0;
      end else begin
        v3 <= v2;
        s3[0] <= s2[0] + s2[1];
        s3[1] <= s2[2] + s2[3];
        k3 <= k2;
      end
    end

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v4 <= 1'b0;
        q_base <= {QW{1'b0}};
        k4 <= 8'b0;
      end else begin
        v4 <= v3;
        q_base <= s3[0] + s3[1];
        k4 <= k3;
      end
    end

    // -------------------- S5: apply shared E4M3 group scale --------------------
    localparam signed [5:0] SCALE_BIAS       = 6'sd7;
    localparam signed [5:0] SCALE_NORM_SHIFT = SCALE_BIAS + 6'sd3;
    localparam signed [5:0] SCALE_SUBN_SHIFT = SCALE_BIAS + 6'sd2;

    wire        sc_sign = k4[7];
    wire [3:0]  sc_exp  = k4[6:3];
    wire [2:0]  sc_mant = k4[2:0];
    wire        sc_zero = (k4[6:0] == 7'b0);
    wire        sc_subn = (sc_exp == 4'b0000);

    wire signed [WW-1:0] q_ext = {{(WW-QW){q_base[QW-1]}}, q_base};
    wire signed [WW+4:0] q_ext_norm = {{5{q_ext[WW-1]}}, q_ext};
    wire signed [WW+4:0] zero_prod = {(WW+5){1'b0}};

    wire signed [WW+4:0] prod_norm =
          (q_ext_norm <<< 3)
        + (sc_mant[2] ? (q_ext_norm <<< 2) : zero_prod)
        + (sc_mant[1] ? (q_ext_norm <<< 1) : zero_prod)
        + (sc_mant[0] ?  q_ext_norm        : zero_prod);

    wire signed [WW+4:0] prod_sub =
          (sc_mant[2] ? (q_ext_norm <<< 2) : zero_prod)
        + (sc_mant[1] ? (q_ext_norm <<< 1) : zero_prod)
        + (sc_mant[0] ?  q_ext_norm        : zero_prod);

    wire signed [5:0] shift_norm = $signed({1'b0, sc_exp}) - SCALE_NORM_SHIFT;

    reg signed [WW-1:0] q_scaled_comb;

    always @(*) begin
      if (sc_zero) begin
        q_scaled_comb = {WW{1'b0}};
      end else if (sc_subn) begin
        q_scaled_comb = sc_sign ? -(prod_sub >>> SCALE_SUBN_SHIFT) : (prod_sub >>> SCALE_SUBN_SHIFT);
      end else if (!sc_sign) begin
        if (shift_norm >= 0)
          q_scaled_comb = prod_norm <<< shift_norm;
        else
          q_scaled_comb = prod_norm >>> (-shift_norm);
      end else begin
        if (shift_norm >= 0)
          q_scaled_comb = -(prod_norm <<< shift_norm);
        else
          q_scaled_comb = -(prod_norm >>> (-shift_norm));
      end
    end

    reg v5;
    reg signed [WW-1:0] q_scaled;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v5 <= 1'b0;
        q_scaled <= {WW{1'b0}};
      end else begin
        v5 <= v4;
        q_scaled <= q_scaled_comb;
      end
    end

    // -------------------- S6: fixed-point to FP32 --------------------
    reg v6;
    reg v7;
    wire [31:0] self_fp32;

    fxp_to_fp32 #(.W(WW), .FRAC(Q_FRAC)) U_Q_TO_FP32 (
        .din(q_scaled),
        .dout(self_fp32)
    );

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) v6 <= 1'b0;
      else        v6 <= v5;
    end

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) v7 <= 1'b0;
      else        v7 <= v6;
    end

    // -------------------- S7: FP32 accumulation --------------------
    reg [31:0] b_in_s8;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n)
        b_in_s8 <= 32'h00000000;
      else
        b_in_s8 <= psum_in_valid ? psum_in_fp32 : 32'h00000000;
    end

    wire [31:0] sum_s8;

    fp32_add U_SOFT_ADD (
        .a(self_fp32),
        .b(b_in_s8),
        .z(sum_s8)
    );

    reg v8;
    reg [31:0] psum_out_reg;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v8 <= 1'b0;
        psum_out_reg <= 32'h00000000;
      end else begin
        v8 <= v7;
        psum_out_reg <= sum_s8;
      end
    end

    assign out_valid = v8;
    assign psum_out_fp32 = psum_out_reg;

endmodule
