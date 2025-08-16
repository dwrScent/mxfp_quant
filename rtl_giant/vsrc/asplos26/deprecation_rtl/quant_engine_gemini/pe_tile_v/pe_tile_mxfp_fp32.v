// PE Tile for MXFP (Verilog-2001, Synthesizable Version)
module pe_tile_mxfp_fp32
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
    input      [2:0]   idx_top,
    input      [1:0]   meta_extra,
    input      [1:0]   sg_em,
    input signed [7:0] shared_scale_e8m0,
    input              psum_in_valid,
    input      [31:0]  psum_in_fp32,
    output             out_valid,
    output     [31:0]  psum_out_fp32
);

    integer si;
    genvar gi0;

    // --------- Unpack Bus (Combinational) ---------
    wire [3:0] x_bus [0:7];
    wire [3:0] w_bus [0:7];
    generate
      for (gi0=0; gi0<8; gi0=gi0+1) begin: UNPACK
        assign x_bus[gi0] = x_fp4[gi0*4+3 -: 4];
        assign w_bus[gi0] = w_fp4[gi0*4+3 -: 4];
      end
    endgenerate

    // ================== PIPELINE STAGES ==================

    // -------------------- S0: Input Registers --------------------
    reg              v0;
    reg [3:0]        x0 [0:7];
    reg [3:0]        w0 [0:7];
    reg [2:0]        idx0;
    reg [1:0]        m2_0, sg0;
    reg signed [7:0] k0;

    // [FIX] Corrected always block structure with full reset logic
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v0 <= 1'b0;
        for (si=0; si<8; si=si+1) begin
          x0[si] <= 4'b0;
          w0[si] <= 4'b0;
        end
        idx0  <= 3'b0;
        m2_0  <= 2'b0;
        sg0   <= 2'b0;
        k0    <= 8'b0;
      end else begin
        v0 <= in_valid;
        for (si=0; si<8; si=si+1) begin
          x0[si] <= x_bus[si];
          w0[si] <= w_bus[si];
        end
        idx0  <= idx_top;
        m2_0  <= meta_extra;
        sg0   <= sg_em;
        k0    <= shared_scale_e8m0;
      end
    end

    // -------------------- S1: 8-way Parallel Multiply (w*x_base) --------------------
    wire signed [QW-1:0] p1_comb [0:7];
    generate
        for (gi0=0; gi0<8; gi0=gi0+1) begin: GEN_MUL_BASE
            mul_base_q2_comb #(.Q_FRAC(Q_FRAC), .QW(QW)) u_mul_base (
                .w(w0[gi0]), .x(x0[gi0]), .q_out(p1_comb[gi0])
            );
        end
    endgenerate

    reg v1;
    reg signed [QW-1:0] p1 [0:7];
    // Pipeline registers for passing data to the next stage
    reg [3:0] x1 [0:7];
    reg [3:0] w1 [0:7];
    reg [2:0] idx1;
    reg [1:0] m2_1, sg1;
    reg signed [7:0] k1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            for (si=0; si<8; si=si+1) begin
                p1[si] <= {QW{1'b0}};
                x1[si] <= 4'b0;
                w1[si] <= 4'b0;
            end
            idx1 <= 3'b0;
            m2_1 <= 2'b0;
            sg1  <= 2'b0;
            k1   <= 8'b0;
        end else begin
            v1 <= v0;
            for (si=0; si<8; si=si+1) begin
                p1[si] <= p1_comb[si];
                x1[si] <= x0[si];
                w1[si] <= w0[si];
            end
            idx1 <= idx0;
            m2_1 <= m2_0;
            sg1  <= sg0;
            k1   <= k0;
        end
    end

    // -------------------- S2-S4: Adder Tree for q_base --------------------
    reg v2, v3, v4;
    reg signed [QW-1:0] s2 [0:3];
    reg signed [QW-1:0] s3 [0:1];
    reg signed [QW-1:0] q_base;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v2 <= 1'b0;
        for (si=0; si<4; si=si+1) s2[si] <= {QW{1'b0}};
      end else begin
        v2    <= v1;
        s2[0] <= p1[0] + p1[1];
        s2[1] <= p1[2] + p1[3];
        s2[2] <= p1[4] + p1[5];
        s2[3] <= p1[6] + p1[7];
      end
    end

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v3 <= 1'b0;
        s3[0] <= {QW{1'b0}};
        s3[1] <= {QW{1'b0}};
      end else begin
        v3    <= v2;
        s3[0] <= s2[0] + s2[1];
        s3[1] <= s2[2] + s2[3];
      end
    end

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v4 <= 1'b0;
        q_base <= {QW{1'b0}};
      end else begin
        v4 <= v3;
        q_base <= s3[0] + s3[1];
      end
    end

    // -------------------- S4: Calculate Extra Path and Combine --------------------
    wire signed [QW-1:0] q_extra_comb;
    mul_extra_q2_comb #(.Q_FRAC(Q_FRAC), .QW(QW)) u_mul_extra (
        .w(w1[idx1]), .x(x1[idx1]), .m2(m2_1), .q_out(q_extra_comb)
    );

    wire signed [QW-1:0] q4_comb = q_base + q_extra_comb;

    reg v4p; // Renamed to avoid confusion with v4
    reg signed [QW-1:0] q4;
    reg [1:0] sg4;
    reg signed [7:0] k4;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v4p <= 1'b0;
        q4  <= {QW{1'b0}};
        sg4 <= 2'b0;
        k4  <= 8'b0;
      end else begin
        v4p <= v4;
        q4  <= q4_comb;
        sg4 <= sg1;
        k4  <= k1;
      end
    end

    // -------------------- S5: Subgroup Scale --------------------
    wire signed [QW-1:0] q5_comb = q4 + (sg4[1] ? (q4 >>> 1) : {QW{1'b0}})
                                      + (sg4[0] ? (q4 >>> 2) : {QW{1'b0}});
    reg v5;
    reg signed [QW-1:0] q5;
    reg signed [7:0] k5;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v5 <= 1'b0;
        q5 <= {QW{1'b0}};
        k5 <= 8'b0;
      end else begin
        v5 <= v4p;
        q5 <= q5_comb;
        k5 <= k4;
      end
    end

    // -------------------- S6: Apply Shared Scale (2^k) --------------------
    reg v6;
    reg signed [WW-1:0] q6_wide;

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        v6 <= 1'b0;
        q6_wide <= {WW{1'b0}};
      end else begin
        v6 <= v5;
        // This variable shift is inside a clocked block, which is synthesizable
        if (k5 >= 0) q6_wide <= {{(WW-QW){q5[QW-1]}}, q5} <<< k5;
        else         q6_wide <= {{(WW-QW){q5[QW-1]}}, q5} >>> (-k5);
      end
    end

    // -------------------- S7: Convert Q -> FP32 --------------------
    reg v7;
    wire [31:0] self_fp32;
    fxp_to_fp32 #(.W(WW), .FRAC(Q_FRAC)) U_Q2_TO_FP32 (.din(q6_wide), .dout(self_fp32));

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) v7 <= 1'b0;
      else        v7 <= v6;
    end
    
    // -------------------- S8: FP32 Accumulation --------------------
    // Use the non-DW version for better portability
    reg [31:0] b_in_s8;
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) b_in_s8 <= 32'h00000000;
      else        b_in_s8 <= psum_in_valid ? psum_in_fp32 : 32'h00000000;
    end
    
    wire [31:0] sum_s8;
    fp32_add U_SOFT_ADD (.a(self_fp32), .b(b_in_s8), .z(sum_s8));
    
    // Final output assignment
    // The result of the FP32 add is combinational, so we register the final output
    // to avoid a long path from the S7 register through the adder.
    reg v8;
    reg [31:0] psum_out_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v8 <= 1'b0;
            psum_out_reg <= 32'h0;
        end else begin
            v8 <= v7;
            psum_out_reg <= sum_s8;
        end
    end

    assign out_valid     = v8;
    assign psum_out_fp32 = psum_out_reg;

endmodule
