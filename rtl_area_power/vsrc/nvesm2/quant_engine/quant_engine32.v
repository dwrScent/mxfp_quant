// =============================================================
// NVESM2 Quantization Engine
//
// Flow for one 32-lane block:
//   1. Build one E4M3 group scale for each 16-lane group.
//   2. Reuse the 32 lane RQUs for ES candidates 0, 1, 2, 3 in four
//      consecutive cycles.  The lane pipelines overlap these candidates.
//   3. Sum each candidate's eight lane costs inside every subgroup.
//   4. Pick the minimum-cost ES per subgroup.
//   5. Publish group scales, selected ES metadata, and matching FP4 values.
//
// Input is accepted only when in_valid and in_ready are high.
// =============================================================
module quant_engine32 (
    input                 clk,
    input                 rst_n,
    input                 in_valid,
    input  [32*32-1:0]    in_fp32_bus,
    output                in_ready,
    output reg            out_valid,
    output reg [2*8-1:0]  group_scale_bus,
    output reg [32*4-1:0] fp4_bus,
    output reg [4*2-1:0]  es_idx_bus
);

    localparam ST_IDLE       = 2'd0;
    localparam ST_WAIT_SCALE = 2'd1;
    localparam ST_ISSUE      = 2'd2;
    localparam ST_DRAIN      = 2'd3;

    integer ii;
    genvar gi;
    genvar gs;

    reg [1:0]        state;
    reg [1:0]        issue_es;
    reg [32*32-1:0]  fp32_hold;
    reg [7:0]        group_scale_g0;
    reg [7:0]        group_scale_g1;
    reg [2*8-1:0]    group_scale_hold;

    assign in_ready = (state == ST_IDLE);
    wire accept_input = in_valid && in_ready;

    // -------------------- S0: 16-lane group scales --------------------
    wire [7:0] group_scale_e4m3_g0;
    wire [7:0] group_scale_e4m3_g1;
    wire       group_scale_valid_g0;
    wire       group_scale_valid_g1;
    wire       group_scale_done = group_scale_valid_g0 && group_scale_valid_g1;

    nvesm2_group_scale U_GROUP_SCALE_G0 (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_valid        (accept_input),
        .fp32_bus        (in_fp32_bus[16*32-1:0]),
        .out_valid       (group_scale_valid_g0),
        .group_scale_e4m3(group_scale_e4m3_g0)
    );

    nvesm2_group_scale U_GROUP_SCALE_G1 (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_valid        (accept_input),
        .fp32_bus        (in_fp32_bus[32*32-1:16*32]),
        .out_valid       (group_scale_valid_g1),
        .group_scale_e4m3(group_scale_e4m3_g1)
    );

    // -------------------- S1-S5: shared lane RQU pipeline --------------------
    wire        lane_issue_valid = (state == ST_ISSUE);
    wire [1:0]  lane_issue_es    = issue_es;

    wire [3:0]  lane_fp4       [0:31];
    wire [13:0] lane_cost      [0:31];
    wire [1:0]  lane_es_out    [0:31];
    wire [31:0] lane_valid_bus;
    wire [32*4-1:0] lane_fp4_bus;

    generate
      for (gi=0; gi<32; gi=gi+1) begin: GEN_LANE
        nvesm2_quant_lane U_LANE (
            .clk         (clk),
            .rst_n       (rst_n),
            .in_valid    (lane_issue_valid),
            .x_fp32      (fp32_hold[gi*32 + 31 : gi*32]),
            .group_scale ((gi < 16) ? group_scale_g0 : group_scale_g1),
            .es_idx      (lane_issue_es),
            .out_valid   (lane_valid_bus[gi]),
            .es_idx_out  (lane_es_out[gi]),
            .fp4         (lane_fp4[gi]),
            .cost        (lane_cost[gi])
        );

        assign lane_fp4_bus[gi*4 + 3 -: 4] = lane_fp4[gi];
      end
    endgenerate

    wire       lane_out_valid = &lane_valid_bus;
    wire [1:0] lane_out_es    = lane_es_out[0];

    reg [32*4-1:0] fp4_es0;
    reg [32*4-1:0] fp4_es1;
    reg [32*4-1:0] fp4_es2;
    reg [32*4-1:0] fp4_es3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fp4_es0 <= {32*4{1'b0}};
            fp4_es1 <= {32*4{1'b0}};
            fp4_es2 <= {32*4{1'b0}};
            fp4_es3 <= {32*4{1'b0}};
        end else if (lane_out_valid) begin
            case (lane_out_es)
                2'd0: fp4_es0 <= lane_fp4_bus;
                2'd1: fp4_es1 <= lane_fp4_bus;
                2'd2: fp4_es2 <= lane_fp4_bus;
                default: fp4_es3 <= lane_fp4_bus;
            endcase
        end
    end

    // -------------------- S6-S8: eight-lane subgroup cost accumulation --------------------
    wire [3:0]        accum_valid_bus;
    wire [1:0]        accum_es     [0:3];
    wire [16:0]       accum_cost   [0:3];
    wire [4*17-1:0]   accum_cost_bus;

    generate
      for (gs=0; gs<4; gs=gs+1) begin: GEN_SUBGROUP_ACCUM
        nvesm2_subgroup_accum U_ACCUM (
            .clk           (clk),
            .rst_n         (rst_n),
            .in_valid      (lane_out_valid),
            .es_idx        (lane_out_es),
            .cost0         (lane_cost[gs*8 + 0]),
            .cost1         (lane_cost[gs*8 + 1]),
            .cost2         (lane_cost[gs*8 + 2]),
            .cost3         (lane_cost[gs*8 + 3]),
            .cost4         (lane_cost[gs*8 + 4]),
            .cost5         (lane_cost[gs*8 + 5]),
            .cost6         (lane_cost[gs*8 + 6]),
            .cost7         (lane_cost[gs*8 + 7]),
            .out_valid     (accum_valid_bus[gs]),
            .es_idx_out    (accum_es[gs]),
            .subgroup_cost (accum_cost[gs])
        );

        assign accum_cost_bus[gs*17 + 16 -: 17] = accum_cost[gs];
      end
    endgenerate

    wire       accum_out_valid = &accum_valid_bus;
    wire [1:0] accum_out_es    = accum_es[0];

    reg [4*17-1:0] cost_es0;
    reg [4*17-1:0] cost_es1;
    reg [4*17-1:0] cost_es2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cost_es0 <= {4*17{1'b0}};
            cost_es1 <= {4*17{1'b0}};
            cost_es2 <= {4*17{1'b0}};
        end else if (accum_out_valid) begin
            case (accum_out_es)
                2'd0: cost_es0 <= accum_cost_bus;
                2'd1: cost_es1 <= accum_cost_bus;
                2'd2: cost_es2 <= accum_cost_bus;
                default: ;
            endcase
        end
    end

    // -------------------- S9-S10: per-subgroup argmin --------------------
    reg        argmin_v1;
    reg [16:0] win01_cost [0:3];
    reg [16:0] win23_cost [0:3];
    reg [1:0]  win01_es   [0:3];
    reg [1:0]  win23_es   [0:3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            argmin_v1 <= 1'b0;
            for (ii=0; ii<4; ii=ii+1) begin
                win01_cost[ii] <= 17'd0;
                win23_cost[ii] <= 17'd0;
                win01_es[ii] <= 2'd0;
                win23_es[ii] <= 2'd0;
            end
        end else begin
            argmin_v1 <= accum_out_valid && (accum_out_es == 2'd3);

            if (accum_out_valid && (accum_out_es == 2'd3)) begin
                for (ii=0; ii<4; ii=ii+1) begin
                    if (cost_es1[ii*17 + 16 -: 17] < cost_es0[ii*17 + 16 -: 17]) begin
                        win01_cost[ii] <= cost_es1[ii*17 + 16 -: 17];
                        win01_es[ii] <= 2'd1;
                    end else begin
                        win01_cost[ii] <= cost_es0[ii*17 + 16 -: 17];
                        win01_es[ii] <= 2'd0;
                    end

                    if (accum_cost_bus[ii*17 + 16 -: 17] < cost_es2[ii*17 + 16 -: 17]) begin
                        win23_cost[ii] <= accum_cost_bus[ii*17 + 16 -: 17];
                        win23_es[ii] <= 2'd3;
                    end else begin
                        win23_cost[ii] <= cost_es2[ii*17 + 16 -: 17];
                        win23_es[ii] <= 2'd2;
                    end
                end
            end
        end
    end

    reg [1:0] best_es [0:3];

    always @(*) begin
        for (ii=0; ii<4; ii=ii+1) begin
            if (win23_cost[ii] < win01_cost[ii])
                best_es[ii] = win23_es[ii];
            else
                best_es[ii] = win01_es[ii];
        end
    end

    // -------------------- Control and output publication --------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            issue_es <= 2'd0;
            fp32_hold <= {32*32{1'b0}};
            group_scale_g0 <= 8'h00;
            group_scale_g1 <= 8'h00;
            group_scale_hold <= {2*8{1'b0}};
            out_valid <= 1'b0;
            group_scale_bus <= {2*8{1'b0}};
            fp4_bus <= {32*4{1'b0}};
            es_idx_bus <= {4*2{1'b0}};
        end else begin
            out_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (accept_input) begin
                        state <= ST_WAIT_SCALE;
                        issue_es <= 2'd0;
                        fp32_hold <= in_fp32_bus;
                    end
                end

                ST_WAIT_SCALE: begin
                    if (group_scale_done) begin
                        state <= ST_ISSUE;
                        issue_es <= 2'd0;
                        group_scale_g0 <= group_scale_e4m3_g0;
                        group_scale_g1 <= group_scale_e4m3_g1;
                        group_scale_hold <= {group_scale_e4m3_g1, group_scale_e4m3_g0};
                    end
                end

                ST_ISSUE: begin
                    issue_es <= issue_es + 2'd1;
                    if (issue_es == 2'd3)
                        state <= ST_DRAIN;
                end

                default: begin
                    if (argmin_v1) begin
                        state <= ST_IDLE;
                        issue_es <= 2'd0;
                        out_valid <= 1'b1;
                        group_scale_bus <= group_scale_hold;

                        for (ii=0; ii<4; ii=ii+1)
                            es_idx_bus[ii*2 + 1 -: 2] <= best_es[ii];

                        for (ii=0; ii<32; ii=ii+1) begin
                            case (best_es[ii/8])
                                2'd0: fp4_bus[ii*4 + 3 -: 4] <= fp4_es0[ii*4 + 3 -: 4];
                                2'd1: fp4_bus[ii*4 + 3 -: 4] <= fp4_es1[ii*4 + 3 -: 4];
                                2'd2: fp4_bus[ii*4 + 3 -: 4] <= fp4_es2[ii*4 + 3 -: 4];
                                default: fp4_bus[ii*4 + 3 -: 4] <= fp4_es3[ii*4 + 3 -: 4];
                            endcase
                        end
                    end
                end
            endcase
        end
    end

endmodule
