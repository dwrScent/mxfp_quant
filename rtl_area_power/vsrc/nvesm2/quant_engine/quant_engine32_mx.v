// =============================================================
// NVESM2 Quantization Engine
// - 32 lanes of FP32 input.
// - per_group_scale = per_group_max / E2M1_max.
// - One E4M3 group_scale/shared_scale per 16 lanes.
// - Four 8-lane subgroups, each choosing ES in {1, 1.25, 1.5, 1.75}.
// - Each lane has one pipelined RQU. The four metadata candidates are issued
//   in consecutive cycles (II=1) and accumulated by subgroup as they return.
// - Output FP4(E2M1) values and one 2-bit ES index per 8-lane subgroup.
// - Input is accepted only when in_valid and in_ready are both high.
// =============================================================
module quant_engine32_mx (
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

    integer ii;
    genvar gi;
    genvar gs;

    // -------------------- S0: 16-lane group scales --------------------
    wire [7:0] group_scale_e4m3_g0;
    wire [7:0] group_scale_e4m3_g1;
    wire       group_scale_valid_g0;
    wire       group_scale_valid_g1;

    nvesm2_group_scale U_GROUP_SCALE_G0 (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_valid        (in_valid && in_ready),
        .fp32_bus        (in_fp32_bus[16*32-1:0]),
        .out_valid       (group_scale_valid_g0),
        .group_scale_e4m3(group_scale_e4m3_g0)
    );

    nvesm2_group_scale U_GROUP_SCALE_G1 (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_valid        (in_valid && in_ready),
        .fp32_bus        (in_fp32_bus[32*32-1:16*32]),
        .out_valid       (group_scale_valid_g1),
        .group_scale_e4m3(group_scale_e4m3_g1)
    );

    // -------------------- Block state held while the four ES candidates run --------------------
    reg              active;
    reg              scale_ready;
    reg [2:0]        issue_count;
    reg [32*32-1:0] in_fp32_hold;
    reg [7:0]        group_scale_hold [0:1];
    reg [2*8-1:0]   group_scale_bus_hold;

    reg [3:0]        fp4_cand0 [0:31];
    reg [3:0]        fp4_cand1 [0:31];
    reg [3:0]        fp4_cand2 [0:31];
    reg [3:0]        fp4_cand3 [0:31];
    reg [35:0]       subgroup_cost0 [0:3];
    reg [35:0]       subgroup_cost1 [0:3];
    reg [35:0]       subgroup_cost2 [0:3];
    reg [35:0]       subgroup_cost3 [0:3];

    // -------------------- 32 pipelined RQUs, one ES candidate per lane per cycle --------------------
    wire        accept_input = in_valid && in_ready;
    wire        group_scale_valid = group_scale_valid_g0 && group_scale_valid_g1;
    wire        rqu_issue_valid = active && scale_ready && (issue_count < 3'd4);
    wire [1:0]  rqu_issue_es    = issue_count[1:0];
    wire [3:0]  lane_fp4        [0:31];
    wire [31:0] lane_cost       [0:31];
    wire        lane_out_valid  [0:31];
    wire [1:0]  lane_es_out     [0:31];

    wire        rqu_out_valid = lane_out_valid[0];
    wire [1:0]  rqu_es_out    = lane_es_out[0];

    assign in_ready = !active;

    generate
      for (gi=0; gi<32; gi=gi+1) begin: GEN_LANE_RQU
        nvesm2_quant_lane U_LANE (
            .clk         (clk),
            .rst_n       (rst_n),
            .in_valid    (rqu_issue_valid),
            .x_fp32      (in_fp32_hold[gi*32 + 31 : gi*32]),
            .group_scale ((gi < 16) ? group_scale_hold[0] : group_scale_hold[1]),
            .es_idx      (rqu_issue_es),
            .out_valid   (lane_out_valid[gi]),
            .es_idx_out  (lane_es_out[gi]),
            .fp4         (lane_fp4[gi]),
            .cost        (lane_cost[gi])
        );
      end
    endgenerate

    // -------------------- Candidate capture at RQU output --------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ii=0; ii<32; ii=ii+1) begin
                fp4_cand0[ii] <= 4'd0;
                fp4_cand1[ii] <= 4'd0;
                fp4_cand2[ii] <= 4'd0;
                fp4_cand3[ii] <= 4'd0;
            end
        end else if (rqu_out_valid && (rqu_es_out == 2'd0)) begin
            for (ii=0; ii<32; ii=ii+1)
                fp4_cand0[ii] <= lane_fp4[ii];
        end else if (rqu_out_valid && (rqu_es_out == 2'd1)) begin
            for (ii=0; ii<32; ii=ii+1)
                fp4_cand1[ii] <= lane_fp4[ii];
        end else if (rqu_out_valid && (rqu_es_out == 2'd2)) begin
            for (ii=0; ii<32; ii=ii+1)
                fp4_cand2[ii] <= lane_fp4[ii];
        end else if (rqu_out_valid) begin
            for (ii=0; ii<32; ii=ii+1)
                fp4_cand3[ii] <= lane_fp4[ii];
        end
    end

    // -------------------- Pipelined subgroup accumulation trees --------------------
    wire        accum_valid [0:3];
    wire [1:0]  accum_es    [0:3];
    wire [35:0] accum_cost  [0:3];

    generate
      for (gs=0; gs<4; gs=gs+1) begin: GEN_SUBGROUP_ACCUM
        nvesm2_subgroup_accum U_ACCUM (
            .clk           (clk),
            .rst_n         (rst_n),
            .in_valid      (rqu_out_valid),
            .es_idx        (rqu_es_out),
            .cost0         (lane_cost[gs*8 + 0]),
            .cost1         (lane_cost[gs*8 + 1]),
            .cost2         (lane_cost[gs*8 + 2]),
            .cost3         (lane_cost[gs*8 + 3]),
            .cost4         (lane_cost[gs*8 + 4]),
            .cost5         (lane_cost[gs*8 + 5]),
            .cost6         (lane_cost[gs*8 + 6]),
            .cost7         (lane_cost[gs*8 + 7]),
            .out_valid     (accum_valid[gs]),
            .es_idx_out    (accum_es[gs]),
            .subgroup_cost (accum_cost[gs])
        );
      end
    endgenerate

    wire        accum_out_valid = accum_valid[0];
    wire [1:0]  accum_es_out    = accum_es[0];

    // -------------------- Cost capture after accumulation tree --------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (ii=0; ii<4; ii=ii+1) begin
                subgroup_cost0[ii] <= 36'd0;
                subgroup_cost1[ii] <= 36'd0;
                subgroup_cost2[ii] <= 36'd0;
                subgroup_cost3[ii] <= 36'd0;
            end
        end else if (accum_out_valid && (accum_es_out == 2'd0)) begin
            for (ii=0; ii<4; ii=ii+1)
                subgroup_cost0[ii] <= accum_cost[ii];
        end else if (accum_out_valid && (accum_es_out == 2'd1)) begin
            for (ii=0; ii<4; ii=ii+1)
                subgroup_cost1[ii] <= accum_cost[ii];
        end else if (accum_out_valid && (accum_es_out == 2'd2)) begin
            for (ii=0; ii<4; ii=ii+1)
                subgroup_cost2[ii] <= accum_cost[ii];
        end else if (accum_out_valid) begin
            for (ii=0; ii<4; ii=ii+1)
                subgroup_cost3[ii] <= accum_cost[ii];
        end
    end

    // -------------------- Pipelined argmin tree --------------------
    reg        cmp_v1, cmp_v2;
    reg [35:0] cmp01_cost [0:3];
    reg [35:0] cmp23_cost [0:3];
    reg [1:0]  cmp01_idx  [0:3];
    reg [1:0]  cmp23_idx  [0:3];
    reg [1:0]  es_idx_final [0:3];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmp_v1 <= 1'b0;
            for (ii=0; ii<4; ii=ii+1) begin
                cmp01_cost[ii] <= 36'd0;
                cmp23_cost[ii] <= 36'd0;
                cmp01_idx[ii] <= 2'd0;
                cmp23_idx[ii] <= 2'd0;
            end
        end else begin
            cmp_v1 <= accum_out_valid && (accum_es_out == 2'd3);
            for (ii=0; ii<4; ii=ii+1) begin
                if (subgroup_cost1[ii] < subgroup_cost0[ii]) begin
                    cmp01_cost[ii] <= subgroup_cost1[ii];
                    cmp01_idx[ii] <= 2'd1;
                end else begin
                    cmp01_cost[ii] <= subgroup_cost0[ii];
                    cmp01_idx[ii] <= 2'd0;
                end

                if (accum_cost[ii] < subgroup_cost2[ii]) begin
                    cmp23_cost[ii] <= accum_cost[ii];
                    cmp23_idx[ii] <= 2'd3;
                end else begin
                    cmp23_cost[ii] <= subgroup_cost2[ii];
                    cmp23_idx[ii] <= 2'd2;
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmp_v2 <= 1'b0;
            for (ii=0; ii<4; ii=ii+1)
                es_idx_final[ii] <= 2'd0;
        end else begin
            cmp_v2 <= cmp_v1;
            for (ii=0; ii<4; ii=ii+1) begin
                if (cmp23_cost[ii] < cmp01_cost[ii])
                    es_idx_final[ii] <= cmp23_idx[ii];
                else
                    es_idx_final[ii] <= cmp01_idx[ii];
            end
        end
    end

    // -------------------- Control: issue four ES candidates and publish after compare tree --------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            scale_ready <= 1'b0;
            issue_count <= 3'd0;
            out_valid <= 1'b0;
            in_fp32_hold <= {32*32{1'b0}};
            group_scale_hold[0] <= 8'h00;
            group_scale_hold[1] <= 8'h00;
            group_scale_bus_hold <= {2*8{1'b0}};
            group_scale_bus <= {2*8{1'b0}};
            fp4_bus <= {32*4{1'b0}};
            es_idx_bus <= {4*2{1'b0}};
        end else begin
            out_valid <= 1'b0;

            if (!active) begin
                if (accept_input) begin
                    active <= 1'b1;
                    scale_ready <= 1'b0;
                    issue_count <= 3'd0;
                    in_fp32_hold <= in_fp32_bus;
                end
            end else begin
                if (!scale_ready && group_scale_valid) begin
                    scale_ready <= 1'b1;
                    group_scale_hold[0] <= group_scale_e4m3_g0;
                    group_scale_hold[1] <= group_scale_e4m3_g1;
                    group_scale_bus_hold <= {group_scale_e4m3_g1, group_scale_e4m3_g0};
                end

                if (scale_ready && (issue_count < 3'd4))
                    issue_count <= issue_count + 3'd1;

                if (cmp_v2) begin
                    active <= 1'b0;
                    scale_ready <= 1'b0;
                    issue_count <= 3'd0;
                    out_valid <= 1'b1;
                    group_scale_bus <= group_scale_bus_hold;

                    for (ii=0; ii<4; ii=ii+1)
                        es_idx_bus[ii*2 + 1 -: 2] <= es_idx_final[ii];

                    for (ii=0; ii<32; ii=ii+1) begin
                        if (es_idx_final[ii/8] == 2'd0)
                            fp4_bus[ii*4 + 3 -: 4] <= fp4_cand0[ii];
                        else if (es_idx_final[ii/8] == 2'd1)
                            fp4_bus[ii*4 + 3 -: 4] <= fp4_cand1[ii];
                        else if (es_idx_final[ii/8] == 2'd2)
                            fp4_bus[ii*4 + 3 -: 4] <= fp4_cand2[ii];
                        else
                            fp4_bus[ii*4 + 3 -: 4] <= fp4_cand3[ii];
                    end
                end
            end
        end
    end

endmodule
