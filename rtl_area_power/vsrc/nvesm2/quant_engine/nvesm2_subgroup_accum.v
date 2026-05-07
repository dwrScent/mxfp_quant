// =============================================================
// Sum eight per-lane costs into one subgroup cost.
// Per-lane cost max is 12544, so the eight-lane sum max is 100352 (17 bits).
// The quant engine instantiates four of these modules, one per 8-lane
// subgroup.  The tree is pipelined to avoid an eight-input add chain.
// =============================================================
module nvesm2_subgroup_accum (
    input             clk,
    input             rst_n,
    input             in_valid,
    input      [1:0]  es_idx,
    input      [13:0] cost0,
    input      [13:0] cost1,
    input      [13:0] cost2,
    input      [13:0] cost3,
    input      [13:0] cost4,
    input      [13:0] cost5,
    input      [13:0] cost6,
    input      [13:0] cost7,
    output reg        out_valid,
    output reg [1:0]  es_idx_out,
    output reg [16:0] subgroup_cost
);
    reg        v1, v2; // valid bits for the two internal adder-tree stages
    reg [1:0]  es1, es2; // ES selector delayed alongside the partial sums
    reg [14:0] s1_0, s1_1, s1_2, s1_3; // stage 1: pairwise cost sums
    reg [15:0] s2_0, s2_1;             // stage 2: four-lane partial sums

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            es1 <= 2'd0;
            s1_0 <= 15'd0;
            s1_1 <= 15'd0;
            s1_2 <= 15'd0;
            s1_3 <= 15'd0;
        end else begin
            v1 <= in_valid;
            es1 <= es_idx;
            s1_0 <= {1'b0, cost0} + {1'b0, cost1};
            s1_1 <= {1'b0, cost2} + {1'b0, cost3};
            s1_2 <= {1'b0, cost4} + {1'b0, cost5};
            s1_3 <= {1'b0, cost6} + {1'b0, cost7};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v2 <= 1'b0;
            es2 <= 2'd0;
            s2_0 <= 16'd0;
            s2_1 <= 16'd0;
        end else begin
            v2 <= v1;
            es2 <= es1;
            s2_0 <= {1'b0, s1_0} + {1'b0, s1_1};
            s2_1 <= {1'b0, s1_2} + {1'b0, s1_3};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            es_idx_out <= 2'd0;
            subgroup_cost <= 17'd0;
        end else begin
            out_valid <= v2;
            es_idx_out <= es2;
            subgroup_cost <= {1'b0, s2_0} + {1'b0, s2_1};
        end
    end
endmodule
