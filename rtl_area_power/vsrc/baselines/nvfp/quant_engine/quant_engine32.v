// =============================================================
// NVFP Quantization Engine
//
// Flow for one 32-lane block:
//   1. Build one E4M3 group scale for each 16-lane group.
//   2. Apply the matching group scale in each lane.
//   3. Quantize each lane directly to FP4 E2M1.
//   4. Publish the two group scales and 32 FP4 values.
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
    output reg [32*4-1:0] fp4_bus
);

    localparam ST_IDLE       = 2'd0;
    localparam ST_WAIT_SCALE = 2'd1;
    localparam ST_ISSUE      = 2'd2;
    localparam ST_WAIT_LANE  = 2'd3;

    integer ii;
    genvar gi;

    reg [1:0]        state;
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

    nvfp_group_scale U_GROUP_SCALE_G0 (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_valid        (accept_input),
        .fp32_bus        (in_fp32_bus[16*32-1:0]),
        .out_valid       (group_scale_valid_g0),
        .group_scale_e4m3(group_scale_e4m3_g0)
    );

    nvfp_group_scale U_GROUP_SCALE_G1 (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_valid        (accept_input),
        .fp32_bus        (in_fp32_bus[32*32-1:16*32]),
        .out_valid       (group_scale_valid_g1),
        .group_scale_e4m3(group_scale_e4m3_g1)
    );

    // -------------------- S1-S2: per-lane shared-scale quantization --------------------
    wire              lane_issue_valid = (state == ST_ISSUE);
    wire [3:0]        lane_fp4 [0:31];
    wire [31:0]       lane_valid_bus;
    wire [32*4-1:0]   lane_fp4_bus;

    generate
      for (gi=0; gi<32; gi=gi+1) begin: GEN_LANE
        nvfp_quant_lane U_LANE (
            .clk        (clk),
            .rst_n      (rst_n),
            .in_valid   (lane_issue_valid),
            .x_fp32     (fp32_hold[gi*32 + 31 : gi*32]),
            .group_scale((gi < 16) ? group_scale_g0 : group_scale_g1),
            .out_valid  (lane_valid_bus[gi]),
            .fp4        (lane_fp4[gi])
        );

        assign lane_fp4_bus[gi*4 + 3 -: 4] = lane_fp4[gi];
      end
    endgenerate

    wire lane_out_valid = &lane_valid_bus;

    // -------------------- Control and output publication --------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            fp32_hold <= {32*32{1'b0}};
            group_scale_g0 <= 8'h00;
            group_scale_g1 <= 8'h00;
            group_scale_hold <= {2*8{1'b0}};
            out_valid <= 1'b0;
            group_scale_bus <= {2*8{1'b0}};
            fp4_bus <= {32*4{1'b0}};
        end else begin
            out_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (accept_input) begin
                        state <= ST_WAIT_SCALE;
                        fp32_hold <= in_fp32_bus;
                    end
                end

                ST_WAIT_SCALE: begin
                    if (group_scale_done) begin
                        state <= ST_ISSUE;
                        group_scale_g0 <= group_scale_e4m3_g0;
                        group_scale_g1 <= group_scale_e4m3_g1;
                        group_scale_hold <= {group_scale_e4m3_g1, group_scale_e4m3_g0};
                    end
                end

                ST_ISSUE: begin
                    state <= ST_WAIT_LANE;
                end

                default: begin
                    if (lane_out_valid) begin
                        state <= ST_IDLE;
                        out_valid <= 1'b1;
                        group_scale_bus <= group_scale_hold;
                        for (ii=0; ii<32; ii=ii+1)
                            fp4_bus[ii*4 + 3 -: 4] <= lane_fp4_bus[ii*4 + 3 -: 4];
                    end
                end
            endcase
        end
    end

endmodule
