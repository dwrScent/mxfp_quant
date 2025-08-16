// =============================================================
// Encode Unit (works with your top1_detection_unit ports)
// Policy:
//   if (idx_base == idx_refine) {
//     fp4_out      = FP4_refine[idx_refine];
//     metadata_out = metadata[idx_refine];
//   } else {
//     fp4_out      = FP4_base[idx_base];
//     metadata_out = 2'b00;
//   }
// =============================================================
module encode_unit (
    // base group
    input [3:0] FP4_base_0, input [3:0] FP4_base_1,
    input [3:0] FP4_base_2, input [3:0] FP4_base_3,
    input [3:0] FP4_base_4, input [3:0] FP4_base_5,
    input [3:0] FP4_base_6, input [3:0] FP4_base_7,

    // refine group
    input [3:0] FP4_refine_0, input [3:0] FP4_refine_1,
    input [3:0] FP4_refine_2, input [3:0] FP4_refine_3,
    input [3:0] FP4_refine_4, input [3:0] FP4_refine_5,
    input [3:0] FP4_refine_6, input [3:0] FP4_refine_7,

    // metadata (2-bit per lane)
    input [1:0] metadata_0, input [1:0] metadata_1,
    input [1:0] metadata_2, input [1:0] metadata_3,
    input [1:0] metadata_4, input [1:0] metadata_5,
    input [1:0] metadata_6, input [1:0] metadata_7,

    // outputs
    output [3:0] FP4_out,
    output [1:0] metadata_out
);
    // -------- two Top-1 Detection Units --------
    wire [2:0] idx_base, idx_refine;

    top1_detection_unit U_TOP1_BASE (
        .fp4_in0(FP4_base_0), .fp4_in1(FP4_base_1),
        .fp4_in2(FP4_base_2), .fp4_in3(FP4_base_3),
        .fp4_in4(FP4_base_4), .fp4_in5(FP4_base_5),
        .fp4_in6(FP4_base_6), .fp4_in7(FP4_base_7),
        .idx_max(idx_base)
    );

    top1_detection_unit U_TOP1_REFINE (
        .fp4_in0(FP4_refine_0), .fp4_in1(FP4_refine_1),
        .fp4_in2(FP4_refine_2), .fp4_in3(FP4_refine_3),
        .fp4_in4(FP4_refine_4), .fp4_in5(FP4_refine_5),
        .fp4_in6(FP4_refine_6), .fp4_in7(FP4_refine_7),
        .idx_max(idx_refine)
    );

    // -------- helpers: select lane by idx --------
    function [3:0] sel_fp4_base;
        input [2:0] idx;
        begin
            case (idx)
                3'd0: sel_fp4_base = FP4_base_0;
                3'd1: sel_fp4_base = FP4_base_1;
                3'd2: sel_fp4_base = FP4_base_2;
                3'd3: sel_fp4_base = FP4_base_3;
                3'd4: sel_fp4_base = FP4_base_4;
                3'd5: sel_fp4_base = FP4_base_5;
                3'd6: sel_fp4_base = FP4_base_6;
                3'd7: sel_fp4_base = FP4_base_7;
                default: sel_fp4_base = 4'b0000;
            endcase
        end
    endfunction

    function [3:0] sel_fp4_refine;
        input [2:0] idx;
        begin
            case (idx)
                3'd0: sel_fp4_refine = FP4_refine_0;
                3'd1: sel_fp4_refine = FP4_refine_1;
                3'd2: sel_fp4_refine = FP4_refine_2;
                3'd3: sel_fp4_refine = FP4_refine_3;
                3'd4: sel_fp4_refine = FP4_refine_4;
                3'd5: sel_fp4_refine = FP4_refine_5;
                3'd6: sel_fp4_refine = FP4_refine_6;
                3'd7: sel_fp4_refine = FP4_refine_7;
                default: sel_fp4_refine = 4'b0000;
            endcase
        end
    endfunction

    function [1:0] sel_metadata;
        input [2:0] idx;
        begin
            case (idx)
                3'd0: sel_metadata = metadata_0;
                3'd1: sel_metadata = metadata_1;
                3'd2: sel_metadata = metadata_2;
                3'd3: sel_metadata = metadata_3;
                3'd4: sel_metadata = metadata_4;
                3'd5: sel_metadata = metadata_5;
                3'd6: sel_metadata = metadata_6;
                3'd7: sel_metadata = metadata_7;
                default: sel_metadata = 2'b00;
            endcase
        end
    endfunction

    // -------- decision logic --------
    wire same_idx = (idx_base == idx_refine);

    wire [3:0] base_pick   = sel_fp4_base  (idx_base);
    wire [3:0] refine_pick = sel_fp4_refine(idx_refine);
    wire [1:0] meta_pick   = sel_metadata  (idx_refine);

    assign FP4_out      = same_idx ? refine_pick : base_pick;
    assign metadata_out = same_idx ? meta_pick   : 2'b00;

endmodule
