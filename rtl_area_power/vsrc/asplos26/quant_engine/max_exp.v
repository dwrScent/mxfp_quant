// 仅比较 exponent(8b, biased) 的最大值；忽略符号/尾数
// Verilog-2001 Compliant Version using a Comparator Tree
module max_exp (
    input  [32*8-1:0] exp_bus,
    // [FIX] Output is now driven by assign statements, so it must be a wire.
    output [7:0]      max_exp_biased
);

    // Stage 1: 32 inputs -> 16 outputs (max of pairs)
    wire [7:0] max_s1 [0:15];
    assign max_s1[ 0] = (exp_bus[  7:  0] > exp_bus[ 15:  8]) ? exp_bus[  7:  0] : exp_bus[ 15:  8];
    assign max_s1[ 1] = (exp_bus[ 23: 16] > exp_bus[ 31: 24]) ? exp_bus[ 23: 16] : exp_bus[ 31: 24];
    assign max_s1[ 2] = (exp_bus[ 39: 32] > exp_bus[ 47: 40]) ? exp_bus[ 39: 32] : exp_bus[ 47: 40];
    assign max_s1[ 3] = (exp_bus[ 55: 48] > exp_bus[ 63: 56]) ? exp_bus[ 55: 48] : exp_bus[ 63: 56];
    assign max_s1[ 4] = (exp_bus[ 71: 64] > exp_bus[ 79: 72]) ? exp_bus[ 71: 64] : exp_bus[ 79: 72];
    assign max_s1[ 5] = (exp_bus[ 87: 80] > exp_bus[ 95: 88]) ? exp_bus[ 87: 80] : exp_bus[ 95: 88];
    assign max_s1[ 6] = (exp_bus[103: 96] > exp_bus[111:104]) ? exp_bus[103: 96] : exp_bus[111:104];
    assign max_s1[ 7] = (exp_bus[119:112] > exp_bus[127:120]) ? exp_bus[119:112] : exp_bus[127:120];
    assign max_s1[ 8] = (exp_bus[135:128] > exp_bus[143:136]) ? exp_bus[135:128] : exp_bus[143:136];
    assign max_s1[ 9] = (exp_bus[151:144] > exp_bus[159:152]) ? exp_bus[151:144] : exp_bus[159:152];
    assign max_s1[10] = (exp_bus[167:160] > exp_bus[175:168]) ? exp_bus[167:160] : exp_bus[175:168];
    assign max_s1[11] = (exp_bus[183:176] > exp_bus[191:184]) ? exp_bus[183:176] : exp_bus[191:184];
    assign max_s1[12] = (exp_bus[199:192] > exp_bus[207:200]) ? exp_bus[199:192] : exp_bus[207:200];
    assign max_s1[13] = (exp_bus[215:208] > exp_bus[223:216]) ? exp_bus[215:208] : exp_bus[223:216];
    assign max_s1[14] = (exp_bus[231:224] > exp_bus[239:232]) ? exp_bus[231:224] : exp_bus[239:232];
    assign max_s1[15] = (exp_bus[247:240] > exp_bus[255:248]) ? exp_bus[247:240] : exp_bus[255:248];

    // Stage 2: 16 inputs -> 8 outputs
    wire [7:0] max_s2 [0:7];
    assign max_s2[0] = (max_s1[ 0] > max_s1[ 1]) ? max_s1[ 0] : max_s1[ 1];
    assign max_s2[1] = (max_s1[ 2] > max_s1[ 3]) ? max_s1[ 2] : max_s1[ 3];
    assign max_s2[2] = (max_s1[ 4] > max_s1[ 5]) ? max_s1[ 4] : max_s1[ 5];
    assign max_s2[3] = (max_s1[ 6] > max_s1[ 7]) ? max_s1[ 6] : max_s1[ 7];
    assign max_s2[4] = (max_s1[ 8] > max_s1[ 9]) ? max_s1[ 8] : max_s1[ 9];
    assign max_s2[5] = (max_s1[10] > max_s1[11]) ? max_s1[10] : max_s1[11];
    assign max_s2[6] = (max_s1[12] > max_s1[13]) ? max_s1[12] : max_s1[13];
    assign max_s2[7] = (max_s1[14] > max_s1[15]) ? max_s1[14] : max_s1[15];

    // Stage 3: 8 inputs -> 4 outputs
    wire [7:0] max_s3 [0:3];
    assign max_s3[0] = (max_s2[0] > max_s2[1]) ? max_s2[0] : max_s2[1];
    assign max_s3[1] = (max_s2[2] > max_s2[3]) ? max_s2[2] : max_s2[3];
    assign max_s3[2] = (max_s2[4] > max_s2[5]) ? max_s2[4] : max_s2[5];
    assign max_s3[3] = (max_s2[6] > max_s2[7]) ? max_s2[6] : max_s2[7];

    // Stage 4: 4 inputs -> 2 outputs
    wire [7:0] max_s4 [0:1];
    assign max_s4[0] = (max_s3[0] > max_s3[1]) ? max_s3[0] : max_s3[1];
    assign max_s4[1] = (max_s3[2] > max_s3[3]) ? max_s3[2] : max_s3[3];

    // Stage 5: 2 inputs -> 1 output
    wire [7:0] max_s5;
    assign max_s5 = (max_s4[0] > max_s4[1]) ? max_s4[0] : max_s4[1];

    // Final assignment
    // Note: The logic does not need to handle the case where all inputs are 0,
    // because the biased exponent of a valid FP32 number is never 0 unless it's a subnormal or zero,
    // and even then, comparing against other valid exponents works. If all inputs were 0,
    // the output would correctly be 0.
    assign max_exp_biased = max_s5;

endmodule
