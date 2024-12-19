// mxint4_pot_scale_mac_i4o32.v
`timescale 1ns / 1ps

module mxint4_pot_scale_mac_i4o32;
    reg clk;
    reg rst;
    reg [3:0] a;
    reg [3:0] b;
    reg [7:0] scale_a;
    reg [7:0] scale_b;
    wire [31:0] c;
    wire error;

    // 实例化 DUT (Device Under Test)
    mxint4_mac uut (
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .scale_a(scale_a),
        .scale_b(scale_b),
        .c_out(c)
    );

    // 时钟生成
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns 时钟周期
    end

    // 仿真过程
    initial begin
        $dumpfile("waveform.vcd"); // 生成波形文件
        $dumpvars(0, mxint4_pot_scale_mac_i4o32);

        // 初始化信号
        rst = 1;
        a = 4'b0;
        b = 4'b0;
        scale_a = 8'b0;
        scale_b = 8'b0;
        #10;

        rst = 0;
        a = 4'b0011; // 3
        b = 4'b0010; // 2
        scale_a = 8'b0010; // 2
        scale_b = 8'b0010; // 2
        #10;

        a = 4'b1110; // -2 (补码)
        b = 4'b0011; // 3
        scale_a = 8'b0010; // 2
        scale_b = 8'b0010; // 2
        #10;

        a = 4'b0100; // 4
        b = 4'b1101; // -3 (补码)
        scale_a = 8'b0010; // 2
        scale_b = 8'b0010; // 2
        #10;

        $finish;
    end
endmodule