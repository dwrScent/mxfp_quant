// int4_mac_acc32bit_tb.v
`timescale 1ns / 1ps

module int4_mac_acc32bit_tb;
    reg clk;
    reg rst;
    reg [3:0] a;
    reg [3:0] b;
    wire [31:0] c;

    // 实例化 DUT (Device Under Test)
    int4_mac_acc32bit uut (
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .c(c)
    );

    // 时钟生成
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns 时钟周期
    end

    // 仿真过程
    initial begin
        $dumpfile("waveform.vcd"); // 生成波形文件
        $dumpvars(0, int4_mac_acc32bit_tb);

        // 初始化信号
        rst = 1;
        a = 4'b0;
        b = 4'b0;
        #10;

        rst = 0;
        a = 4'b0011; // 3
        b = 4'b0010; // 2
        #10;

        a = 4'b1110; // -2 (补码)
        b = 4'b0011; // 3
        #10;

        a = 4'b0100; // 4
        b = 4'b1101; // -3 (补码)
        #10;

        $finish;
    end
endmodule