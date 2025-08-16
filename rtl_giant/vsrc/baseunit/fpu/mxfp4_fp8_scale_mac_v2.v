module mxfp4_mac (
    input  wire        clk,        // 时钟
    input  wire        rst,        // 复位
    input  wire [3:0]  a,          // FP4 输入 a
    input  wire [3:0]  b,          // FP4 输入 b
    input  wire [7:0]  scale_a,    // Scaling factor A (FP8)
    input  wire [7:0]  scale_b,    // Scaling factor B (FP8)
    output reg  [15:0] c_out,      // 最终结果 (FP16)
    output wire        error       // 错误标志
);

    reg [15:0] c_acc;              // 累加结果 (FP16)
    wire [15:0] c_tmp;             // 中间累加结果
    wire [15:0] combined_scale;    // 合成 Scaling Factor
    wire [15:0] final_result;      // 最终结果
    wire        mul_error;         // 错误标志

    // 累加逻辑
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_acc <= 16'b0; // 复位
        else
            c_acc <= c_tmp; // 累加结果
    end

    // 实例化 FP4 MAC 核心逻辑
    fp4_mac core_mac (
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .c(c_tmp),
        .error(error)
    );

    // 合并两个 Scaling Factor
    fp8_mul scale_combiner (
        .a(scale_a),
        .b(scale_b),
        .c(combined_scale),
        .error()
    );

    // 将累加结果与合成 Scaling Factor 相乘
    fp16_mul final_multiplier (
        .a(c_acc),
        .b(combined_scale),
        .c(final_result),
        .error(mul_error)
    );

    // 输出最终结果
    always @(posedge clk or posedge rst) begin
        if (rst)
            c_out <= 16'b0; // 复位
        else
            c_out <= final_result; // 最终结果
    end

    assign error = mul_error; // 传播错误标志

endmodule

module fp4_mac (
    input  wire        clk,       // 时钟
    input  wire        rst,       // 复位
    input  wire [3:0]  a,         // FP4 输入 a
    input  wire [3:0]  b,         // FP4 输入 b
    output reg  [15:0] c,         // FP16 累加结果
    output wire        error      // 溢出/下溢错误
);

    wire [15:0] mul_result;       // FP4 -> FP16 乘法结果
    wire [15:0] c_tmp;            // 中间累加结果
    wire        mul_error;        // 乘法溢出标志

    // 主逻辑：累加结果寄存器
    always @(posedge clk or posedge rst) begin
        if (rst)
            c <= 16'b0; // 复位
        else
            c <= c_tmp;
    end
        // 实例化 FP4 乘法器
    fp4_mul u1 (
        .a(a),
        .b(b),
        .c(mul_result),
        .error(mul_error)
    );

    // 实例化 FP16 加法器
    fp16_adder u2 (
        .a(c),
        .b(mul_result),
        .c(c_tmp)
    );

    assign error = mul_error; // 传播乘法错误标志
endmodule

module fp4_mul (
    input  wire [3:0]  a,         // FP4 输入 a
    input  wire [3:0]  b,         // FP4 输入 b
    output wire [15:0] c,         // FP16 乘法结果
    output wire        error      // 溢出/下溢标志
);

    wire a_sign, b_sign, c_sign;
    wire [1:0] a_exp, b_exp;
    wire [1:0] a_mant, b_mant;      // FP4 尾数，包含隐含位
    wire [3:0] mant_product;        // 尾数乘积
    wire [5:0] exp_sum;
    wire underflow, overflow;

    // 提取 FP4 的符号位、指数位、尾数位
    assign a_sign = a[3];
    assign b_sign = b[3];
    assign a_exp  = a[2:1];
    assign b_exp  = b[2:1];
    assign a_mant = {1'b1, a[0]}; // 扩展尾数，隐含位为 1
    assign b_mant = {1'b1, b[0]}; // 扩展尾数，隐含位为 1

    // 计算结果的符号位和尾数位
    assign c_sign = a_sign ^ b_sign;
    assign mant_product = a_mant * b_mant; // 尾数乘积为 4 位

    // 指数相加，考虑偏置（Bias = 15 for FP16, Bias = 1 for FP4）
    assign exp_sum = a_exp + b_exp - 6'b000001 + 6'b011111; // Adjust bias

    // 检测溢出和下溢
    assign overflow = (exp_sum > 6'b111110); // 超出 FP16 指数范围
    assign underflow = (exp_sum < 6'b000001); // 小于 FP16 最小指数

    // 规范化输出，扩展尾数为 10 位
    assign c = overflow ? {c_sign, 5'b11111, 10'b0} :            // 溢出
               underflow ? {c_sign, 5'b00000, 10'b0} :           // 下溢
               {c_sign, exp_sum[4:0], {mant_product, 6'b0}};     // 正常输出

    assign error = overflow | underflow;

endmodule

module fp16_adder (
    input  [15:0] a,
    input  [15:0] b,
    output [15:0] c
);

    wire [10:0] adder_input_1,adder_input_2,aligned_small,adder_output;
    wire if_sub,a_sign, b_sign, c_sign,c1;
    wire [15:0] normalized_out;

    wire [15:0] result;
    reg [14:0] bigger, smaller;
    reg a_larger_b;

    assign a_sign        = a[15];
    assign b_sign        = b[15];
    assign if_sub        = (a_sign == b_sign) ? 1'b0 : 1'b1;
    assign c_sign        = a_larger_b ? a_sign : b_sign;
    assign adder_input_1 = {1'b1,bigger[9:0]};
    assign adder_input_2 = (if_sub ? ~aligned_small + 1'b1 : aligned_small);
    assign c             = result;

    //compare two number regardless sign
    always @(*) begin
        if (a[14:0] > b[14:0]) begin
            bigger = a[14:0];
            smaller = b[14:0];
            a_larger_b = 1'b1;
        end else begin 
            bigger = b[14:0];
            smaller = a[14:0];
            a_larger_b = 1'b0;
        end 
    end

    // align small number
    alignment u1(bigger,smaller,aligned_small);

    assign {c1, adder_output} = adder_input_1 + adder_input_2;

    add_normalizer u4(c_sign,bigger[14:10],adder_output,result,c1,if_sub);

endmodule

module alignment (
	input  [14:0] bigger, 
	input  [14:0] smaller,
	output [10:0] aligned_small
	);

	wire c1;
	wire [4:0] bigger_exponent, smaller_exponent,shift_bits;

	assign bigger_exponent  = bigger  [14:10];
	assign smaller_exponent = smaller [14:10];
	assign aligned_small    = ({1'b1,smaller[9:0]} >> shift_bits);

    assign {c1, shift_bits} = bigger_exponent - smaller_exponent;
endmodule

module add_normalizer (
    input             sign,
    input      [ 4:0] exponent,
    input      [10:0] mantissa_add,
    output reg [15:0] result,
    input             if_carray,
    input             if_sub
    );

    reg [4:0] number_of_zero_lead;
    reg [10:0] norm_mantissa_add;
    reg [9:0] mantissa_tmp;

    wire [4:0] shift_left_exp;
    wire c1;

    always @ (*) begin
        if (mantissa_add[10:4] == 7'b0000_001) begin
            number_of_zero_lead = 5'd6;
            norm_mantissa_add   = (mantissa_add << 4'd6);
        end else if (mantissa_add[10:5] == 6'b0000_01) begin 
            number_of_zero_lead = 5'd5;
            norm_mantissa_add   = (mantissa_add << 4'd5);
        end else if (mantissa_add[10:6] == 5'b0000_1) begin
            number_of_zero_lead = 5'd4;
            norm_mantissa_add   = (mantissa_add << 4'd4);
        end else if (mantissa_add[10:7] == 4'b0001) begin
            number_of_zero_lead = 5'd3;
            norm_mantissa_add   = (mantissa_add << 4'd3);
        end else if (mantissa_add[10:8] == 3'b001) begin
            number_of_zero_lead = 5'd2;
            norm_mantissa_add   = (mantissa_add << 4'd2);
        end else if (mantissa_add[10:9] == 2'b01) begin
            number_of_zero_lead = 5'd1;
            norm_mantissa_add   = (mantissa_add << 4'd1);
        end else begin 
            number_of_zero_lead = 5'd0;
            norm_mantissa_add   = mantissa_add[10:0];
        end 
    end

    always @(*) begin
        result[15]        = sign;
        if (!if_sub) begin 
            result[14:10] = if_carray ? exponent + 1'b1 : exponent;
            result[9:0]   = if_carray ? mantissa_add[10:1] : mantissa_add[9:0];
        end else begin 
            result[14:10] = shift_left_exp;
            result[9:0]   = norm_mantissa_add[9:0];
        end 
    end

	assign {c1, shift_left_exp} = exponent - number_of_zero_lead;
endmodule

module fp8_mul (
    input  [7:0] a,
    input  [7:0] b,
    output [7:0] c,
    output        error // valid in fp16 mode 
    );

    wire [7:0] c_tmp;
    wire        c_sign,a_zero,b_zero;
    wire [4:0] sum_exponent, biased_sum_exponent; 
    wire [3:0] multiplier_input1,multiplier_input2; 

    wire [7:0] multiplier_output; 
    wire [6:0] normalized_out; 
    wire [5:0] mantissa_prod; 
    wire c1,c2,underflow,overflow;

    assign overflow = (c1 && c2 && ~biased_sum_exponent[4]) ? 1'b1 :1'b0; 
    assign underflow = (~c1 && ~c2 && biased_sum_exponent[4]) ? 1'b1:1'b0; 

    assign a_zero = ~(|a);
    assign b_zero = ~(|b);
    assign c_sign = a[7] ^ b[7];
    assign multiplier_input1 = {1'b0,1'b1,a[1:0]};
    assign multiplier_input2 = {1'b0,1'b1,b[1:0]};
    
    assign c = ((a_zero | b_zero) ? 8'b0 : c_tmp);
    //error detect
    assign c_tmp = (~error) ? {c_sign,normalized_out} : (underflow ? {c_sign,7'b0000000} : {c_sign,5'b1111_1,2'b00});
    
    assign error = overflow | underflow; 

    assign mantissa_prod = multiplier_output[5:0];
    assign multiplier_output = multiplier_input1 * multiplier_input2;
    
    assign {c1,sum_exponent} = a[6:2] + b[6:2];
    assign {c2,biased_sum_exponent} = sum_exponent + 5'b10001; //minux bias

    mul_normalizer_fp8 u4(biased_sum_exponent,mantissa_prod,normalized_out);
endmodule

module mul_normalizer_fp8 (
	input  [4:0] exponent,
	input  [5:0] mantissa_prod,
	output [6:0] result
);

	wire [4:0] result_exponent;
	wire [1:0] result_mantissa;

	assign result_exponent = (mantissa_prod[5]) ? (exponent + 1'b1): exponent;
	assign result_mantissa = (mantissa_prod[5]) ? mantissa_prod[4:3]:mantissa_prod[3:2];
	assign result 		   = {result_exponent,result_mantissa};

// No rounding and No overflow/underflow detection
endmodule

module fp16_mul (
    input  [15:0] a,
    input  [15:0] b,
    output [15:0] c,
    output        error // valid in fp16 mode 
    );

    wire [15:0] c_tmp;
    wire        c_sign,a_zero,b_zero;
    wire [4:0] sum_exponent, biased_sum_exponent;
    wire [15:0] multiplier_input1,multiplier_input2;

    wire [31:0] multiplier_output;
    wire [14:0] normalized_out;
    wire [21:0] mantissa_prod;
    wire c1,c2,underflow,overflow;

    assign overflow = (c1 && c2 && ~biased_sum_exponent[4]) ? 1'b1 :1'b0;
    assign underflow = (~c1 && ~c2 && biased_sum_exponent[4]) ? 1'b1:1'b0;

    assign a_zero = ~(|a);
    assign b_zero = ~(|b);
    assign c_sign = a[15] ^ b[15];
    assign multiplier_input1 = {5'b0,1'b1,a[9:0]};
    assign multiplier_input2 = {5'b0,1'b1,b[9:0]};
    
    assign c = ((a_zero | b_zero) ? 16'b0 : c_tmp);
    //error detect
    assign c_tmp = (~error) ? {c_sign,normalized_out} : (underflow ? {c_sign,15'b0000_0000_0000_000} : {c_sign,5'b1111_1,10'b0000_0000_00});
    
    assign error = overflow | underflow; 

    assign mantissa_prod = multiplier_output[21:0];
    assign multiplier_output = multiplier_input1 * multiplier_input2;
    
    assign {c1, sum_exponent} = a[14:10] + b[14:10];
    assign {c2, biased_sum_exponent} = sum_exponent + 5'b10001; //minux bias

    mul_normalizer_fp16 u4(biased_sum_exponent,mantissa_prod,normalized_out);

endmodule

module mul_normalizer_fp16 (
	input  [ 4:0] exponent,
	input  [21:0] mantissa_prod,
	output [14:0] result
);

	wire [4:0] result_exponent;
	wire [9:0] result_mantissa;

	assign result_exponent = (mantissa_prod[21]) ? (exponent + 1'b1): exponent;
	assign result_mantissa = (mantissa_prod[21]) ? mantissa_prod[20:11]:mantissa_prod[19:10];
	assign result 		   = {result_exponent,result_mantissa};

// No rounding and No overflow/underflow detection

endmodule
