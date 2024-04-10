module fp16_max(
    input [15:0] a,
    input [15:0] b,
    output reg [15:0] max
    );

    // Splitting the inputs into their sign, exponent, and significand components
    wire a_sign = a[15];
    wire b_sign = b[15];
    wire [4:0] a_exp = a[14:10];
    wire [4:0] b_exp = b[14:10];
    wire [9:0] a_mant = a[9:0];
    wire [9:0] b_mant = b[9:0];

    // Comparing the numbers
    always @(a or b) begin
        // If signs are different, the positive number is larger
        if (a_sign != b_sign) begin
            max = a_sign ? b : a;
        end else if (a_exp > b_exp) begin
            // If exponents are different, the one with the larger exponent is larger
            max = a_sign ? b : a; // If negative, reverse logic
        end else if (a_exp < b_exp) begin
            max = a_sign ? a : b; // If negative, reverse logic
        end else begin
            // If exponents are equal, compare significands
            if (a_mant > b_mant) begin
                max = a_sign ? b : a; // If negative, reverse logic
            end else begin
                max = a_sign ? a : b; // If negative, reverse logic
			end
        end
    end
endmodule

module fp16_comparator(
    input clk,
    input rst,
    input mode, // 0 for up mode and 1 for left mode
    input [15:0] left,
    input [15:0] up,
    output [15:0] down,
    output [15:0] right
    );
    wire [15:0] in_data = (mode==0) ? left : up;
    reg [15:0] result;
    assign down = result;
    assign right = result;
	wire [15:0] max_data;

	fp16_max fp16_max_module(
		.a(in_data),
		.b(result),
		.max(max_data)
	);
    always @(posedge clk) begin
        if(rst) begin
            result <= 16'b0111110000000000;
        end else begin
			result <= max_data;
        end
    end
endmodule 
