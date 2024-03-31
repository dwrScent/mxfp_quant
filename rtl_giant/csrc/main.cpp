#include "verilated.h"
#include "verilated_vcd_c.h"
#include <inttypes.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <Vpe.h>

using namespace std;

VerilatedContext* contextp = NULL;
VerilatedVcdC* tfp = NULL;

static Vpe* top;

void step_and_dump_wave(){
    top->eval();
    contextp->timeInc(1);
    tfp->dump(contextp->time());
}

void sim_init(){
    contextp = new VerilatedContext;
    tfp = new VerilatedVcdC;
    top = new Vpe;
    contextp->traceEverOn(true);
    top->trace(tfp, 0);
    tfp->open("./build/dump.vcd");   
}

void sim_exit(){
    step_and_dump_wave();
    tfp->close();
}

int main(){
    sim_init();

    // initialize weight
    top->init_weight = 0b10011100;
    // initialize input
    top->in_activation = 0b11101011;
    top->in_psum_1 = 0b01010101;
    top->in_psum_2 = 0b00110011;

    // reset
    top->clk = 0b0; top->rst = 0b1; step_and_dump_wave();
    top->clk = 0b1; top->rst = 0b1; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();
       
    // empty cycle
    top->clk = 0b1; top->rst = 0b0; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();   

    // int8*int2
    top->precision = 0b00;
    top->clk = 0b1; top->rst = 0b0; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();   
    top->clk = 0b1; top->rst = 0b0; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();   
   
    //int8*int4
    top->precision = 0b01;
    top->clk = 0b1; top->rst = 0b0; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();   
    top->clk = 0b1; top->rst = 0b0; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();   
  
    //int8*int8
    top->precision = 0b10;
    top->clk = 0b1; top->rst = 0b0; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();   
    top->clk = 0b1; top->rst = 0b0; step_and_dump_wave();
    top->clk = 0b0; top->rst = 0b0; step_and_dump_wave();   

    // verification
    uint16_t test_weight= uint16_t(top->init_weight); 
    uint16_t test_activation = uint16_t(top->in_activation);
    uint16_t test_psum_1_in = uint16_t(top->in_psum_1);
    uint16_t test_psum_2_in = uint16_t(top->in_psum_2);
    uint16_t test_psum_1_out;

    // int16_t test_psum_2_out;
    uint16_t test_precision = 0;

    for (test_precision = 0; test_precision <=2 ; test_precision ++){
	    if(test_precision == 0) {// int8*int2
	        test_psum_1_out = ((test_weight>>6) & 0x03) * test_activation \ 
	                        + ((test_weight>>4) & 0x03) * test_activation \ 
	                        + ((test_weight>>2) & 0x03) * test_activation \ 
	                        + ((test_weight) & 0x03) * test_activation \ 
	                        + test_psum_1_in;
            // uint16_t unsigned_shifted[4];
            // for(int i=0;i<4;i++){
            //     unsigned_shifted[i] = test_activation << ((test_weight>>(i*2)) & 0x01);
            // }
            // int16_t shifted[4];
            // for(int i=0;i<4;i++){
            //     shifted[i] = test_weight[2*i+1]==0 ? int16_t(unsigned_shifted[i]) : -int16_t(unsigned_shifted[i]);
            // }
            test_psum_2_out = (test_activation << )
	    } else if (test_precision == 1) { // int8*int4
	        test_psum_1_out = ((test_weight>>4) & 0x0f) * test_activation \ 
	                        + ((test_weight) & 0x0f) * test_activation \ 
	                        + test_psum_1_in;
	    } else if (test_precision == 2) { // int8*int8
	        test_psum_1_out = test_weight * test_activation + test_psum_1_in;
	    }
        cout<<uint16_t(test_psum_1_out)<<endl;
    }
    // start conputation
    // top->clk = 0b1; top->in_activation = 10; top->in_psum_1 = 8; top->in_psum_2 = 20;
    // step_and_dump_wave();
    // top->clk = 0b0;
    // step_and_dump_wave();
    // top->clk = 0b1;
    // step_and_dump_wave();
    // top->clk = 0b0;
    // step_and_dump_wave();
    // top->clk = 0b1;
    // step_and_dump_wave();
    sim_exit();
}