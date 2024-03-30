#include "verilated.h"
#include "verilated_vcd_c.h"

#include <Vpe.h>

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

    top->weight = 0b010;
    top->clk = 0b1; top->in_activation = 10; top->in_psum_1 = 8; top->in_psum_2 = 20;
    step_and_dump_wave();
    top->clk = 0b0;
    step_and_dump_wave();
    top->clk = 0b1;
    step_and_dump_wave();
    top->clk = 0b0;
    step_and_dump_wave();
    top->clk = 0b1;
    step_and_dump_wave();
    // top->en=0b1; top->x = 0b00000001; step_and_dump_wave();
    //              top->x = 0b00000010; step_and_dump_wave();
    //              top->x = 0b00000100; step_and_dump_wave();
    //              top->x = 0b00001000; step_and_dump_wave();
    //              top->x = 0b00010000; step_and_dump_wave();
    //              top->x = 0b00100000; step_and_dump_wave();
    //              top->x = 0b01000000; step_and_dump_wave();
    //              top->x = 0b10000000; step_and_dump_wave();
    sim_exit();
}