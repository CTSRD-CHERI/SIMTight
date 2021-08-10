#include <stdint.h>
#include <verilated.h>

#include <Sim/DRAM.h>
#include <Sim/JTAGUART.h>
#include "VSIMTight.h"

VSIMTight *top;
vluint64_t main_time = 0;

// Called by $time in Verilog
double sc_time_stamp () {
  return main_time;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  // Instantiate SoC
  top = new VSIMTight;
  
  // DRAM simulator
  DRAM dram;
  dram.ifc.readdata = top->in0_socDRAMIns_avl_dram_readdata;
  dram.ifc.waitrequest = &top->in0_socDRAMIns_avl_dram_waitrequest;
  dram.ifc.readdatavalid = &top->in0_socDRAMIns_avl_dram_readdatavalid;
  dram.ifc.writedata = top->out_socDRAMOuts_avl_dram_writedata;

  // JTAG UART simulator
  JTAGUART uart;
  uart.ifc.readdata = &top->in0_socUARTIns_avl_jtaguart_readdata;
  uart.ifc.waitrequest = &top->in0_socUARTIns_avl_jtaguart_waitrequest;
  uart.ifc.writedata = &top->out_socUARTOuts_avl_jtaguart_writedata;
  uart.ifc.address = &top->out_socUARTOuts_avl_jtaguart_address;
  uart.ifc.read = &top->out_socUARTOuts_avl_jtaguart_read;
  uart.ifc.write = &top->out_socUARTOuts_avl_jtaguart_write;

  while (!Verilated::gotFinish()) {
    top->reset = main_time < 1;
    if (top->reset == 0) {
      dram.ifc.address = top->out_socDRAMOuts_avl_dram_address;
      dram.ifc.burstcount = top->out_socDRAMOuts_avl_dram_burstcount;
      dram.ifc.byteen = top->out_socDRAMOuts_avl_dram_byteen;
      dram.ifc.read = top->out_socDRAMOuts_avl_dram_read;
      dram.ifc.write = top->out_socDRAMOuts_avl_dram_write;
      dram.tick();
      uart.tick();
    }
    top->clock = 0; top->eval();
    top->clock = 1; top->eval();
    main_time++;
  }

  top->final();
  delete top;

  return 0;
}
