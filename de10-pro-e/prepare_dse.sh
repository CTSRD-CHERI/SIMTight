#!/bin/bash

# DSE seems to require every verilog file to be explicitly mentioned
for VFILE in ./*.v \
             ../src/*.v \
             ../pebbles/blarney/Verilog/Altera/*.v \
             ../pebbles/src/CHERI/Verilog/*.v \
             ../pebbles/src/FloatingPoint/Wrappers/*.v

do
  echo set_global_assignment -name VERILOG_FILE $VFILE >> DE10_Pro.qsf
done
