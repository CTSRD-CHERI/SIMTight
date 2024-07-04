SIMTIGHT_ROOT = $(realpath ../)
PEBBLES_ROOT = $(SIMTIGHT_ROOT)/pebbles
BLARNEY_ROOT = $(PEBBLES_ROOT)/blarney
SIM_CFLAGS = -I $(PEBBLES_ROOT)/inc/ \
             -I $(SIMTIGHT_ROOT)/inc \
             -fmax-errors=1
CFILES = $(PEBBLES_ROOT)/src/FloatingPoint/Sim/FP.c

sim: Main.cpp ../src/SIMTight.v boot.hex
	verilator -cc ../src/SIMTight.v $(CFILES) -exe Main.cpp -o sim \
    -Wno-UNSIGNED -Wno-CMPCONST -y $(SIMTIGHT_ROOT)/src/ \
    -y $(BLARNEY_ROOT)/Verilog \
    -y $(PEBBLES_ROOT)/src/CHERI/Verilog \
    -y $(PEBBLES_ROOT)/src/FloatingPoint/Wrappers \
    -y $(PEBBLES_ROOT)/src/FloatingPoint/Sim \
    -y $(PEBBLES_ROOT)/blarney/blarney-vendor-ip/Sim/ \
    --x-assign unique --x-initial unique -D_SIM_ \
    -CFLAGS "$(SIM_CFLAGS)"
	make -C obj_dir -j -f VSIMTight.mk sim
	cp obj_dir/sim .
	rm -rf obj_dir
	ls *.hex | xargs -i basename {} .hex | xargs -i cp {}.hex {}.mif

../src/SIMTight.v:
	make -C ../src

boot.hex:
	make -C ../boot

.PHONY: clean 
clean:
	rm -rf sim *.hex *.mif obj_dir
