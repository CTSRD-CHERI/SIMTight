SIMTIGHT_ROOT = $(realpath ../)
PEBBLES_ROOT = $(SIMTIGHT_ROOT)/pebbles
BLARNEY_ROOT = $(PEBBLES_ROOT)/blarney
SIM_CFLAGS = -I $(PEBBLES_ROOT)/inc/ \
             -I $(SIMTIGHT_ROOT)/inc \
             -fmax-errors=1

sim: Main.cpp ../src/SIMTight.v boot.hex
	verilator -cc ../src/SIMTight.v -exe Main.cpp -o sim \
    -Wno-UNSIGNED -y $(SIMTIGHT_ROOT)/src/ \
    -y $(BLARNEY_ROOT)/Verilog \
    -y $(PEBBLES_ROOT)/src/CHERI/Verilog \
    -y $(PEBBLES_ROOT)/src/Pebbles/Instructions/Units/DivUnit \
    -y ip \
    --x-assign unique --x-initial unique \
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
