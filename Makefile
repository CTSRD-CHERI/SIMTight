PEBBLES_ROOT ?= pebbles

.PHONY: verilog
verilog:
	make -C src

.PHONY: sim
sim: verilog
	make -C sim

.PHONY: clean
clean:
	make -C boot clean
	make -C apps clean
	make -C src clean
	make -C de10-pro clean
	make -C sim clean
	make -C pebbles clean
