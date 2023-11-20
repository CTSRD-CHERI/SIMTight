SIMTIGHT_ROOT ?= $(realpath ../)
PEBBLES_ROOT  ?= $(realpath $(SIMTIGHT_ROOT)/pebbles)
EXAMPLE_BASE  ?= target/riscv32ima-unknown-none-elf/release/examples/
EXAMPLE       ?= vec_add
TARGET_BINARY ?= $(EXAMPLE_BASE)/$(EXAMPLE)
RV_OBJCOPY     = riscv64-unknown-elf-objcopy

.PHONY: all
all: $(EXAMPLE)-code.v $(EXAMPLE)-data.v

$(EXAMPLE)-code.v: $(TARGET_BINARY)
	@$(RV_OBJCOPY) -O verilog --only-section=.text $(TARGET_BINARY) $@

$(EXAMPLE)-data.v: $(TARGET_BINARY)
	@$(RV_OBJCOPY) -O verilog --remove-section=.text \
                --set-section-flags .bss=alloc,load,contents \
                --set-section-flags .sbss=alloc,load,contents \
                $(TARGET_BINARY) $@

.PHONY: run-sim
run-sim: RunSim $(EXAMPLE)-code.v $(EXAMPLE)-data.v
	./RunSim $(EXAMPLE)

.PHONY: run
run: RunSim $(EXAMPLE)-code.v $(EXAMPLE)-data.v
	./Run $(EXAMPLE)

Run: checkenv Run.cpp
	@g++ -std=c++11 -O2 -I $(PEBBLES_ROOT)/inc \
    -I $(SIMTIGHT_ROOT)/inc -o Run Run.cpp \
    -fno-exceptions -ljtag_atlantic -lpthread \
    -Wl,--no-as-needed -ljtag_client \
    -L $(QUARTUS_ROOTDIR)/linux64/ -Wl,-rpath,$(QUARTUS_ROOTDIR)/linux64

RunSim: Run.cpp
	@g++ -DSIMULATE -O2 -I $(PEBBLES_ROOT)/inc \
    -I $(SIMTIGHT_ROOT)/inc -o RunSim Run.cpp

$(TARGET_BINARY): memory.x src/prims/config.rs src/*.rs
	cargo build --example $(EXAMPLE) --release

memory.x: memory.x.h
	cpp -P -I $(SIMTIGHT_ROOT)/inc memory.x.h | grep '.' > memory.x

src/prims/config.rs: config.rs.h
	mkdir -p src/prims/
	cpp -P -I $(SIMTIGHT_ROOT)/inc -I $(SIMTIGHT_ROOT)/pebbles/inc config.rs.h \
    | grep '.' > src/prims/config.rs

# Raise error if QUARTUS_ROOTDIR not set
.PHONY: checkenv
checkenv:
	$(if $(value QUARTUS_ROOTDIR), , $(error Please set QUARTUS_ROOTDIR))

.PHONY: clean
clean:
	cargo clean
	rm -f memory.x Run RunSim Cargo.lock src/prims/config.rs *.v
