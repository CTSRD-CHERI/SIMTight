SIMTIGHT_ROOT ?= $(realpath ../../)
SIMTIGHT_APPS_ROOT ?= $(SIMTIGHT_ROOT)/apps
PEBBLES_ROOT ?= $(realpath $(SIMTIGHT_ROOT)/pebbles)
CONFIG_H = $(SIMTIGHT_ROOT)/inc/Config.h

# Is CHERI enabled?
CHERI_EN ?= $(shell echo -n EnableCHERI \
              | cpp -P -imacros $(CONFIG_H) - | xargs)
CHERI_EN_COND = $(findstring 1, $(CHERI_EN))

# Use Clang or GCC
USE_CLANG ?= $(shell echo -n UseClang \
              | cpp -P -imacros $(CONFIG_H) - | xargs)

# Use RV32E
USE_RV32E ?= $(shell echo -n UseRV32E \
              | cpp -P -imacros $(CONFIG_H) - | xargs)

# RISC-V subset
ifeq ($(CHERI_EN), 1)
  RV_ARCH = rv32imaxcheri
  RV_ABI = il32pc64
else
  ifeq ($(USE_RV32E), 1)
    RV_ARCH = rv32ema
    RV_ABI = ilp32e
  else
    RV_ARCH = rv32ima
    RV_ABI = ilp32
  endif
endif

ifeq ($(USE_CLANG), 1)
CFLAGS     = -fuse-ld=lld -g
RV_CC      = riscv64-unknown-freebsd-clang
RV_LD      = riscv64-unknown-freebsd-ld.lld
RV_OBJCOPY = riscv64-unknown-elf-objcopy
else
CFLAGS     =
RV_CC      = riscv64-unknown-elf-gcc
RV_LD      = riscv64-unknown-elf-ld
RV_OBJCOPY = riscv64-unknown-elf-objcopy
endif

# Compiler and linker flags for code running on the SoC
CFLAGS := $(CFLAGS) -mabi=$(RV_ABI) -march=$(RV_ARCH) -O2 \
         -I $(PEBBLES_ROOT)/inc \
         -I $(SIMTIGHT_ROOT)/inc \
         -static -mcmodel=medany \
         -fvisibility=hidden -nostdlib \
         -fno-builtin-printf -ffp-contract=off \
         -fno-builtin -ffreestanding -ffunction-sections

CFILES = $(SIMTIGHT_APPS_ROOT)/Common/Start.cpp \
         $(APP_CPP) \
         $(PEBBLES_ROOT)/lib/UART/IO.cpp \
         $(PEBBLES_ROOT)/lib/memcpy.c

.PHONY: all
all: Run

code.v: app.elf
	$(RV_OBJCOPY) -O verilog --only-section=.text app.elf code.v

data.v: app.elf
	$(RV_OBJCOPY) -O verilog --remove-section=.text \
                --set-section-flags .bss=alloc,load,contents \
                --set-section-flags .sbss=alloc,load,contents \
                app.elf data.v

app.elf: link.ld $(CFILES)
	$(RV_CC) $(CFLAGS) -T link.ld -o app.elf $(CFILES)

link.ld: $(SIMTIGHT_APPS_ROOT)/Common/link.ld.h
	cpp -P -I $(SIMTIGHT_ROOT)/inc $< > link.ld

Run: checkenv code.v data.v $(RUN_CPP) $(RUN_H)
	g++ -std=c++11 -O2 -I $(PEBBLES_ROOT)/inc \
    -I $(SIMTIGHT_ROOT)/inc -o Run $(RUN_CPP) \
    -fno-exceptions -ljtag_atlantic -lpthread \
    -Wl,--no-as-needed -ljtag_client \
    -L $(QUARTUS_ROOTDIR)/linux64/ -Wl,-rpath,$(QUARTUS_ROOTDIR)/linux64

RunSim: code.v data.v $(RUN_CPP) $(RUN_H)
	g++ -DSIMULATE -O2 -I $(PEBBLES_ROOT)/inc \
    -I $(SIMTIGHT_ROOT)/inc -o RunSim $(RUN_CPP)

# Raise error if QUARTUS_ROOTDIR not set
.PHONY: checkenv
checkenv:
	$(if $(value QUARTUS_ROOTDIR), , $(error Please set QUARTUS_ROOTDIR))

.PHONY: clean
clean:
	rm -f *.o *.elf link.ld code.v data.v Run RunSim
