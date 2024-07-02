SIMTIGHT_ROOT ?= $(realpath ../)
PEBBLES_ROOT ?= $(SIMTIGHT_ROOT)/pebbles
CONFIG_H = $(SIMTIGHT_ROOT)/inc/Config.h

# Is CHERI enabled?
CHERI_EN ?= $(shell echo -n EnableCHERI \
              | cpp -P -imacros $(CONFIG_H) - | xargs)
CHERI_EN_COND = $(findstring 1, $(CHERI_EN))

# Use Clang or GCC
USE_CLANG ?= $(shell echo -n UseClang \
              | cpp -P -imacros $(CONFIG_H) - | xargs)

# RISC-V subset
ifeq ($(CHERI_EN), 1)
  RV_ARCH = rv32ima_xcheri
  RV_ABI = il32pc64
else
  RV_ARCH = rv32ima
  RV_ABI = ilp32
endif

# Location of simulator
SIM = $(SIMTIGHT_ROOT)/sim

ifeq ($(USE_CLANG), 1)
CFLAGS     = --target=riscv64-unknown
RV_CC      = riscv64-unknown-freebsd-clang
RV_LD      = riscv64-unknown-freebsd-ld.lld
RV_OBJCOPY = riscv64-unknown-elf-objcopy
else
CFLAGS     = 
RV_CC      = riscv64-unknown-elf-gcc
RV_LD      = riscv64-unknown-elf-ld
RV_OBJCOPY = riscv64-unknown-elf-objcopy
GCC_VER    = $(shell $(RV_CC) --version | head -n1 \
                 | sed 's/.* //g' | cut -d'.' -f1)
ifeq ($(shell expr $(GCC_VER) \>= 13), 1)
  RV_ARCH := $(RV_ARCH)_zicsr
endif
endif

# Compiler and linker flags for code running on the SoC
CFLAGS := $(CFLAGS) -mabi=$(RV_ABI) -march=$(RV_ARCH) -Os \
         -I $(PEBBLES_ROOT)/inc -I $(SIMTIGHT_ROOT)/inc \
         -static -mcmodel=medany \
         -fvisibility=hidden -nostdlib \
         -fno-builtin-printf -ffp-contract=off \
         -fno-builtin -ffreestanding -ffunction-sections
LDFLAGS := -melf32lriscv -G 0

# C files to compile
CFILES = boot.c
OFILES = $(patsubst %.c,%.o,$(CFILES))

# Size and base of instruction memory
INSTR_BASE = $(shell echo -n MemBase \
  | cpp -P -imacros $(CONFIG_H) - | xargs)
LOG_INSTR_BYTES = $(shell echo -n CPUInstrMemLogWords \
  | cpp -P -imacros $(CONFIG_H) - | xargs)
INSTR_BYTES = $(shell echo "3 * (2 ^ $(LOG_INSTR_BYTES))" | bc)

.PHONY: all
all: $(SIM)/boot.hex ../de10-pro/boot.mif ../de10-pro-e/boot.mif

link.ld: link.ld.h
	cpp -P -I $(SIMTIGHT_ROOT)/inc link.ld.h > link.ld

$(SIM)/boot.hex: InstrMem.ihex
	./ihex-to-img.py InstrMem.ihex hex $(INSTR_BASE) 4 $(INSTR_BYTES) 1 \
    > $(SIM)/boot.hex

../de10-pro/boot.mif: InstrMem.ihex
	./ihex-to-img.py InstrMem.ihex mif $(INSTR_BASE) 4 $(INSTR_BYTES) 1 \
    > ../de10-pro/boot.mif

../de10-pro-e/boot.mif: ../de10-pro/boot.mif
	cp ../de10-pro/boot.mif ../de10-pro-e/boot.mif

InstrMem.ihex: out.elf
	$(RV_OBJCOPY) --only-section=.text -O ihex out.elf InstrMem.ihex

DataMem.ihex: out.elf
	$(RV_OBJCOPY) -O ihex --remove-section=.text \
    --set-section-flags .bss=alloc,load,contents out.elf DataMem.ihex

out.elf: entry.S $(OFILES) link.ld
	$(RV_CC) $(CFLAGS) -Wall -c -o entry.o entry.S
	$(RV_LD) $(LDFLAGS) -T link.ld -o out.elf entry.o $(OFILES)

%.o: %.c $(HFILES)
	$(RV_CC) $(CFLAGS) -Wall -c -o $@ $<

.PHONY: clean
clean:
	rm -f *.o *.elf *.ihex link.ld *.mif
