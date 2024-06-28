SIMTIGHT_ROOT ?= $(realpath ../../)
PEBBLES_ROOT ?= $(realpath $(SIMTIGHT_ROOT)/pebbles)
CONFIG_H = $(SIMTIGHT_ROOT)/inc/Config.h

# Is CHERI enabled?
CHERI_EN ?= $(shell echo -n EnableCHERI \
              | cpp -P -imacros $(CONFIG_H) - | xargs)
CHERI_EN_COND = $(findstring 1, $(CHERI_EN))

# Is floating-point enabled?
FP_EN ?= $(shell echo -n EnableFP \
           | cpp -P -imacros $(CONFIG_H) - | xargs)
FP_EN_COND = $(findstring 1, $(FP_EN))

# Use Clang or GCC
USE_CLANG ?= $(shell echo -n UseClang \
              | cpp -P -imacros $(CONFIG_H) - | xargs)

# RISC-V subset
ifeq ($(CHERI_EN), 1)
  ifeq ($(FP_EN), 1)
    RV_ARCH = rv32ima_xcheri_zfinx
  else
    RV_ARCH = rv32ima_xcheri
  endif
  RV_ABI = il32pc64
else
  ifeq ($(FP_EN), 1)
    RV_ARCH = rv32ima_zfinx
  else
    RV_ARCH = rv32ima
  endif
  RV_ABI = ilp32
endif

# Compiler and linker flags
CFLAGS  = -mabi=$(RV_ABI) -march=$(RV_ARCH) -O2 -I./inc \
          -I$(SIMTIGHT_ROOT)/inc \
          -I$(PEBBLES_ROOT)/inc \
          -nostdlib -ffp-contract=off \
          -fno-builtin -ffreestanding
LDFLAGS = -melf32lriscv -G 0

ifeq ($(USE_CLANG), 1)
CFLAGS    += --target=riscv64-unknown -mno-relax
RV_CC      = riscv64-unknown-freebsd-clang++
RV_LD      = riscv64-unknown-freebsd-ld.lld
RV_OBJCOPY = riscv64-unknown-elf-objcopy
else
RV_CC      = riscv64-unknown-elf-g++
RV_LD      = riscv64-unknown-elf-ld
RV_OBJCOPY = riscv64-unknown-elf-objcopy
endif

.PHONY: help
help:
	@echo "Targets:"
	@echo "  test-cpu        test CPU core on FPGA"
	@echo "  test-cpu-sim    test CPU core in simulation"
	@echo "  test-simt       test SIMT core on FPGA"
	@echo "  test-simt-sim   test SIMT core in simulation"
	@echo "  clean           remove intermediate files"

# Helper function
v-files-for = $(patsubst %.S,%.$(1).code.v,$(wildcard $(2)/*.S)) \
              $(patsubst %.S,%.$(1).data.v,$(wildcard $(2)/*.S))

.PHONY: v-files
v-files: $(call v-files-for,cpu,I) \
         $(call v-files-for,cpu,M) \
         $(call v-files-for,cpu,F) \
         $(call v-files-for,simt,I) \
         $(call v-files-for,simt,M) \
         $(call v-files-for,simt,F) \
         $(if $(CHERI_EN_COND), , $(call v-files-for,cpu,I/NoCap)) \
         $(if $(CHERI_EN_COND), , $(call v-files-for,simt,I/NoCap)) \
         $(if $(CHERI_EN_COND), , $(call v-files-for,simt,A)) \
         $(if $(CHERI_EN_COND), $(call v-files-for,cpu,CHERI), ) \
         $(if $(CHERI_EN_COND), $(call v-files-for,simt,CHERI), ) \
         $(if $(CHERI_EN_COND), $(call v-files-for,simt,CHERI/A), )

%.code.v: %.elf
	@$(RV_OBJCOPY) -O verilog --only-section=.text $< $@

%.data.v: %.elf
	@$(RV_OBJCOPY) -O verilog --remove-section=.text \
                 --set-section-flags .bss=alloc,load,contents $< $@

.PRECIOUS: %.cpu.elf
%.cpu.elf: %.cpu.o StartCPU.o link.ld
	@$(RV_LD) $(LDFLAGS) --entry _start -T link.ld -o $@ StartCPU.o $<

.PRECIOUS: %.simt.elf
%.simt.elf: %.simt.o StartSIMT.o link.ld
	@$(RV_LD) $(LDFLAGS) --entry _start -T link.ld -o $@ StartSIMT.o $<

link.ld: $(SIMTIGHT_ROOT)/apps/Common/link.ld.h
	@cpp -P -I $(SIMTIGHT_ROOT)/inc $< > link.ld

StartCPU.o: StartCPU.cpp
	@$(RV_CC) $(CFLAGS) -Wall -c -o $@ $<

StartSIMT.o: StartSIMT.cpp
	@$(RV_CC) $(CFLAGS) -Wall -c -o $@ $<

.PRECIOUS: %.cpu.o
%.cpu.o: %.S
	@$(RV_CC) $(CFLAGS) -Wall -c -o $@ $<

.PRECIOUS: %.simt.o
%.simt.o: %.S
	@$(RV_CC) -D_TEST_SIMT_ $(CFLAGS) -Wall -c -o $@ $<

TestCPU: checkenv TestCPU.cpp
	@g++ -std=c++11 -O2 -I $(PEBBLES_ROOT)/inc -o TestCPU TestCPU.cpp \
    -I $(SIMTIGHT_ROOT)/inc \
        -fno-exceptions -lpthread -ljtag_atlantic \
    -Wl,--no-as-needed -ljtag_client \
    -L $(QUARTUS_ROOTDIR)/linux64/ -Wl,-rpath,$(QUARTUS_ROOTDIR)/linux64

TestCPUSim: TestCPU.cpp
	@g++ -std=c++11 -DSIMULATE -O2 -I $(PEBBLES_ROOT)/inc \
    -I $(SIMTIGHT_ROOT)/inc \
    -o TestCPUSim TestCPU.cpp

TestSIMT: checkenv TestSIMT.cpp
	@g++ -std=c++11 -O2 -I $(PEBBLES_ROOT)/inc -o TestSIMT TestSIMT.cpp \
    -I $(SIMTIGHT_ROOT)/inc \
    -fno-exceptions -lpthread -ljtag_atlantic \
    -Wl,--no-as-needed -ljtag_client \
    -L $(QUARTUS_ROOTDIR)/linux64/ -Wl,-rpath,$(QUARTUS_ROOTDIR)/linux64

TestSIMTSim: TestSIMT.cpp
	@g++ -std=c++11 -DSIMULATE -O2 -I $(PEBBLES_ROOT)/inc \
    -I $(SIMTIGHT_ROOT)/inc \
    -o TestSIMTSim TestSIMT.cpp

CPU_TESTS = $(if $(CHERI_EN_COND), \
  I/*.S M/*.S CHERI/*.S, \
  I/*.S I/NoCap/*.S M/*.S)
CPU_TESTS += $(if $(FP_EN_COND), F/*.S, )

SIMT_TESTS = $(if $(CHERI_EN_COND), \
  I/*.S M/*.S CHERI/*.S CHERI/A/*.S, \
  I/*.S I/NoCap/*.S M/*.S A/*.S)
SIMT_TESTS += $(if $(FP_EN_COND), F/*.S, )

.PHONY: test-cpu
test-cpu: v-files TestCPU
	@./TestCPU $(CPU_TESTS)

.PHONY: test-cpu-sim
test-cpu-sim: v-files TestCPUSim
	@./TestCPUSim $(CPU_TESTS)

.PHONY: test-simt
test-simt: v-files TestSIMT
	@./TestSIMT $(SIMT_TESTS)

.PHONY: test-simt-sim
test-simt-sim: v-files TestSIMTSim
	@./TestSIMTSim $(SIMT_TESTS)

# Raise error if QUARTUS_ROOTDIR not set
.PHONY: checkenv
checkenv:
	$(if $(value QUARTUS_ROOTDIR), , $(error Please set QUARTUS_ROOTDIR))

.PHONY: clean
clean:
	@rm -f link.ld *.o \
    I/*.o I/*.elf I/*.v \
    I/NoCap/*.o I/NoCap/*.elf I/NoCap/*.v \
    M/*.o M/*.elf M/*.v \
    A/*.o A/*.elf A/*.v \
    F/*.o F/*.elf F/*.v \
    CHERI/*.o CHERI/*.elf CHERI/*.v \
    CHERI/A/*.o CHERI/A/*.elf CHERI/A/*.v \
    TestCPU TestCPUSim TestSIMT TestSIMTSim
