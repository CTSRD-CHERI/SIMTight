SIMTIGHT_ROOT ?= $(realpath ..)
QPF = DE10_Pro.qpf
CONFIG_H = $(SIMTIGHT_ROOT)/inc/Config.h

# Are DSP hard block disabled?
DISABLE_DSP = $(shell echo -n DisableHardDSPBlocks \
               | cpp -P -imacros $(CONFIG_H) - | xargs)

.PHONY: one
one: checkenv ip
	time quartus_sh --flow compile $(QPF)

.PHONY: many
many: checkenv ip
	quartus_dse $(QPF)             \
    --seeds 200-216              \
    --launcher local             \
    --num-concurrent 4
	quartus_dse $(QPF) --report utilization
	quartus_dse $(QPF) --report fmax_summary

.PHONY: ip
ip: checkenv
	make -C $(SIMTIGHT_ROOT)/src
	make -C $(SIMTIGHT_ROOT)/boot
	cd $(SIMTIGHT_ROOT)/de10-pro && ./prepare_dse.sh
ifeq ($(DISABLE_DSP),1)
	echo "set_global_assignment -name AUTO_DSP_RECOGNITION OFF" >> DE10_Pro.qsf
	../pebbles/src/FloatingPoint/Stratix10/rmdsp-pre-ipgenerate.sh
endif
	quartus_ipgenerate DE10_Pro.qpf
	../pebbles/blarney/blarney-vendor-ip/IntelFPGA/AvalonStreamClockCrosser/post-ipgenerate.sh
ifeq ($(DISABLE_DSP),1)
	../pebbles/src/FloatingPoint/Stratix10/rmdsp-post-ipgenerate.sh
endif

.PHONY: report
report: checkenv
	quartus_dse $(QPF) --report utilization
	quartus_dse $(QPF) --report fmax_summary

.PHONY: update-mif
update-mif: checkenv
	quartus_cdb --update_mif DE10_Pro.qpf
	quartus_asm DE10_Pro.qpf

.PHONY: download-sof
download-sof: checkenv
	env -u LM_LICENSE_FILE quartus_pgm -m jtag -o "p;output_files/DE10_Pro.sof"

.PHONY: clean
clean:
	rm -rf tmp-clearbox synth_dumps qdb output_files
	rm -rf DE10_Pro_QSYS reset_release db dse*
	rm -rf ../pebbles/blarney/blarney-vendor-ip/IntelFPGA/AvalonStreamClockCrosser/AvalonStreamClockCrosser
	rm -f DE10_Pro.qws *.rpt *.csv *.mif
	ls ip/DE10_Pro_QSYS/ | grep -v -E '\.ip$$' \
    | xargs -i rm -rf ip/DE10_Pro_QSYS/{}
	ls ../pebbles/src/FloatingPoint/Stratix10/ | grep -v -E '\.ip$$' \
    | grep -v -E '\.sh$$' \
    | grep -v -E '\.py$$' \
    | xargs -i rm -rf ../pebbles/src/FloatingPoint/Stratix10/{}

# Raise error if QUARTUS_ROOTDIR not set
.PHONY: checkenv
checkenv:
	$(if $(value QUARTUS_ROOTDIR), , $(error Please set QUARTUS_ROOTDIR))
