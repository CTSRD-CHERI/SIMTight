SIMTIGHT_ROOT ?= $(realpath ..)
NUM_SEEDS ?= 4
QPF = DE10_Pro.qpf

.PHONY: one
one: checkenv ip
	time quartus_sh --flow compile $(QPF)

.PHONY: many
many: checkenv ip
	quartus_dse $(QPF)             \
    --num-seeds $$(($(NUM_SEEDS) - 1)) \
    --launcher local                   \
    --num-concurrent 4
	quartus_dse $(QPF) --report utilization
	quartus_dse $(QPF) --report fmax_summary

.PHONY: ip
ip: checkenv
	make -C $(SIMTIGHT_ROOT)/src
	make -C $(SIMTIGHT_ROOT)/boot
	quartus_ipgenerate DE10_Pro.qpf
	../pebbles/blarney/blarney-vendor-ip/IntelFPGA/AvalonStreamClockCrosser/post-ipgenerate.sh

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
	quartus_pgm -m jtag -o "p;output_files/DE10_Pro.sof"

.PHONY: clean
clean:
	rm -rf tmp-clearbox synth_dumps qdb output_files
	rm -rf DE10_Pro_QSYS reset_release db dse*
	rm -rf ../pebbles/blarney/blarney-vendor-ip/IntelFPGA/AvalonStreamClockCrosser/AvalonStreamClockCrosser
	rm -f DE10_Pro.qws *.rpt *.csv *.mif
	ls ip/DE10_Pro_QSYS/ | grep -v -E '\.ip$$' \
    | xargs -i rm -rf ip/DE10_Pro_QSYS/{}

# Raise error if QUARTUS_ROOTDIR not set
.PHONY: checkenv
checkenv:
	$(if $(value QUARTUS_ROOTDIR), , $(error Please set QUARTUS_ROOTDIR))
