AvalonStreamClockCrosser: AvalonStreamClockCrosser.ip
	# Due to a bug in Intel's code gen for the clock crossing IP,
	# we use sed to ensure the DATA_WIDTH parameter is actually used
	# to define the Avalon stream width
	qsys-generate AvalonStreamClockCrosser.ip \
    --synthesis=VERILOG --family="Stratix 10" --part=1SX280HU2F50E1VG
	sed -i 's/\[7:0\] in_data/[DATA_WIDTH-1:0] in_data/g' \
    AvalonStreamClockCrosser/synth/AvalonStreamClockCrosser.v
	sed -i 's/\[7:0\] out_data/[DATA_WIDTH-1:0] out_data/g' \
    AvalonStreamClockCrosser/synth/AvalonStreamClockCrosser.v

.PHONY: clean
clean:
	rm -rf AvalonStreamClockCrosser
