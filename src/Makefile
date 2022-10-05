SIMTIGHT_ROOT = $(realpath ../)
PEBBLES_ROOT = $(SIMTIGHT_ROOT)/pebbles
export BLARNEY_ROOT = $(PEBBLES_ROOT)/blarney
BLC = $(BLARNEY_ROOT)/Scripts/blc
BLC_INC = -I$(SIMTIGHT_ROOT)/inc \
          -I$(PEBBLES_ROOT)/src
BUILD_DIR = $(SIMTIGHT_ROOT)/src/build

.PHONY: all
all:
	$(BLC) -fno-cpr-anal $(BLC_FLAGS) -i$(PEBBLES_ROOT)/src $(BLC_INC) \
    -i$(PEBBLES_ROOT)/blarney/blarney-vendor-ip/Haskell \
    -hidir $(BUILD_DIR) -odir $(BUILD_DIR) Main.hs -o Main
	./Main --enable-dont-care-de-inline
	@cp *.mif $(SIMTIGHT_ROOT)/de10-pro 2> /dev/null || true
	@cp *.mif $(SIMTIGHT_ROOT)/de10-pro-e 2> /dev/null || true
	@cp *.hex $(SIMTIGHT_ROOT)/sim 2> /dev/null || true

clean:
	rm -rf $(BUILD_DIR)
	rm -f Main *.v *.mif *.hex
