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
	make -C de10-pro-e clean
	make -C sim clean
	make -C pebbles clean
	make -C rust clean

.PHONY: mrproper
mrproper: clean
	rm -rf cheri-tools/cheri
	rm -rf cheri-tools/cheribuild
	rm -rf cheri-tools/llvm-project

# Fetch submodules
.PHONY: sync
sync:
	git submodule sync
	git submodule update --init --recursive

# Docker variables
USER=$(if $(shell id -u),$(shell id -u),9001)
GROUP=$(if $(shell id -g),$(shell id -g),1000)

# Build the docker image
.PHONY: build-docker
build-docker:
	 (cd docker; docker build --build-arg UID=$(USER) --build-arg GID=$(GROUP) . --tag simtight-ubuntu2204)

# Enter the docker image
.PHONY: shell
shell: build-docker
	docker run -it --shm-size 256m --hostname simtight-ubuntu2204 -u $(USER) -v /home/$(shell whoami)/.ssh:/home/dev-user/.ssh  -v $(shell pwd):/workspace simtight-ubuntu2204:latest /bin/bash

# Build known-compatible version of the CHERI tools
.PHONY: build-cheri-tools
build-cheri-tools: 
	bash cheri-tools/build-cheri.sh	

