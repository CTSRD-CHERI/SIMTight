PEBBLES_ROOT ?= pebbles
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)

# Build the docker image
build-docker:
	 (cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) . --tag simtight-ubuntu2204)

# Enter the docker image
shell: build-docker
	docker run -it --shm-size 256m --hostname simtight-ubuntu2204 -u $(user) -v $(shell pwd):/workspace simtight-ubuntu2204:latest /bin/bash

# Fetch submodules
sync:
	git submodule sync
	git submodule update --init --recursive
	git clone https://github.com/CTSRD-CHERI/cheribuild

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
