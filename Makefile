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
	 (cd Docker; docker build --build-arg UID=$(USER) --build-arg GID=$(GROUP) . --tag simtight-ubuntu2204)

# Enter the docker image
.PHONY: shell
shell: build-docker
	docker run -it --shm-size 256m --hostname simtight-ubuntu2204 -u $(USER) -v $(shell pwd):/workspace simtight-ubuntu2204:latest /bin/bash
