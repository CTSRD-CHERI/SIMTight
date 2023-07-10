# Start with Docker 

You can try SIMTight in Docker, which has installed all the
prerequisites for running SIMTight in simulation (except a
known-working CHERI toolchain, which we hope to add soon).

To build the Docker image from scratch:

```sh
cd SIMTight; make build-docker 
```

This might take a long time for the first time.

To enter the Docker container:

```sh
cd SIMTight; make shell
```

This brings you to the working directory `/workspace` and you should
be able to try out all the commands directly!
