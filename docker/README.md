# SIMTight in Docker 

You can try SIMTight in Docker, which has installed all the
prerequisites for running SIMTight in simulation.

In the root of the SIMTight repo, to build the Docker image from scratch:

```sh
make build-docker 
```

This might take a long time for the first time.

To enter the Docker container:

```sh
make shell
```

This brings you to the working directory `/workspace` and you should
be able to try out all the suggested commands from the [SIMTight
README](../).
