# Deploying the Neural Codec Steering Dashboard

This guide describes how to deploy the Neural Codec Steering Dashboard using Docker.

## Prerequisites

- Docker installed on your machine.

## Build the Docker Image

Run the following command in the root of the project to build the Docker image:

```bash
docker build -t neural-codec-steering .
```

## Run the Container

Once the image is built, you can run the application container:

```bash
docker run -p 8501:8501 neural-codec-steering
```

## Access the Dashboard

Open your web browser and navigate to:

`http://localhost:8501`

## Notes

- The dashboard uses GPU acceleration if available. To enable GPU support in Docker, ensure you have the NVIDIA Container Toolkit installed and run with the `--gpus all` flag:

  ```bash
  docker run --gpus all -p 8501:8501 neural-codec-steering
  ```
