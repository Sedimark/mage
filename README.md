# Sedimark Mage AI

This repository provides a Docker image for [Mage AI](https://github.com/mage-ai/mage-ai). Mage is an open-source data pipeline tool built for transforming, training, and deploying machine learning models.

## Table of Contents

- [Sedimark Mage AI](#sedimark-mage-ai)
	- [Table of Contents](#table-of-contents)
	- [Features](#features)
	- [Prerequisites](#prerequisites)
	- [Build the Docker Image](#build-the-docker-image)
	- [Pull the Docker Image](#pull-the-docker-image)
	- [Run the Docker Container](#run-the-docker-container)
		- [Access Mage](#access-mage)
	- [Repository Structure](#repository-structure)
	- [Contributing](#contributing)
		- [Pipeline Templates](#pipeline-templates)
	- [License](#license)

---

## Features

- **Pre-installed Mage AI** – Quickly start running Mage with minimal setup.
- **Python 3.11 Support** – Uses pyenv to manage Python 3.11 alongside the base Mage AI environment.
- **Custom Libraries** – Includes fleviden library and custom feature extraction utilities.
- **Volume Mount for Pipelines** – You can mount a local volume (e.g., `./default_repo`) to save your pipelines, transformations, or model files, so that they persist outside the container.
- **Port Exposed** – Port `6789` is exposed for Mage's web interface.
- **NGSI-LD Integration** – Support for NGSI-LD broker connectivity.

---

## Prerequisites

1. **Docker**: Make sure you have Docker installed on your system.
    - [Install Docker](https://docs.docker.com/get-docker/)

2. (Optional) **Git**: If you plan to clone this repo locally and make contributions, you'll need Git.

---

## Build the Docker Image

If you want to build the image locally from this repository, do the following:

1. **Clone this repository** (or fork it and then clone your fork):
```bash
git clone https://github.com/Sedimark/mage 
cd mage
```
    
2. **Build the image**:
```bash
docker build -t sedimark-mage .
```
    
- `-t sedimark-mage` tags the image locally with the name `sedimark-mage`. Feel free to use another name or tag if you prefer.

---

## Pull the Docker Image

Alternatively, you can pull the pre-built image from a registry (if available):

```bash
docker pull ghcr.io/sedimark/mage/sedimark-mage:latest
```

---

## Run the Docker Container

After building or pulling the image, you can run it:

```bash
docker run -itd --rm -p 6789:6789 -v ./default_repo:/home/src/default_repo sedimark-mage
```

- `-itd` – Runs the container in detached mode while keeping stdin open (useful for debugging if needed).
- `--rm` – Automatically removes the container when it exits.
- `-p 6789:6789` – Exposes port `6789` so you can access Mage's web interface at `http://localhost:6789`.
- `-v ./default_repo:/home/src/default_repo` – Mounts your local directory (`./default_repo`) into the container's filesystem at `/home/src/default_repo`. Any pipelines or configuration you create in that directory will be saved on your host machine.

If NGSI-LD broker is needed, deploy the broker and place it in an external network (by default should be **shared_network**), and run:

```bash
docker run -itd --rm -p 6789:6789 -v ./default_repo:/home/src/default_repo --network shared_network -e NGSI_LD_HOST=http://api-gateway:8080 sedimark-mage 
```

**api-gateway** is the name of the broker api as set in the docker compose of the broker and needs to be changed accordingly.

### Access Mage

Once the container is running, open your browser and visit:

`http://localhost:6789`

to access the Mage UI and start building pipelines.

---

## Repository Structure

The repository contains the following key components:

- **[Dockerfile](Dockerfile)** – Multi-stage Docker build that installs Python 3.11 via pyenv and custom dependencies
- **[default_repo/](default_repo/)** – Contains Mage AI pipelines, configurations, and custom utilities:
  - **[transformers/](default_repo/transformers/)** – Data transformation pipelines (e.g., [`train_hydro_series.py`](default_repo/transformers/train_hydro_series.py))
  - **[data_exporters/](default_repo/data_exporters/)** – Data export pipelines with prediction capabilities
  - **[custom/](default_repo/custom/)** – Custom pipeline components
  - **[utils/](default_repo/utils/)** – Utility modules including:
    - **[fleviden/](default_repo/utils/fleviden/)** – Federated learning utilities with Docker compose setup
    - **feature_extraction/** – Custom feature extraction modules (PCA, t-SNE, LDA, etc.)
  - **Configuration files** – `io_config.yaml`, `metadata.yaml`, `requirements.txt`

---

## Contributing

We welcome contributions! Whether you're creating new pipeline templates, updating documentation, or enhancing functionality, here's how you can get involved:

1. **Clone the repository**
```bash
git clone https://github.com/Sedimark/mage 
cd mage
```

2. **Create a new branch**
```bash
git checkout -b <branch_name>
```
- **branch_name** should be descriptive of the feature or pipeline you're adding

3. **Create or Update Pipelines**
    - If you want to contribute Mage pipelines, create or modify them inside your volume-mounted folder (`./default_repo`) while running the container.
    - You can also create templates directly in this repo if you want them to ship with the Docker image.
    - From a tested pipeline, right-click and select "Create template"
    - To add external dependencies as git submodules, run in **default_repo/utils**:
    ```bash
    git submodule add https://github.com/sedimark/<repo_name>.git path/to/submodule
    ```

4. **Commit and Push**
    - Commit your pipeline files, Dockerfile changes, or documentation updates:
    ```bash
    git add .
    git commit -m "Add new pipeline template"
    git push origin <branch_name>
    ```

5. **Submit a Pull Request**
    - Go to your branch on GitHub and create a Pull Request (PR) against the main repository's `main` branch.
    - Provide a clear description of your changes and the problem you're solving.

### Pipeline Templates

- By default, the container is configured to look into `/home/src/default_repo` for pipeline projects.
- After creating or modifying a pipeline, you can push the changes (that exist in your local `default_repo`) directly to the branch created.
- In your PR, please make sure your pipeline files are included in a logical folder structure, e.g. `default_repo/custom_templates/pipelines/<pipeline_name>`.

---

## License

This project is licensed under [MIT LICENSE](LICENSE)

---

**Thank you for using Sedimark Mage AI!** If you have any questions or issues, feel free to open an issue in this repository.