# Sedimark Mage AI

This repository provides a Docker image for [Mage AI](https://github.com/mage-ai/mage-ai). Mage is an open-source data pipeline tool built for transforming, training, and deploying machine learning models.

## Table of Contents

1. [Features](#features)
    
2. [Prerequisites](#prerequisites)
    
3. [Build the Docker Image](#build-the-docker-image)
    
4. [Pull the Docker Image](#pull-the-docker-image)
    
5. [Run the Docker Container](#run-the-docker-container)
    
6. [Contributing](#contributing)
    
7. [License](#license)
    

---

## Features

- **Pre-installed Mage AI** – Quickly start running Mage with minimal setup.
    
- **Volume Mount for Pipelines** – You can mount a local volume (e.g., `./default_repo`) to save your pipelines, transformations, or model files, so that they persist outside the container.
    
- **Port Exposed** – Port `6789` is exposed for Mage’s web interface.
    

---

## Prerequisites

1. **Docker**: Make sure you have Docker installed on your system.
    
    - Install Docker

2. (Optional) **Git**: If you plan to clone this repo locally and make contributions, you’ll need Git.
    

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
    
	- -t sedimark-mage tags the image locally with the name `sedimark-mage`. Feel free to use another name or tag if you prefer.

---

## Run the Docker Container

After building or pulling the image, you can run it:


```bash
docker run -itd --rm -p 6789:6789 -v ./default_repo:/home/src/default_repo sedimark-mage
```

- `-itd` – Runs the container in detached mode while keeping stdin open (useful for debugging if needed).
    
- `--rm` – Automatically removes the container when it exits.
    
- `-p 6789:6789` – Exposes port `6789` so you can access Mage’s web interface at `http://localhost:6789`.
    
- `-v ./default_repo:/home/src/default_repo` – Mounts your local directory (`./default_repo`) into the container’s filesystem at `/home/src/default_repo`. Any pipelines or configuration you create in that directory will be saved on your host machine.

If NGSI-LD broker is needed, deploy the broker and place it in an external network, by default should be **sharde_network**, and run:

```bash
docker run -itd --rm -p 6789:6789 -v ./default_repo:/home/src/default_repo --network shared_network -e NGSI_LD_HOST=http://api-gateway:8080 -e sedimark-mage 
```

**api-gateway** is the name of the broker api as set in the docker compose of the broker and needs to be changed accordingly

### Access Mage

Once the container is running, open your browser and visit:

`http://localhost:6789`

to access the Mage UI and start building pipelines.

---

## Contributing

We welcome contributions! Whether you’re creating new pipeline templates, updating documentation, or enhancing functionality, here’s how you can get involved:

1. **Clone the repository**
	```bash
	git clone https://github.com/Sedimark/mage 
	cd mage
	```
2. Create a new branch
	```bash
	git checkout -b <branch_name>
	```
	- **branch_name** should be the name of the pipeline that will be integrated
3. **Create or Update Pipelines**
    
    - If you want to contribute Mage pipelines, create or modify them inside your volume-mounted folder (`./default_repo`) while running the container.
        
    - You can also create templates directly in this repo if you want them to ship with the Docker image.

	- From a tested pipeline right click and do create template

	- To add external dependencies run in **default_repo/utils**:
		```bash
		git submodule add https://github.com/sedimark/<repo_name>.git path/to/submodule
		```
        
4. **Commit and Push**
    
    - Commit your pipeline files, Dockerfile changes, or documentation updates:
	    
    ```bash
		git add . git commit -m "Add new pipeline template" 
		git push origin main
	```

        
5. **Submit a Pull Request**
    
    - Go to your branch on GitHub and create a Pull Request (PR) against the main repository’s `main` (or `master`) branch.
        
    - Provide a clear description of your changes, the problem you’re solving.
        

### Pipeline Templates

- By default, the container is configured to look into `/home/src/default_repo` for pipeline projects.
    
- After creating or modifying a pipeline, you can push the changes (that exist in your local `default_repo`) directly to the branch created.
    
- In your PR, please make sure your pipeline files are included in a logical folder structure, e.g. `default_repo/custom_templates/pipelines/<pipeline_name>`.

---

**Thank you for using Sedimark Mage AI!** If you have any questions or issues, feel free to open an issue in this repository.
