## Automated Deployment Script: `deploy_application.launch`

The `deploy_application.launch` script automates the deployment of the entire application on a fresh Linux machine. 
It installs necessary dependencies, sets up Minikube as a local Kubernetes cluster, builds Docker images, 
and deploys all services (YOLO Service, MLFlow Service, and MongoDB).

### Usage

1. Ensure the script is executable:
   ```bash
   chmod +x deploy_application.launch
   ```

2. Execute the script:
   ```bash
   ./deploy_application.launch
   ```

### Features of the Script

- Installs Docker, kubectl, and Minikube on the host machine.
- Starts Minikube and configures it to run Kubernetes services.
- Builds Docker images for YOLO and MLFlow services within Minikube's Docker environment.
- Deploys Kubernetes resources (namespace, deployments, services, ingress).
- Verifies that all services are running and provides endpoint URLs.

### Minikube Cluster Information

- YOLO Service: Accessible at `http://<minikube-ip>:8000`
- MLFlow Service: Accessible at `http://<minikube-ip>:8001`
- MongoDB: Internally available within the Kubernetes cluster.

This script simplifies the setup and deployment process, making it easy to replicate and test the application in a controlled environment.





## AI SaaS: YOLO and MLFlow Microservices

This project provides an AI-based SaaS application for object classification using YOLOv5, with metrics tracking and logging via MLFlow and MongoDB.

## Features

- **YOLO Service:** Handles image predictions and logs metrics.
- **MLFlow Service:** Tracks metrics and parameters for experiments.
- **MongoDB:** Serves as the database for both services.

## Architecture

- **Microservices:** YOLO and MLFlow services run independently.
- **Database:** MongoDB stores prediction and metrics data.
- **Containerization:** Docker Compose ensures seamless service orchestration.

## Services

### 1. YOLO Service
- **Description:** Performs object detection and classification using YOLOv5.
- **Base URL:** `http://localhost:8000`

### 2. MLFlow Service
- **Description:** Logs metrics and parameters for ML experiments.
- **Base URL:** `http://localhost:8001`

### 3. MongoDB
- **Description:** Stores logs and metrics for YOLO and MLFlow services.
