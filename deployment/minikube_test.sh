#!/usr/bin/env bash
# Local Minikube test — validates all K8s configs apply cleanly.
# vLLM won't run without a GPU, but this confirms the orchestration
# layer (pods, services, HPA, configmap) is wired correctly.
set -e

echo "==> Starting Minikube..."
minikube start --cpus=4 --memory=8192

echo "==> Building FastAPI image inside Minikube's Docker daemon..."
eval $(minikube docker-env)
docker build -t modeledge-api:latest .

echo "==> Applying configs..."
kubectl apply -f deployment/configmap.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/hpa.yaml

echo "==> Waiting for API pod to be ready (vLLM will stay Pending without GPU)..."
kubectl rollout status deployment/modeledge-api --timeout=120s

echo "==> Pod status:"
kubectl get pods -l app=modeledge

echo "==> Services:"
kubectl get services

echo "==> HPA:"
kubectl get hpa

echo "==> Forwarding port 8000 → localhost:8000 (Ctrl+C to stop)"
kubectl port-forward service/modeledge-api 8000:80
