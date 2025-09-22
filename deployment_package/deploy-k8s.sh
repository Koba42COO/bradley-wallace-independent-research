#!/bin/bash
# chAIos Platform Kubernetes Deployment Script

echo "☸️  Deploying chAIos Platform to Kubernetes"

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed"
    exit 1
fi

# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets (you should modify these for production)
kubectl create secret generic chaios-secrets   --from-literal=jwt-secret-key=$(openssl rand -hex 32)   --from-literal=db-password=$(openssl rand -hex 16)   --from-literal=grafana-password=admin   --namespace chaios-platform

# Deploy infrastructure
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/pvc.yaml

# Deploy databases
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/postgres-service.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/redis-service.yaml

echo "⏳ Waiting for databases to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres --namespace chaios-platform --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis --namespace chaios-platform --timeout=300s

# Deploy application services
kubectl apply -f k8s/api-gateway-deployment.yaml
kubectl apply -f k8s/api-gateway-service.yaml
kubectl apply -f k8s/knowledge-system-deployment.yaml
kubectl apply -f k8s/knowledge-system-service.yaml
kubectl apply -f k8s/polymath-brain-deployment.yaml
kubectl apply -f k8s/polymath-brain-service.yaml
kubectl apply -f k8s/cudnt-accelerator-deployment.yaml
kubectl apply -f k8s/cudnt-accelerator-service.yaml

# Deploy frontend
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml

# Deploy ingress
kubectl apply -f k8s/ingress.yaml

echo "⏳ Waiting for services to be ready..."
sleep 60

echo "🏥 Checking service health..."
kubectl get pods --namespace chaios-platform

echo "✅ Kubernetes deployment complete!"
echo "🌐 Check ingress for external access"
