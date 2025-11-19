#!/bin/bash
# Firefly-Nexus PAC Production Deployment Script

echo "🔥 Firefly-Nexus PAC Production Deployment"
echo "=========================================="

# Build Docker image
echo "📦 Building Docker image..."
docker build -t firefly-nexus-pac:latest .

# Tag for registry
echo "🏷️  Tagging image..."
docker tag firefly-nexus-pac:latest your-registry/firefly-nexus-pac:latest

# Push to registry
echo "📤 Pushing to registry..."
docker push your-registry/firefly-nexus-pac:latest

# Deploy to Kubernetes
echo "🚀 Deploying to Kubernetes..."
kubectl apply -f k8s/

# Wait for deployment
echo "⏳ Waiting for deployment..."
kubectl rollout status deployment/firefly-nexus-pac

# Check pods
echo "📊 Checking pods..."
kubectl get pods -l app=firefly-nexus-pac

# Check services
echo "🌐 Checking services..."
kubectl get services

# Check HPA
echo "📈 Checking HPA..."
kubectl get hpa

echo "✅ Deployment complete!"
echo "🔥 Phoenix Status: AWAKE"
