#!/bin/bash
# chAIos Platform Docker Deployment Script

echo "🐳 Deploying chAIos Platform with Docker Compose"

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Create environment file
cat > .env << EOF
ENVIRONMENT=production
LOG_LEVEL=INFO
JWT_SECRET_KEY=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -hex 16)
GRAFANA_PASSWORD=admin
EOF

echo "🔧 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

echo "🏥 Checking service health..."
curl -f http://localhost:8000/health || echo "⚠️  API Gateway not ready yet"

echo "✅ Deployment complete!"
echo "🌐 Frontend: http://localhost:80"
echo "🔌 API: http://localhost:8000"
echo "📊 Monitoring: http://localhost:9090"
