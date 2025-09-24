# Production Deployment Guide

## 🚀 Quick Start Deployment

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Node.js 16+
- PostgreSQL (optional)
- Redis (optional)

### Step 1: Clone Repository
```bash
git clone https://github.com/koba42/production-systems.git
cd production-systems
```

### Step 2: Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Step 3: Deploy Services
```bash
# Deploy all services
./deploy.sh

# Or deploy individually
docker-compose up -d therapai_ethics
docker-compose up -d deepfake_detection
docker-compose up -d gaussian_splat
docker-compose up -d qzk_rollout
docker-compose up -d pvdm_system
docker-compose up -d nft_system
```

## 🔧 Service Configuration

### API Services
- **TherapAi Ethics**: http://localhost:5000
- **Deepfake Detection**: http://localhost:5001
- **Gaussian Splat**: http://localhost:5002
- **QZK Rollout**: http://localhost:5003
- **PVDM System**: http://localhost:5004
- **NFT System**: http://localhost:5005

### Monitoring
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Logs**: `./logs/`

## 📊 Health Checks

### API Health Endpoints
```bash
curl http://localhost:5000/health  # TherapAi Ethics
curl http://localhost:5001/health  # Deepfake Detection
curl http://localhost:5002/health  # Gaussian Splat
curl http://localhost:5003/health  # QZK Rollout
curl http://localhost:5004/health  # PVDM System
curl http://localhost:5005/health  # NFT System
```

### Service Management
```bash
# Start services
./start.sh

# Stop services
./stop.sh

# Restart services
./restart.sh

# View logs
./logs.sh

# Create backup
./backup.sh
```

## 🔒 Security Configuration

### API Authentication
All API endpoints require authentication using API keys:
- TherapAi: `tk_ethics_2025`
- Deepfake: `dk_detection_2025`
- Gaussian: `gk_splat_2025`

### CORS Configuration
Configured for:
- https://koba42.com
- https://dracattus.com
- https://api.koba42.com

## 📈 Performance Monitoring

### Metrics
- Request rate
- Response time
- Error rates
- Resource usage
- Custom business metrics

### Alerts
- Service downtime
- High error rates
- Resource exhaustion
- Security incidents

## 🔄 Backup and Recovery

### Automated Backups
- Daily database backups
- Configuration backups
- Log rotation
- Disaster recovery procedures

### Recovery Procedures
1. Stop affected services
2. Restore from backup
3. Verify data integrity
4. Restart services
5. Monitor health

---

*For production deployment assistance, contact our support team.*
