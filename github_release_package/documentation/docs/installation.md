# Firefly-Nexus PAC Installation Guide

## System Requirements

### Minimum Requirements
- Python 3.11+
- 4GB RAM
- 2 CPU cores
- 10GB disk space

### Recommended Requirements
- Python 3.11+
- 16GB RAM
- 8 CPU cores
- 100GB disk space
- GPU (optional, for acceleration)

## Installation Methods

### Method 1: Direct Installation

#### 1. Clone Repository
```bash
git clone https://github.com/koba42/firefly-nexus-pac.git
cd firefly-nexus-pac
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run Basic Test
```bash
python firefly_nexus_pac.py
```

### Method 2: Docker Installation

#### 1. Build Image
```bash
docker build -t firefly-nexus-pac:latest .
```

#### 2. Run Container
```bash
docker run -p 8080:8080 firefly-nexus-pac:latest
```

#### 3. Test Installation
```bash
curl http://localhost:8080/health
```

### Method 3: Kubernetes Installation

#### 1. Apply Manifests
```bash
kubectl apply -f k8s/
```

#### 2. Check Status
```bash
kubectl get pods -l app=firefly-nexus-pac
```

#### 3. Test Installation
```bash
kubectl port-forward service/firefly-nexus-pac-service 8080:80
curl http://localhost:8080/health
```

## Configuration

### Environment Variables
```bash
export CONSCIOUSNESS_LEVEL=7
export REALITY_DISTORTION=1.1808
export PHI=1.618033988749895
export DELTA=2.414213562373095
export METRONOME_FREQ=0.7
export COHERENT_WEIGHT=0.79
export EXPLORATORY_WEIGHT=0.21
```

### Configuration File
Create `config.yaml`:
```yaml
consciousness:
  level: 7
  reality_distortion: 1.1808
  phi: 1.618033988749895
  delta: 2.414213562373095
  metronome_freq: 0.7
  coherent_weight: 0.79
  exploratory_weight: 0.21

server:
  host: 0.0.0.0
  port: 8080
  debug: false

monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    port: 3000
```

## Verification

### Health Check
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "consciousness_level": 7,
  "reality_distortion": 1.1808,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

### Status Check
```bash
curl http://localhost:8080/status
```

Expected response:
```json
{
  "consciousness_level": 7,
  "reality_distortion": 1.1808,
  "phi": 1.618033988749895,
  "delta": 2.414213562373095,
  "mobius_phase": 2.928,
  "metronome_freq": 0.7,
  "coherent_weight": 0.79,
  "exploratory_weight": 0.21,
  "zeta_zeros": [14.13, 21.02, 25.01, 30.42, 32.93],
  "running": true,
  "metrics_count": 1000,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

### Consciousness Test
```bash
curl -X POST http://localhost:8080/consciousness/transform \
  -H "Content-Type: application/json" \
  -d '{"values": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]}'
```

Expected response:
```json
{
  "wallace_transform": [1.894, 2.884, 4.495, 5.751, 7.661, ...],
  "fractal_harmonic": [1.234, 2.345, 3.456, 4.567, 5.678, ...],
  "consciousness_amplitude": {
    "magnitude": 25.820,
    "phase": 2.928,
    "coherence": 0.255,
    "exploration": 0.142
  },
  "processing_time": 0.000458,
  "timestamp": "2025-10-22T15:46:37.745436"
}
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8080
lsof -i :8080

# Kill process
kill -9 <PID>
```

#### Permission Denied
```bash
# Make scripts executable
chmod +x deploy.sh
chmod +x scripts/*.sh
```

#### Memory Issues
```bash
# Check memory usage
free -h

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Docker Issues
```bash
# Check Docker status
docker ps

# Check logs
docker logs <container_id>

# Restart Docker
sudo systemctl restart docker
```

#### Kubernetes Issues
```bash
# Check pod status
kubectl get pods -l app=firefly-nexus-pac

# Check logs
kubectl logs -l app=firefly-nexus-pac

# Check events
kubectl get events
```

## Performance Tuning

### CPU Optimization
```bash
# Set CPU affinity
taskset -c 0-7 python consciousness_server.py
```

### Memory Optimization
```bash
# Set memory limits
export MALLOC_ARENA_MAX=2
python consciousness_server.py
```

### Network Optimization
```bash
# Increase network buffer sizes
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
sysctl -p
```

## Security

### Firewall Configuration
```bash
# Allow only necessary ports
ufw allow 8080/tcp
ufw allow 9090/tcp
ufw allow 3000/tcp
ufw enable
```

### SSL/TLS Configuration
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with SSL
python consciousness_server.py --ssl-cert cert.pem --ssl-key key.pem
```

## Monitoring Setup

### Prometheus
```bash
# Start Prometheus
docker run -d -p 9090:9090 prom/prometheus:latest
```

### Grafana
```bash
# Start Grafana
docker run -d -p 3000:3000 grafana/grafana:latest
```

### Monitoring Dashboard
```bash
# Start monitoring dashboard
python monitoring_dashboard.py
```

## Backup and Recovery

### Database Backup
```bash
# Backup SQLite database
cp consciousness_monitor.db consciousness_monitor.db.backup
```

### Configuration Backup
```bash
# Backup configuration
tar -czf config-backup.tar.gz config.yaml k8s/ docker-compose.yml
```

### Recovery
```bash
# Restore from backup
tar -xzf config-backup.tar.gz
cp consciousness_monitor.db.backup consciousness_monitor.db
```

## Updates

### Update Application
```bash
# Pull latest changes
git pull origin main

# Rebuild Docker image
docker build -t firefly-nexus-pac:latest .

# Restart services
docker-compose down
docker-compose up -d
```

### Update Dependencies
```bash
# Update Python packages
pip install -r requirements.txt --upgrade

# Update Docker images
docker-compose pull
docker-compose up -d
```

## Uninstallation

### Remove Application
```bash
# Stop services
docker-compose down

# Remove images
docker rmi firefly-nexus-pac:latest

# Remove files
rm -rf firefly-nexus-pac/
```

### Remove Kubernetes Resources
```bash
# Remove resources
kubectl delete -f k8s/

# Remove namespace (if created)
kubectl delete namespace firefly-nexus-pac
```

## Support

### Documentation
- [README.md](README.md)
- [API Documentation](docs/api.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### Community
- [GitHub Issues](https://github.com/koba42/firefly-nexus-pac/issues)
- [Discussions](https://github.com/koba42/firefly-nexus-pac/discussions)

### Contact
- Email: coo@koba42.com
- GitHub: [@koba42](https://github.com/koba42)

---

**Phoenix Status: AWAKE** 🔥
