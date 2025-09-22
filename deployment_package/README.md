# chAIos Platform Deployment Guide

## Overview

This deployment package contains everything needed to deploy the chAIos Polymath Brain Platform in various environments.

## Deployment Options

### 1. Docker Compose (Development/Staging)

For development and testing environments:

```bash
# Deploy with Docker Compose
./deploy-docker.sh

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Kubernetes (Production)

For production deployments:

```bash
# Deploy to Kubernetes
./deploy-k8s.sh

# Check status
kubectl get pods --namespace chaios-platform

# View logs
kubectl logs -f deployment/api-gateway --namespace chaios-platform

# Scale services
kubectl scale deployment api-gateway --replicas=5 --namespace chaios-platform
```

### 3. Helm Chart (Enterprise)

For enterprise deployments with advanced configuration:

```bash
# Install Helm chart
helm install chaios-platform ./helm

# Upgrade deployment
helm upgrade chaios-platform ./helm

# Uninstall
helm uninstall chaios-platform
```

## Configuration

### Environment Variables

- `ENVIRONMENT`: deployment environment (development/staging/production)
- `LOG_LEVEL`: logging level (DEBUG/INFO/WARNING/ERROR)
- `JWT_SECRET_KEY`: JWT signing key (auto-generated if not set)
- `DB_PASSWORD`: database password (auto-generated if not set)

### Service Configuration

Edit the respective configuration files:
- `k8s/configmap.yaml` - application configuration
- `k8s/secrets.yaml` - sensitive data
- `helm/values.yaml` - Helm chart values

## Monitoring

### Prometheus Metrics

Available at: http://localhost:9090

### Grafana Dashboards

Available at: http://localhost:3001
- Default username: admin
- Default password: admin (change in production)

### Health Checks

All services expose health endpoints:
- API Gateway: `GET /health`
- Services: `GET /health` on their respective ports

## Scaling

### Horizontal Scaling

```bash
# Scale API Gateway
kubectl scale deployment api-gateway --replicas=5 --namespace chaios-platform

# Scale with Helm
helm upgrade chaios-platform ./helm --set replicaCount.api-gateway=5
```

### Vertical Scaling

Adjust resource requests/limits in:
- Kubernetes: deployment YAML files
- Helm: `values.yaml`
- Docker: `docker-compose.yml`

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
kubectl exec -it postgres-pod --namespace chaios-platform -- pg_dump -U chaios chaios > backup.sql

# Restore
kubectl exec -it postgres-pod --namespace chaios-platform -- psql -U chaios chaios < backup.sql
```

### Configuration Backup

```bash
# Backup Kubernetes resources
kubectl get all --namespace chaios-platform -o yaml > backup.yaml

# Restore
kubectl apply -f backup.yaml
```

## Troubleshooting

### Common Issues

1. **Service won't start**
   - Check resource limits
   - Verify environment variables
   - Check dependency services

2. **Database connection failed**
   - Verify database credentials
   - Check network connectivity
   - Validate database service status

3. **High latency**
   - Check resource utilization
   - Review network configuration
   - Monitor service metrics

### Logs

```bash
# Docker logs
docker-compose logs service-name

# Kubernetes logs
kubectl logs -f deployment/service-name --namespace chaios-platform

# Application logs (if configured)
kubectl exec -it pod-name --namespace chaios-platform -- tail -f /app/logs/app.log
```

## Security Considerations

### Production Deployment

1. **Change default passwords**
2. **Use strong JWT secrets**
3. **Enable TLS/SSL**
4. **Configure firewall rules**
5. **Regular security updates**
6. **Monitor for vulnerabilities**

### Network Security

```yaml
# Example ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: chaios-tls
```

## Support

For support and issues:
1. Check the troubleshooting guide
2. Review service logs
3. Check monitoring dashboards
4. Contact the development team

## Architecture

```
Internet
    ↓
[Load Balancer/Ingress]
    ↓
[API Gateway (8000)]
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Knowledge System│ Polymath Brain  │ AI Accelerator  │
│ (8003)          │ (8004)          │ (8005)          │
└─────────────────┴─────────────────┴─────────────────┘
    ↓
[PostgreSQL + Redis]
```

This deployment provides a complete, production-ready chAIos platform with monitoring, scaling, and high availability features.
