# Social Pub/Sub Backend Server Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    sqlite \
    && rm -rf /var/cache/apk/*

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY social-pubsub-backend-server.js .
COPY social-pubsub.service.ts .

# Create data directory
RUN mkdir -p data logs

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3005

# Expose ports
EXPOSE 3005 8081

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3005/health || exit 1

# Default command
CMD ["node", "social-pubsub-backend-server.js"]

