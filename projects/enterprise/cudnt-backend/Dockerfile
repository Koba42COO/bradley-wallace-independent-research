# Multi-stage build for CUDNT Backend
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Production stage
FROM node:18-alpine AS production

# Install Python and required packages for CUDNT bridge
RUN apk add --no-cache python3 py3-pip py3-numpy py3-scipy
RUN python3 -m pip install --no-cache-dir numpy scipy

# Set working directory
WORKDIR /app

# Copy built node modules
COPY --from=builder /app/node_modules ./node_modules

# Copy source code
COPY . .

# Build TypeScript
RUN npm run build

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S cudnt -u 1001

# Change ownership
RUN chown -R cudnt:nodejs /app
USER cudnt

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js

# Start the application
CMD ["node", "dist/server.js"]
