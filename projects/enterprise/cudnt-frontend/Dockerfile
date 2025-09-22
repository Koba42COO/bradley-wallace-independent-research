# Multi-stage build for CUDNT Frontend
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build Angular app
RUN npm run build --prod

# Production stage with Nginx
FROM nginx:alpine

# Copy built app to nginx
COPY --from=builder /app/dist/cudnt-frontend /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create non-root user
RUN addgroup -g 1001 -S nginx
RUN adduser -S cudnt -u 1001 -G nginx

# Change ownership
RUN chown -R cudnt:nginx /usr/share/nginx/html
RUN chown -R cudnt:nginx /var/cache/nginx
RUN chown -R cudnt:nginx /var/log/nginx
RUN chown -R cudnt:nginx /etc/nginx/conf.d
RUN touch /var/run/nginx.pid
RUN chown -R cudnt:nginx /var/run/nginx.pid

USER cudnt

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
