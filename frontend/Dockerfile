FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY angular.json ./
COPY tsconfig*.json ./

# Install dependencies
RUN npm install -g @angular/cli@17
RUN npm install

# Copy application files
COPY . .

# Expose port
EXPOSE 4200

# Start application
CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4200"]

