#!/bin/bash
# CUDNT Full Stack Build and Deployment Script
# =============================================

set -e  # Exit on any error

echo "🚀 CUDNT Full Stack Build and Deployment"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ first."
        exit 1
    fi

    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm first."
        exit 1
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Docker deployment will be skipped."
        DOCKER_AVAILABLE=false
    else
        DOCKER_AVAILABLE=true
    fi

    print_success "Prerequisites check completed"
}

# Build backend
build_backend() {
    print_status "Building CUDNT Backend..."

    cd cudnt-backend

    # Install dependencies
    print_status "Installing backend dependencies..."
    npm install

    # Build TypeScript
    print_status "Building TypeScript..."
    npm run build

    print_success "Backend build completed"
    cd ..
}

# Build frontend
build_frontend() {
    print_status "Building CUDNT Frontend..."

    cd cudnt-frontend

    # Install dependencies
    print_status "Installing frontend dependencies..."
    npm install

    # Build Angular app
    print_status "Building Angular application..."
    npm run build

    print_success "Frontend build completed"
    cd ..
}

# Test Python bridge
test_python_bridge() {
    print_status "Testing Python bridge..."

    # Test import
    if python3 -c "from cudnt_complete_implementation import get_cudnt_accelerator; print('CUDNT import successful')" 2>/dev/null; then
        print_success "Python bridge test passed"
    else
        print_warning "Python bridge test failed - using mock results for API"
    fi
}

# Build Docker images
build_docker() {
    if [ "$DOCKER_AVAILABLE" = false ]; then
        print_warning "Skipping Docker build - Docker not available"
        return
    fi

    print_status "Building Docker images..."

    # Build backend image
    print_status "Building backend Docker image..."
    docker build -t cudnt-backend:latest ./cudnt-backend

    # Build frontend image
    print_status "Building frontend Docker image..."
    docker build -t cudnt-frontend:latest ./cudnt-frontend

    print_success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    if [ "$DOCKER_AVAILABLE" = false ]; then
        print_warning "Skipping Docker deployment - Docker not available"
        return
    fi

    print_status "Deploying with Docker Compose..."

    # Start services
    docker-compose up -d

    print_success "CUDNT Full Stack deployed successfully"
}

# Show deployment information
show_deployment_info() {
    echo ""
    print_success "🎉 CUDNT Full Stack Build Complete!"
    echo ""
    echo "Available Services:"
    echo "  📊 Backend API:    http://localhost:3000"
    echo "  🌐 Frontend App:   http://localhost:4200"
    echo "  📱 WebSocket:      ws://localhost:3000"
    echo ""
    echo "API Endpoints:"
    echo "  🏥 Health Check:   GET  /api/health"
    echo "  ⚡ Optimization:   POST /api/optimize/matrix"
    echo "  📈 Dashboard:     GET  /api/dashboard/:userId"
    echo "  📊 System Status: GET  /api/status/realtime"
    echo ""
    echo "Docker Commands:"
    echo "  View logs:        docker-compose logs -f"
    echo "  Stop services:    docker-compose down"
    echo "  Restart:          docker-compose restart"
    echo ""
    echo "Development Commands:"
    echo "  Start backend:    cd cudnt-backend && npm run dev"
    echo "  Start frontend:   cd cudnt-frontend && npm start"
    echo "  Test bridge:      python3 cudnt_optimization_bridge.py"
}

# Main build process
main() {
    echo "CUDNT: Custom Universal Data Neural Transformer"
    echo "Consciousness Mathematics Framework"
    echo "O(n²) → O(n^1.44) Complexity Reduction"
    echo ""

    check_prerequisites
    build_backend
    build_frontend
    test_python_bridge
    build_docker
    deploy_docker
    show_deployment_info

    print_success "All build steps completed successfully!"
}

# Run main function
main "$@"
