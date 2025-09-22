#!/bin/bash

# Enterprise Consciousness Platform - Complete System Startup
# This script starts both the backend API server and frontend development server

set -e  # Exit on any error

# ================================
# Configuration
# ================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT=8000
FRONTEND_PORT=3000
BACKEND_PID=""
FRONTEND_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ================================
# Functions
# ================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

cleanup() {
    log_warning "Cleaning up processes..."

    # Kill backend if running
    if [ ! -z "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        log_info "Stopping backend server (PID: $BACKEND_PID)"
        kill $BACKEND_PID 2>/dev/null || true
    fi

    # Kill frontend if running
    if [ ! -z "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        log_info "Stopping frontend server (PID: $FRONTEND_PID)"
        kill $FRONTEND_PID 2>/dev/null || true
    fi

    log_info "Cleanup complete"
    exit 0
}

check_dependencies() {
    log_step "Checking system dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    # Check Node.js (optional for frontend)
    if ! command -v node &> /dev/null; then
        log_warning "Node.js not found - frontend will not be available"
        HAS_NODE=false
    else
        HAS_NODE=true
    fi

    # Check npm (if Node.js is available)
    if [ "$HAS_NODE" = true ] && ! command -v npm &> /dev/null; then
        log_warning "npm not found - frontend will not be available"
        HAS_NODE=false
    fi

    log_success "Dependencies check complete"
}

start_backend() {
    log_step "Starting Enterprise Consciousness Platform API Server..."

    # Check if port is available
    if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null; then
        log_error "Port $API_PORT is already in use"
        exit 1
    fi

    # Start backend server in background
    cd "$SCRIPT_DIR"
    python3 api_server.py &
    BACKEND_PID=$!

    # Wait for server to start
    log_info "Waiting for backend server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            log_success "Backend server started successfully (PID: $BACKEND_PID)"
            log_info "API Server: http://localhost:$API_PORT"
            log_info "API Docs: http://localhost:$API_PORT/docs"
            log_info "Health Check: http://localhost:$API_PORT/health"
            return 0
        fi
        sleep 1
    done

    log_error "Backend server failed to start within 30 seconds"
    cleanup
    exit 1
}

start_frontend() {
    if [ "$HAS_NODE" = false ]; then
        log_warning "Skipping frontend - Node.js/npm not available"
        return 0
    fi

    log_step "Starting React Frontend Development Server..."

    # Check if frontend directory exists
    if [ ! -d "$SCRIPT_DIR/frontend" ]; then
        log_warning "Frontend directory not found - skipping frontend startup"
        return 0
    fi

    # Check if port is available
    if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null; then
        log_error "Port $FRONTEND_PORT is already in use"
        exit 1
    fi

    # Navigate to frontend directory
    cd "$SCRIPT_DIR/frontend"

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_info "Installing frontend dependencies..."
        npm install
        if [ $? -ne 0 ]; then
            log_error "Failed to install frontend dependencies"
            return 1
        fi
    fi

    # Start frontend server in background
    npm start &
    FRONTEND_PID=$!

    # Wait for server to start
    log_info "Waiting for frontend server to start..."
    for i in {1..60}; do
        if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
            log_success "Frontend server started successfully (PID: $FRONTEND_PID)"
            log_info "Frontend: http://localhost:$FRONTEND_PORT"
            return 0
        fi
        sleep 1
    done

    log_error "Frontend server failed to start within 60 seconds"
    cleanup
    exit 1
}

test_system_integration() {
    log_step "Testing system integration..."

    # Test backend API
    log_info "Testing backend API endpoints..."
    if curl -s http://localhost:$API_PORT/health | grep -q "healthy"; then
        log_success "Backend health check: PASSED"
    else
        log_error "Backend health check: FAILED"
        return 1
    fi

    # Test consciousness endpoint
    if curl -s http://localhost:$API_PORT/consciousness | grep -q "field_data"; then
        log_success "Consciousness API: PASSED"
    else
        log_error "Consciousness API: FAILED"
        return 1
    fi

    # Test metrics endpoint
    if curl -s http://localhost:$API_PORT/metrics | grep -q "timestamp"; then
        log_success "Metrics API: PASSED"
    else
        log_error "Metrics API: FAILED"
        return 1
    fi

    # Test frontend (if available)
    if [ "$HAS_NODE" = true ] && [ -d "$SCRIPT_DIR/frontend" ]; then
        log_info "Testing frontend accessibility..."
        if curl -s http://localhost:$FRONTEND_PORT | grep -q "react"; then
            log_success "Frontend accessibility: PASSED"
        else
            log_warning "Frontend accessibility: Could not verify (may still be loading)"
        fi
    fi

    log_success "System integration tests completed successfully!"
    return 0
}

display_access_info() {
    log_header "🚀 ENTERPRISE CONSCIOUSNESS PLATFORM - SYSTEM STARTED"

    echo
    log_success "All systems operational!"
    echo

    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${GREEN}📡 Backend API Server${NC}"
        echo "   URL: http://localhost:$API_PORT"
        echo "   Docs: http://localhost:$API_PORT/docs"
        echo "   Health: http://localhost:$API_PORT/health"
        echo "   Status: http://localhost:$API_PORT/status"
        echo "   PID: $BACKEND_PID"
        echo
    fi

    if [ ! -z "$FRONTEND_PID" ] && [ "$HAS_NODE" = true ]; then
        echo -e "${GREEN}🌐 Frontend Dashboard${NC}"
        echo "   URL: http://localhost:$FRONTEND_PORT"
        echo "   PID: $FRONTEND_PID"
        echo
    fi

    echo -e "${YELLOW}Useful Commands:${NC}"
    echo "   View backend logs: kill -USR1 $BACKEND_PID"
    echo "   Stop system: Ctrl+C or ./stop_system.sh"
    echo "   Test API: curl http://localhost:$API_PORT/health"
    echo

    echo -e "${CYAN}Ready to explore consciousness computing! 🧠✨${NC}"
    echo
}

monitor_system() {
    log_info "System is running. Press Ctrl+C to stop..."

    # Set up signal handler for cleanup
    trap cleanup SIGINT SIGTERM

    # Monitor loop
    while true; do
        # Check if processes are still running
        if [ ! -z "$BACKEND_PID" ] && ! kill -0 $BACKEND_PID 2>/dev/null; then
            log_error "Backend server process died unexpectedly"
            cleanup
            exit 1
        fi

        if [ ! -z "$FRONTEND_PID" ] && ! kill -0 $FRONTEND_PID 2>/dev/null; then
            log_error "Frontend server process died unexpectedly"
            cleanup
            exit 1
        fi

        sleep 5
    done
}

# ================================
# Main Execution
# ================================

main() {
    log_header "🚀 ENTERPRISE CONSCIOUSNESS PLATFORM - SYSTEM STARTUP"

    # Parse command line arguments
    TEST_ONLY=false
    SKIP_FRONTEND=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --test-only)
                TEST_ONLY=true
                shift
                ;;
            --skip-frontend)
                SKIP_FRONTEND=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Start the complete Enterprise Consciousness Platform"
                echo ""
                echo "Options:"
                echo "  --test-only      Run tests without starting servers"
                echo "  --skip-frontend  Skip frontend startup (backend only)"
                echo "  --help, -h       Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                     # Start complete system"
                echo "  $0 --skip-frontend    # Start backend only"
                echo "  $0 --test-only        # Run tests only"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Check dependencies
    check_dependencies

    # Handle test-only mode
    if [ "$TEST_ONLY" = true ]; then
        log_info "Running tests only..."
        test_system_integration
        exit $?
    fi

    # Start backend
    start_backend

    # Start frontend (unless skipped)
    if [ "$SKIP_FRONTEND" = false ]; then
        start_frontend
    fi

    # Test system integration
    if test_system_integration; then
        display_access_info
        monitor_system
    else
        log_error "System integration tests failed"
        cleanup
        exit 1
    fi
}

# Run main function
main "$@"
