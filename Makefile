# ==========================================
# Consciousness Platform Ecosystem Makefile
# Enterprise-Grade Build & Deployment System
# ==========================================

# ==========================================
# Configuration
# ==========================================
.PHONY: help install test build deploy clean docs security monitoring

# Environment Variables
ENV ?= development
VERSION ?= $(shell git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP ?= $(shell date +%Y%m%d_%H%M%S)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ==========================================
# Help System
# ==========================================
help: ## Display this help message
	@echo "$(BLUE)=========================================="
	@echo " Consciousness Platform Ecosystem"
	@echo " Enterprise Build & Deployment System"
	@echo "==========================================$(NC)"
	@echo ""
	@echo "Environment: $(ENV)"
	@echo "Version: $(VERSION)"
	@echo "Commit: $(COMMIT)"
	@echo "Timestamp: $(TIMESTAMP)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ==========================================
# Installation & Setup
# ==========================================
install: ## Install all dependencies for development
	@echo "$(BLUE)🔧 Installing dependencies...$(NC)"
	@echo "$(YELLOW)Python Dependencies:$(NC)"
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "$(YELLOW)Node.js Dependencies:$(NC)"
	cd enterprise/applications/web && npm install
	cd enterprise/applications/mobile && npm install
	@echo "$(YELLOW)Docker Images:$(NC)"
	docker pull postgres:14
	docker pull redis:6-alpine
	@echo "$(GREEN)✅ Dependencies installed successfully$(NC)"

install-production: ## Install dependencies for production
	@echo "$(BLUE)🔧 Installing production dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Production dependencies installed$(NC)"

# ==========================================
# Development Environment
# ==========================================
dev: ## Start development environment
	@echo "$(BLUE)🚀 Starting development environment...$(NC)"
	cd enterprise/deployment && docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✅ Development environment started$(NC)"
	@echo "📊 Dashboard: http://localhost:3000"
	@echo "🔌 API: http://localhost:8000"
	@echo "📈 Monitoring: http://localhost:9090"

dev-stop: ## Stop development environment
	@echo "$(BLUE)🛑 Stopping development environment...$(NC)"
	cd enterprise/deployment && docker-compose -f docker-compose.dev.yml down
	@echo "$(GREEN)✅ Development environment stopped$(NC)"

dev-logs: ## View development logs
	cd enterprise/deployment && docker-compose -f docker-compose.dev.yml logs -f

# ==========================================
# Testing Suite
# ==========================================
test: ## Run complete test suite
	@echo "$(BLUE)🧪 Running complete test suite...$(NC)"
	@echo "$(YELLOW)Unit Tests:$(NC)"
	python -m pytest enterprise/testing/unit/ -v --cov=enterprise --cov-report=html
	@echo "$(YELLOW)Integration Tests:$(NC)"
	python -m pytest enterprise/testing/integration/ -v
	@echo "$(YELLOW)End-to-End Tests:$(NC)"
	python -m pytest enterprise/testing/e2e/ -v
	@echo "$(GREEN)✅ All tests completed$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)🧪 Running unit tests...$(NC)"
	python -m pytest enterprise/testing/unit/ -v --cov=enterprise --cov-report=html
	@echo "$(GREEN)✅ Unit tests completed$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)🧪 Running integration tests...$(NC)"
	python -m pytest enterprise/testing/integration/ -v
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

test-e2e: ## Run end-to-end tests only
	@echo "$(BLUE)🧪 Running end-to-end tests...$(NC)"
	python -m pytest enterprise/testing/e2e/ -v
	@echo "$(GREEN)✅ End-to-end tests completed$(NC)"

test-performance: ## Run performance tests
	@echo "$(BLUE)⚡ Running performance tests...$(NC)"
	python -m pytest enterprise/testing/performance/ -v --benchmark-only
	@echo "$(GREEN)✅ Performance tests completed$(NC)"

test-security: ## Run security tests
	@echo "$(BLUE)🔒 Running security tests...$(NC)"
	bandit -r enterprise/ -f json -o enterprise/testing/security/results.json
	safety check --json > enterprise/testing/security/dependencies.json
	@echo "$(GREEN)✅ Security tests completed$(NC)"

test-coverage: ## Generate test coverage report
	@echo "$(BLUE)📊 Generating test coverage report...$(NC)"
	python -m pytest --cov=enterprise --cov-report=html --cov-report=xml
	open htmlcov/index.html
	@echo "$(GREEN)✅ Coverage report generated$(NC)"

# ==========================================
# Build System
# ==========================================
build: ## Build all components
	@echo "$(BLUE)🔨 Building all components...$(NC)"
	@echo "$(YELLOW)Core Engine:$(NC)"
	cd enterprise/core && python setup.py build_ext --inplace
	@echo "$(YELLOW)Web Application:$(NC)"
	cd enterprise/applications/web && npm run build
	@echo "$(YELLOW)Mobile Applications:$(NC)"
	cd enterprise/applications/mobile && npm run build
	@echo "$(YELLOW)Docker Images:$(NC)"
	docker build -t consciousnessplatform/api:$(VERSION) enterprise/
	docker build -t consciousnessplatform/web:$(VERSION) enterprise/applications/web/
	@echo "$(GREEN)✅ All components built successfully$(NC)"

build-core: ## Build core consciousness engine
	@echo "$(BLUE)🔨 Building core engine...$(NC)"
	cd enterprise/core && python setup.py build_ext --inplace
	@echo "$(GREEN)✅ Core engine built$(NC)"

build-web: ## Build web application
	@echo "$(BLUE)🔨 Building web application...$(NC)"
	cd enterprise/applications/web && npm run build
	@echo "$(GREEN)✅ Web application built$(NC)"

build-mobile: ## Build mobile applications
	@echo "$(BLUE)🔨 Building mobile applications...$(NC)"
	cd enterprise/applications/mobile && npm run build
	@echo "$(GREEN)✅ Mobile applications built$(NC)"

build-docker: ## Build Docker images
	@echo "$(BLUE)🐳 Building Docker images...$(NC)"
	docker build -t consciousnessplatform/api:$(VERSION) enterprise/
	docker build -t consciousnessplatform/web:$(VERSION) enterprise/applications/web/
	docker build -t consciousnessplatform/worker:$(VERSION) enterprise/services/
	@echo "$(GREEN)✅ Docker images built$(NC)"

build-docs: ## Build documentation
	@echo "$(BLUE)📚 Building documentation...$(NC)"
	cd enterprise/documentation && make html
	@echo "$(GREEN)✅ Documentation built$(NC)"

# ==========================================
# Deployment System
# ==========================================
deploy: ## Deploy to production environment
	@echo "$(BLUE)🚀 Deploying to production...$(NC)"
	@echo "$(RED)⚠️  This will deploy to production environment$(NC)"
	@echo "$(YELLOW)Environment: $(ENV)$(NC)"
	@echo "$(YELLOW)Version: $(VERSION)$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || (echo "Deployment cancelled" && exit 1)
	@echo "$(BLUE)Starting deployment...$(NC)"
	ansible-playbook enterprise/deployment/ansible/deploy.yml -i inventory/production
	kubectl apply -f enterprise/deployment/kubernetes/
	@echo "$(GREEN)✅ Deployment completed$(NC)"

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)🚀 Deploying to staging...$(NC)"
	ansible-playbook enterprise/deployment/ansible/deploy.yml -i inventory/staging
	kubectl apply -f enterprise/deployment/kubernetes/ --context=staging
	@echo "$(GREEN)✅ Staging deployment completed$(NC)"

deploy-local: ## Deploy to local environment
	@echo "$(BLUE)🚀 Deploying locally...$(NC)"
	cd enterprise/deployment && docker-compose -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✅ Local deployment completed$(NC)"

rollback: ## Rollback to previous version
	@echo "$(BLUE)⏪ Rolling back deployment...$(NC)"
	kubectl rollout undo deployment/consciousness-api
	kubectl rollout undo deployment/consciousness-web
	@echo "$(GREEN)✅ Rollback completed$(NC)"

# ==========================================
# Enterprise Operations
# ==========================================
security: ## Run security audit and compliance checks
	@echo "$(BLUE)🔒 Running security audit...$(NC)"
	@echo "$(YELLOW)Dependency Scanning:$(NC)"
	safety check
	@echo "$(YELLOW)Code Security:$(NC)"
	bandit -r enterprise/
	@echo "$(YELLOW)Container Security:$(NC)"
	trivy image consciousnessplatform/api:$(VERSION)
	@echo "$(YELLOW)Compliance Check:$(NC)"
	python enterprise/security/compliance/audit.py
	@echo "$(GREEN)✅ Security audit completed$(NC)"

monitoring: ## Start monitoring stack
	@echo "$(BLUE)📊 Starting monitoring stack...$(NC)"
	cd enterprise/monitoring && docker-compose up -d
	@echo "$(GREEN)✅ Monitoring stack started$(NC)"
	@echo "📈 Grafana: http://localhost:3001"
	@echo "📊 Prometheus: http://localhost:9090"
	@echo "🚨 Alert Manager: http://localhost:9093"

monitoring-stop: ## Stop monitoring stack
	@echo "$(BLUE)🛑 Stopping monitoring stack...$(NC)"
	cd enterprise/monitoring && docker-compose down
	@echo "$(GREEN)✅ Monitoring stack stopped$(NC)"

backup: ## Create enterprise backup
	@echo "$(BLUE)💾 Creating enterprise backup...$(NC)"
	@echo "$(YELLOW)Database:$(NC)"
	pg_dump consciousness_db > enterprise/infrastructure/database/backups/backup_$(TIMESTAMP).sql
	@echo "$(YELLOW)Configuration:$(NC)"
	tar -czf enterprise/infrastructure/backups/config_$(TIMESTAMP).tar.gz enterprise/
	@echo "$(YELLOW)Assets:$(NC)"
	tar -czf enterprise/assets/backups/assets_$(TIMESTAMP).tar.gz enterprise/assets/
	@echo "$(GREEN)✅ Backup completed$(NC)"

restore: ## Restore from backup
	@echo "$(BLUE)🔄 Restoring from backup...$(NC)"
	@echo "$(RED)⚠️  This will overwrite current data$(NC)"
	@read -p "Backup timestamp: " timestamp && \
	read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || (echo "Restore cancelled" && exit 1)
	psql consciousness_db < enterprise/infrastructure/database/backups/backup_$$timestamp.sql
	tar -xzf enterprise/infrastructure/backups/config_$$timestamp.tar.gz
	tar -xzf enterprise/assets/backups/assets_$$timestamp.tar.gz
	@echo "$(GREEN)✅ Restore completed$(NC)"

# ==========================================
# Quality Assurance
# ==========================================
lint: ## Run code linting
	@echo "$(BLUE)🔍 Running code linting...$(NC)"
	flake8 enterprise/ --config enterprise/.flake8
	black --check enterprise/
	isort --check-only enterprise/
	@echo "$(GREEN)✅ Code linting completed$(NC)"

format: ## Format code
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	black enterprise/
	isort enterprise/
	@echo "$(GREEN)✅ Code formatting completed$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)🔍 Running type checking...$(NC)"
	mypy enterprise/ --config-file enterprise/mypy.ini
	@echo "$(GREEN)✅ Type checking completed$(NC)"

quality: lint type-check test-security ## Run complete quality assurance suite
	@echo "$(GREEN)✅ Quality assurance completed$(NC)"

# ==========================================
# Documentation
# ==========================================
docs: ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	cd enterprise/documentation && make html
	sphinx-build -b pdf enterprise/documentation enterprise/documentation/_build/pdf
	@echo "$(GREEN)✅ Documentation generated$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)📚 Serving documentation...$(NC)"
	cd enterprise/documentation/_build/html && python -m http.server 8001
	@echo "📖 Documentation available at: http://localhost:8001"

# ==========================================
# Database Operations
# ==========================================
db-migrate: ## Run database migrations
	@echo "$(BLUE)🗄️ Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✅ Database migrations completed$(NC)"

db-seed: ## Seed database with test data
	@echo "$(BLUE)🌱 Seeding database...$(NC)"
	python enterprise/scripts/db_seed.py
	@echo "$(GREEN)✅ Database seeded$(NC)"

db-reset: ## Reset database (WARNING: Destroys all data)
	@echo "$(BLUE)💥 Resetting database...$(NC)"
	@echo "$(RED)⚠️  This will destroy all data$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || (echo "Database reset cancelled" && exit 1)
	alembic downgrade base
	alembic upgrade head
	python enterprise/scripts/db_seed.py
	@echo "$(GREEN)✅ Database reset completed$(NC)"

# ==========================================
# Platform Operations
# ==========================================
platform-list: ## List all platforms
	@echo "$(BLUE)📋 Available Platforms:$(NC)"
	@find enterprise/platforms -name "*.py" -exec basename {} \; | sed 's/.py//' | sort
	@echo "$(GREEN)✅ Platform list displayed$(NC)"

platform-test: ## Test specific platform
	@echo "$(BLUE)🧪 Testing platform...$(NC)"
	@echo "Usage: make platform-test PLATFORM=physics"
	@if [ -z "$(PLATFORM)" ]; then \
		echo "$(RED)Error: Please specify PLATFORM$(NC)"; \
		echo "Example: make platform-test PLATFORM=physics"; \
		exit 1; \
	fi
	python -m pytest enterprise/platforms/scientific/$(PLATFORM)/tests/ -v
	@echo "$(GREEN)✅ Platform $(PLATFORM) tested$(NC)"

platform-run: ## Run specific platform
	@echo "$(BLUE)🚀 Running platform...$(NC)"
	@echo "Usage: make platform-run PLATFORM=physics"
	@if [ -z "$(PLATFORM)" ]; then \
		echo "$(RED)Error: Please specify PLATFORM$(NC)"; \
		echo "Example: make platform-run PLATFORM=physics"; \
		exit 1; \
	fi
	python enterprise/platforms/scientific/$(PLATFORM)/consciousness_$(PLATFORM)_platform.py
	@echo "$(GREEN)✅ Platform $(PLATFORM) executed$(NC)"

# ==========================================
# Utility Commands
# ==========================================
clean: ## Clean build artifacts and temporary files
	@echo "$(BLUE)🧹 Cleaning build artifacts...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	docker system prune -f
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-all: clean ## Clean everything including Docker images
	@echo "$(BLUE)🧹 Deep cleaning...$(NC)"
	docker rmi $(docker images -q) 2>/dev/null || true
	docker system prune -a -f
	rm -rf enterprise/applications/web/node_modules/
	rm -rf enterprise/applications/mobile/node_modules/
	@echo "$(GREEN)✅ Deep cleanup completed$(NC)"

status: ## Show system status
	@echo "$(BLUE)📊 System Status$(NC)"
	@echo "$(YELLOW)Environment:$(NC) $(ENV)"
	@echo "$(YELLOW)Version:$(NC) $(VERSION)"
	@echo "$(YELLOW)Commit:$(NC) $(COMMIT)"
	@echo "$(YELLOW)Docker:$(NC)"
	@docker --version
	@echo "$(YELLOW)Docker Compose:$(NC)"
	@docker-compose --version
	@echo "$(YELLOW)Python:$(NC)"
	@python --version
	@echo "$(YELLOW)Node.js:$(NC)"
	@node --version 2>/dev/null || echo "Not installed"
	@echo "$(YELLOW)Git:$(NC)"
	@git --version
	@echo "$(GREEN)✅ Status check completed$(NC)"

version: ## Show version information
	@echo "$(BLUE)🔖 Version Information$(NC)"
	@echo "Version: $(VERSION)"
	@echo "Commit: $(COMMIT)"
	@echo "Build Date: $(TIMESTAMP)"
	@echo "Environment: $(ENV)"

# ==========================================
# Emergency Commands
# ==========================================
emergency-stop: ## Emergency stop all services
	@echo "$(RED)🚨 EMERGENCY STOP$(NC)"
	docker-compose -f enterprise/deployment/docker-compose.*.yml down 2>/dev/null || true
	docker stop $(docker ps -q) 2>/dev/null || true
	kubectl delete pods --all --force --grace-period=0 2>/dev/null || true
	@echo "$(GREEN)✅ Emergency stop completed$(NC)"

emergency-restart: ## Emergency restart all services
	@echo "$(RED)🚨 EMERGENCY RESTART$(NC)"
	make emergency-stop
	sleep 5
	make deploy-local
	@echo "$(GREEN)✅ Emergency restart completed$(NC)"

# ==========================================
# Development Workflow
# ==========================================
workflow-setup: ## Setup development workflow
	@echo "$(BLUE)🔧 Setting up development workflow...$(NC)"
	pre-commit install
	pre-commit run --all-files
	@echo "$(GREEN)✅ Development workflow setup completed$(NC)"

workflow-check: ## Check development workflow compliance
	@echo "$(BLUE)🔍 Checking development workflow...$(NC)"
	pre-commit run --all-files
	make quality
	@echo "$(GREEN)✅ Development workflow check completed$(NC)"

# ==========================================
# CI/CD Integration
# ==========================================
ci-build: ## CI build process
	@echo "$(BLUE)🔨 CI Build Process...$(NC)"
	make install-production
	make lint
	make test-unit
	make build
	make test-integration
	@echo "$(GREEN)✅ CI build completed$(NC)"

ci-deploy: ## CI deployment process
	@echo "$(BLUE)🚀 CI Deployment Process...$(NC)"
	make test-e2e
	make security
	make deploy-staging
	make test-performance
	make deploy
	@echo "$(GREEN)✅ CI deployment completed$(NC)"

# ==========================================
# Enterprise Standards
# ==========================================
audit: ## Run enterprise audit
	@echo "$(BLUE)📋 Running enterprise audit...$(NC)"
	python enterprise/testing/enterprise_audit_system.py
	@echo "$(GREEN)✅ Enterprise audit completed$(NC)"

compliance: ## Check compliance status
	@echo "$(BLUE)⚖️ Checking compliance status...$(NC)"
	python enterprise/security/compliance/audit.py
	@echo "$(GREEN)✅ Compliance check completed$(NC)"

# ==========================================
# Default Target
# ==========================================
.DEFAULT_GOAL := help

# ==========================================
# Enterprise Makefile Footer
# ==========================================
# This Makefile follows Jeff Enterprise Standards
# Version: 1.0.0
# Last Updated: September 10, 2025
# Compliance: SOC2, Enterprise Security Standards
