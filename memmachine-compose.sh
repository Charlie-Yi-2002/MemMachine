#!/usr/bin/env bash

# MemMachine Docker Startup Script
# This script helps you get MemMachine running with Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
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

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        if [ -f "sample_configs/env.dockercompose" ]; then
            cp sample_configs/env.dockercompose .env
            print_success "Created .env file from sample_configs/env.dockercompose"
            print_warning "Please edit .env file with your configuration before continuing"
            print_warning "Especially set your OPENAI_API_KEY"
            print_info "Exiting script. Please edit .env file and re-run the script."
            exit 0
        else
            print_error "sample_configs/env.dockercompose file not found. Please create .env file manually."
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# Check if configuration.yml file exists
check_config_file() {
    if [ ! -f "configuration.yml" ]; then
        print_warning "configuration.yml file not found. Creating from template..."
        if [ -f "sample_configs/episodic_memory_config.sample" ]; then
            cp sample_configs/episodic_memory_config.sample configuration.yml
            print_success "Created configuration.yml file from sample_configs/episodic_memory_config.sample"
            print_warning "Please edit configuration.yml file with your configuration before continuing"
            print_warning "Especially set your API keys and database credentials"
            print_info "Exiting script. Please edit configuration.yml file and re-run the script."
            exit 0
        else
            print_error "sample_configs/episodic_memory_config.sample file not found. Please create configuration.yml file manually."
            exit 1
        fi
    else
        print_success "configuration.yml file found"
    fi
}

# Check if required environment variables are set
check_required_env() {
    if [ -f ".env" ]; then
        source .env
        
        if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
            print_warning "OPENAI_API_KEY is not set or is using placeholder value"
            print_warning "Please set your OpenAI API key in the .env file"
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "OPENAI_API_KEY is configured"
        fi
    fi
}

# Check if configuration.yml has required fields
check_required_config() {
    if [ -f "configuration.yml" ]; then
        # Check for API key in configuration.yml - look for actual placeholder patterns
        if grep -q "api_key.*your_.*_api_key_here" configuration.yml || grep -q "api_key.*sk-example" configuration.yml || grep -q "api_key.*sk-test" configuration.yml; then
            print_warning "API key in configuration.yml appears to be a placeholder or example value"
            print_warning "Please set your actual API key in the configuration.yml file"
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "API key in configuration.yml appears to be configured"
        fi
        
        # Check for database credentials - look for generic placeholder passwords
        if grep -q "password.*password" configuration.yml && ! grep -q "password.*memmachine_password" configuration.yml; then
            print_warning "Database password in configuration.yml appears to be a placeholder"
            print_warning "Please set your actual database password in the configuration.yml file"
            read -p "Press Enter to continue anyway (some features may not work)..."
        else
            print_success "Database credentials in configuration.yml appear to be configured"
        fi
    fi
}

# Pull and start services
start_services() {
    print_info "Pulling and starting MemMachine services..."
    
    # Use docker-compose or docker compose based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Pull and start services (no build needed since we use pre-built image)
    $COMPOSE_CMD up -d
    
    print_success "Services started successfully!"
}

# Wait for services to be healthy
wait_for_health() {
    print_info "Waiting for services to be healthy..."
    
    # Use docker-compose or docker compose based on what's available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Wait for services to be healthy
    $COMPOSE_CMD ps
    
    print_info "Checking service health..."
    
    # Wait for PostgreSQL
    print_info "Waiting for PostgreSQL to be ready..."
    if timeout 120 bash -c 'until docker exec memmachine-postgres pg_isready -U memmachine -d memmachine; do sleep 2; done'; then
        print_success "PostgreSQL is ready"
    else
        print_error "PostgreSQL failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
    
    # Wait for Neo4j
    print_info "Waiting for Neo4j to be ready..."
    if timeout 120 bash -c 'until docker exec memmachine-neo4j cypher-shell -u neo4j -p neo4j_password "RETURN 1" > /dev/null 2>&1; do sleep 2; done'; then
        print_success "Neo4j is ready"
    else
        print_error "Neo4j failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
    
    # Wait for MemMachine
    print_info "Waiting for MemMachine to be ready..."
    if timeout 120 bash -c 'until curl -f http://localhost:8080/health > /dev/null 2>&1; do sleep 5; done'; then
        print_success "MemMachine is ready"
    else
        print_error "MemMachine failed to become ready in 120 seconds. Check container logs and configuration."
        exit 1
    fi
}

# Show service information
show_service_info() {
    print_success "🎉 MemMachine is now running!"
    echo ""
    echo "Service URLs:"
    echo "  📊 MemMachine API: http://localhost:8080"
    echo "  🗄️  Neo4j Browser: http://localhost:7474"
    echo "  📈 Health Check: http://localhost:8080/health"
    echo "  📊 Metrics: http://localhost:8080/metrics"
    echo ""
    echo "Database Access:"
    echo "  🐘 PostgreSQL: localhost:5432 (user: memmachine, db: memmachine)"
    echo "  🔗 Neo4j Bolt: localhost:7687 (user: neo4j)"
    echo ""
    echo "Useful Commands:"
    echo "  📋 View logs: docker-compose logs -f"
    echo "  🛑 Stop services: docker-compose down"
    echo "  🔄 Restart: docker-compose restart"
    echo "  🧹 Clean up: docker-compose down -v"
    echo ""
}

# Main execution
main() {
    echo "MemMachine Docker Startup Script"
    echo "===================================="
    echo ""
    
    check_docker
    check_env_file
    check_config_file
    check_required_env
    check_required_config
    start_services
    wait_for_health
    show_service_info
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_info "Stopping MemMachine services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose down
        else
            docker compose down
        fi
        print_success "Services stopped"
        ;;
    "restart")
        print_info "Restarting MemMachine services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose restart
        else
            docker compose restart
        fi
        print_success "Services restarted"
        ;;
    "logs")
        print_info "Showing MemMachine logs..."
        if command -v docker-compose &> /dev/null; then
            docker-compose logs -f
        else
            docker compose logs -f
        fi
        ;;
    "clean")
        print_warning "This will remove all data and volumes!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Cleaning up MemMachine services and data..."
            if command -v docker-compose &> /dev/null; then
                docker-compose down -v
            else
                docker compose down -v
            fi
            print_success "Cleanup completed"
        else
            print_info "Cleanup cancelled"
        fi
        ;;
    "help"|"-h"|"--help")
        echo "MemMachine Docker Startup Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  Start MemMachine services"
        echo "  stop       Stop MemMachine services"
        echo "  restart    Restart MemMachine services"
        echo "  logs       Show service logs"
        echo "  clean      Remove all services and data"
        echo "  help       Show this help message"
        echo ""
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
