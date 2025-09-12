#!/bin/bash

# Café Mapper - Development Docker Script
set -e

echo "🚀 Café Mapper - Development Environment"
echo "======================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if ports are available
check_ports() {
    local ports=("5173" "8000")
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "⚠️  Port $port is already in use. Please free it first."
            echo "   You can check what's using it with: lsof -i :$port"
            exit 1
        fi
    done
}

# Function to start development environment
start_dev() {
    echo "🏗️  Building development containers..."
    docker-compose -f docker-compose.dev.yml build
    
    echo "🚀 Starting development environment..."
    docker-compose -f docker-compose.dev.yml up -d
    
    echo ""
    echo "✅ Development environment is starting up!"
    echo ""
    echo "📊 Services:"
    echo "   Frontend (Vite): http://localhost:5173"
    echo "   Backend (FastAPI): http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo ""
    echo "📝 Logs:"
    echo "   View all logs: docker-compose -f docker-compose.dev.yml logs -f"
    echo "   Backend logs: docker-compose -f docker-compose.dev.yml logs -f api"
    echo "   Frontend logs: docker-compose -f docker-compose.dev.yml logs -f web"
    echo ""
    echo "🔧 Management:"
    echo "   Stop: ./scripts/docker-dev.sh stop"
    echo "   Restart: ./scripts/docker-dev.sh restart"
    echo "   Logs: ./scripts/docker-dev.sh logs"
    echo ""
}

# Function to stop development environment
stop_dev() {
    echo "🛑 Stopping development environment..."
    docker-compose -f docker-compose.dev.yml down
    echo "✅ Development environment stopped."
}

# Function to restart development environment
restart_dev() {
    echo "🔄 Restarting development environment..."
    docker-compose -f docker-compose.dev.yml restart
    echo "✅ Development environment restarted."
}

# Function to show logs
show_logs() {
    echo "📝 Showing development logs (Ctrl+C to exit)..."
    docker-compose -f docker-compose.dev.yml logs -f
}

# Function to show status
show_status() {
    echo "📊 Development Environment Status:"
    echo "================================="
    docker-compose -f docker-compose.dev.yml ps
    echo ""
    echo "💾 Volume Usage:"
    docker system df
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up development environment..."
    docker-compose -f docker-compose.dev.yml down --volumes --remove-orphans
    echo "✅ Development environment cleaned up."
}

# Function to rebuild
rebuild() {
    echo "🔨 Rebuilding development environment..."
    docker-compose -f docker-compose.dev.yml down
    docker-compose -f docker-compose.dev.yml build --no-cache
    docker-compose -f docker-compose.dev.yml up -d
    echo "✅ Development environment rebuilt and started."
}

# Main script logic
case "${1:-start}" in
    start)
        check_docker
        check_ports
        start_dev
        ;;
    stop)
        check_docker
        stop_dev
        ;;
    restart)
        check_docker
        restart_dev
        ;;
    logs)
        check_docker
        show_logs
        ;;
    status)
        check_docker
        show_status
        ;;
    clean)
        check_docker
        cleanup
        ;;
    rebuild)
        check_docker
        rebuild
        ;;
    *)
        echo "🚀 Café Mapper Development Environment"
        echo ""
        echo "Usage: $0 {start|stop|restart|logs|status|clean|rebuild}"
        echo ""
        echo "Commands:"
        echo "  start   - Start development environment (default)"
        echo "  stop    - Stop development environment"
        echo "  restart - Restart development environment"
        echo "  logs    - Show live logs"
        echo "  status  - Show container status"
        echo "  clean   - Stop and remove all containers and volumes"
        echo "  rebuild - Rebuild containers from scratch"
        echo ""
        exit 1
        ;;
esac