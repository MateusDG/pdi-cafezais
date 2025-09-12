#!/bin/bash

# Café Mapper - Production Docker Script
set -e

echo "🏭 Café Mapper - Production Environment"
echo "======================================"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to check if port 80 is available
check_port() {
    if lsof -Pi :80 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port 80 is already in use. Please free it first."
        echo "   You can check what's using it with: sudo lsof -i :80"
        exit 1
    fi
}

# Function to start production environment
start_prod() {
    echo "🏗️  Building production containers..."
    docker-compose build
    
    echo "🚀 Starting production environment..."
    docker-compose up -d
    
    echo ""
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
    
    # Check health status
    if docker-compose ps | grep -q "unhealthy"; then
        echo "⚠️  Some services are not healthy. Checking logs..."
        docker-compose logs --tail=20
    else
        echo "✅ Production environment is running!"
        echo ""
        echo "🌐 Access:"
        echo "   Website: http://localhost"
        echo "   API: http://localhost:8000"
        echo "   API Docs: http://localhost:8000/docs"
        echo ""
        echo "📊 Status: docker-compose ps"
        echo "📝 Logs: docker-compose logs -f"
        echo "🛑 Stop: ./scripts/docker-prod.sh stop"
    fi
}

# Function to stop production environment
stop_prod() {
    echo "🛑 Stopping production environment..."
    docker-compose down
    echo "✅ Production environment stopped."
}

# Function to restart production environment
restart_prod() {
    echo "🔄 Restarting production environment..."
    docker-compose restart
    echo "✅ Production environment restarted."
}

# Function to show logs
show_logs() {
    echo "📝 Showing production logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Function to show status
show_status() {
    echo "📊 Production Environment Status:"
    echo "================================"
    docker-compose ps
    echo ""
    echo "🏥 Health Checks:"
    echo "Frontend: $(curl -s http://localhost/health 2>/dev/null || echo 'Not responding')"
    echo "Backend: $(curl -s http://localhost:8000/api/health 2>/dev/null || echo 'Not responding')"
    echo ""
    echo "💾 Volume Usage:"
    docker system df
    echo ""
    echo "🖥️  Resource Usage:"
    docker stats --no-stream cafe-mapper-web cafe-mapper-api 2>/dev/null || echo "Containers not running"
}

# Function to backup data
backup() {
    echo "💾 Creating backup of production data..."
    BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup volumes
    docker run --rm -v cafe-data:/data -v "$PWD/$BACKUP_DIR":/backup alpine tar czf /backup/cafe-data.tar.gz -C /data .
    docker run --rm -v cafe-results:/data -v "$PWD/$BACKUP_DIR":/backup alpine tar czf /backup/cafe-results.tar.gz -C /data .
    
    echo "✅ Backup created at: $BACKUP_DIR"
    echo "   - cafe-data.tar.gz"
    echo "   - cafe-results.tar.gz"
}

# Function to restore data
restore() {
    if [ -z "$2" ]; then
        echo "❌ Please specify backup directory to restore from"
        echo "   Usage: $0 restore /path/to/backup"
        exit 1
    fi
    
    BACKUP_DIR="$2"
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "❌ Backup directory not found: $BACKUP_DIR"
        exit 1
    fi
    
    echo "🔄 Restoring from backup: $BACKUP_DIR"
    echo "⚠️  This will overwrite current data. Continue? (y/N)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "❌ Restore cancelled."
        exit 1
    fi
    
    # Stop services
    docker-compose down
    
    # Restore volumes
    if [ -f "$BACKUP_DIR/cafe-data.tar.gz" ]; then
        docker run --rm -v cafe-data:/data -v "$BACKUP_DIR":/backup alpine tar xzf /backup/cafe-data.tar.gz -C /data
        echo "✅ cafe-data restored"
    fi
    
    if [ -f "$BACKUP_DIR/cafe-results.tar.gz" ]; then
        docker run --rm -v cafe-results:/data -v "$BACKUP_DIR":/backup alpine tar xzf /backup/cafe-results.tar.gz -C /data
        echo "✅ cafe-results restored"
    fi
    
    # Restart services
    docker-compose up -d
    echo "✅ Restore completed and services restarted"
}

# Function to update production
update() {
    echo "🔄 Updating production environment..."
    echo "This will rebuild containers with latest code."
    echo "Continue? (y/N)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "❌ Update cancelled."
        exit 1
    fi
    
    # Create backup first
    backup
    
    # Pull latest code (if using git)
    if [ -d ".git" ]; then
        echo "📥 Pulling latest code..."
        git pull
    fi
    
    # Rebuild and restart
    echo "🔨 Rebuilding containers..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    
    echo "✅ Production environment updated!"
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up production environment..."
    echo "⚠️  This will remove all data and containers. Continue? (y/N)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "❌ Cleanup cancelled."
        exit 1
    fi
    
    docker-compose down --volumes --remove-orphans --rmi local
    docker system prune -f
    echo "✅ Production environment cleaned up."
}

# Main script logic
case "${1:-start}" in
    start)
        check_docker
        check_port
        start_prod
        ;;
    stop)
        check_docker
        stop_prod
        ;;
    restart)
        check_docker
        restart_prod
        ;;
    logs)
        check_docker
        show_logs
        ;;
    status)
        check_docker
        show_status
        ;;
    backup)
        check_docker
        backup
        ;;
    restore)
        check_docker
        restore "$@"
        ;;
    update)
        check_docker
        update
        ;;
    clean)
        check_docker
        cleanup
        ;;
    *)
        echo "🏭 Café Mapper Production Environment"
        echo ""
        echo "Usage: $0 {start|stop|restart|logs|status|backup|restore|update|clean}"
        echo ""
        echo "Commands:"
        echo "  start          - Start production environment (default)"
        echo "  stop           - Stop production environment"
        echo "  restart        - Restart production environment"
        echo "  logs           - Show live logs"
        echo "  status         - Show detailed status and health"
        echo "  backup         - Create backup of data volumes"
        echo "  restore <dir>  - Restore from backup directory"
        echo "  update         - Update to latest code and rebuild"
        echo "  clean          - Remove everything (DESTRUCTIVE)"
        echo ""
        exit 1
        ;;
esac