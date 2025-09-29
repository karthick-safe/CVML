#!/bin/bash

# CVML Cardio Health Check Kit Analyzer - Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up CVML Cardio Health Check Kit Analyzer..."

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

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is required but not installed"
        exit 1
    fi
}

# Check if Node.js is installed
check_node() {
    print_status "Checking Node.js installation..."
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION found"
    else
        print_error "Node.js is required but not installed"
        exit 1
    fi
}

# Setup Python backend
setup_backend() {
    print_status "Setting up Python backend..."
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Create necessary directories
    mkdir -p models
    mkdir -p data/kit_detection
    mkdir -p data/classification
    
    print_success "Backend setup completed"
    cd ..
}

# Setup Node.js frontend
setup_frontend() {
    print_status "Setting up Node.js frontend..."
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Frontend setup completed"
    cd ..
}

# Create environment files
create_env_files() {
    print_status "Creating environment files..."
    
    # Backend .env
    cat > backend/.env << EOF
# CVML Backend Environment Variables
PYTHONPATH=/app
MODEL_PATH=/app/models
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
EOF

    # Frontend .env.local
    cat > frontend/.env.local << EOF
# CVML Frontend Environment Variables
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=CVML Cardio Health Check Kit Analyzer
EOF

    print_success "Environment files created"
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting CVML Backend..."
cd backend
source venv/bin/activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
EOF

    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting CVML Frontend..."
cd frontend
npm run dev
EOF

    # Full startup script
    cat > start_all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting CVML Application..."

# Start backend in background
echo "Starting backend..."
cd backend
source venv/bin/activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

# Wait for user interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
EOF

    # Make scripts executable
    chmod +x start_backend.sh start_frontend.sh start_all.sh
    
    print_success "Startup scripts created"
}

# Main setup function
main() {
    print_status "Starting CVML setup process..."
    
    # Check prerequisites
    check_python
    check_node
    
    # Setup components
    setup_backend
    setup_frontend
    
    # Create configuration files
    create_env_files
    create_startup_scripts
    
    print_success "🎉 CVML setup completed successfully!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Add training data to the 'data/' directory"
    print_status "2. Train models using the scripts in 'backend/models/'"
    print_status "3. Start the application with './start_all.sh'"
    print_status ""
    print_status "Available commands:"
    print_status "  ./start_all.sh     - Start both backend and frontend"
    print_status "  ./start_backend.sh - Start only the backend"
    print_status "  ./start_frontend.sh - Start only the frontend"
    print_status ""
    print_status "Access the application at:"
    print_status "  Frontend: http://localhost:3000"
    print_status "  Backend API: http://localhost:8000"
    print_status "  API Docs: http://localhost:8000/docs"
}

# Run main function
main "$@"
