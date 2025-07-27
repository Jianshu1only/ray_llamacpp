#!/bin/bash
# Ray Llama Deployment - Installation Script

set -e

echo "=========================================="
echo "Ray Llama Deployment - Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
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

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check pip
check_pip() {
    print_status "Checking pip..."
    
    if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip not found. Please install pip"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Use pip3 if available, otherwise pip
    PIP_CMD="pip"
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    fi
    
    $PIP_CMD install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
}

# Check Ray installation
check_ray() {
    print_status "Checking Ray installation..."
    
    if python3 -c "import ray; print('Ray version:', ray.__version__)" 2>/dev/null; then
        print_success "Ray is working correctly"
    else
        print_error "Ray installation failed"
        exit 1
    fi
}

# Check llama.cpp
check_llama_cpp() {
    print_status "Checking llama.cpp installation..."
    
    LLAMA_PATH="/mnt/weka/home/jianshu.she/jianshu/llama.cpp/build/bin/llama-cli"
    
    if [ -f "$LLAMA_PATH" ]; then
        print_success "llama-cli found at $LLAMA_PATH"
    else
        print_warning "llama-cli not found at expected location"
        print_warning "You can still use mock mode for testing"
        print_warning "For real deployments, update the path in config.py"
    fi
}

# Run quick test
run_test() {
    print_status "Running quick test..."
    
    if python3 ray_llama_flexible.py --mock --num-instances 2 --test-only &>/dev/null; then
        print_success "Quick test passed!"
    else
        print_error "Quick test failed"
        print_warning "Try running manually: python3 ray_llama_flexible.py --mock --test-only"
    fi
}

# Set permissions
set_permissions() {
    print_status "Setting executable permissions..."
    
    chmod +x ray_llama_flexible.py
    chmod +x examples/demo.py
    chmod +x examples/test_client.py
    chmod +x install.sh
    
    print_success "Permissions set"
}

# Create symlinks (optional)
create_symlinks() {
    if [ "$1" = "--symlinks" ]; then
        print_status "Creating symlinks..."
        
        # Create symlink to make it easier to run
        if [ -w /usr/local/bin ]; then
            ln -sf "$(pwd)/ray_llama_flexible.py" /usr/local/bin/ray-llama
            print_success "Created symlink: ray-llama"
        else
            print_warning "Cannot create symlink in /usr/local/bin (permission denied)"
            print_warning "You can create it manually: sudo ln -sf $(pwd)/ray_llama_flexible.py /usr/local/bin/ray-llama"
        fi
    fi
}

# Print usage information
print_usage() {
    echo ""
    echo "=========================================="
    echo "Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Quick Start Commands:"
    echo "  Test with mock servers:"
    echo "    python3 ray_llama_flexible.py --mock --test-only"
    echo ""
    echo "  Run demo:"
    echo "    python3 examples/demo.py"
    echo ""
    echo "  Deploy 8 mock instances:"
    echo "    python3 ray_llama_flexible.py --mock --num-instances 8"
    echo ""
    echo "  With real model:"
    echo "    python3 ray_llama_flexible.py --model-path /path/to/model.gguf --num-instances 16"
    echo ""
    echo "Documentation:"
    echo "  README.md - Main documentation"
    echo "  docs/USAGE_GUIDE.md - Detailed usage guide"
    echo "  examples/ - Example scripts and tests"
    echo ""
    echo "Makefile targets:"
    echo "  make test     - Run all tests"
    echo "  make demo     - Run demo scenarios"
    echo "  make clean    - Clean temporary files"
    echo ""
}

# Main installation flow
main() {
    echo "Starting installation process..."
    echo ""
    
    # Check requirements
    check_python
    check_pip
    
    # Install and verify
    install_dependencies
    check_ray
    check_llama_cpp
    set_permissions
    
    # Optional symlinks
    create_symlinks "$1"
    
    # Test installation
    run_test
    
    # Show usage
    print_usage
    
    print_success "Installation completed successfully!"
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --symlinks    Create symlinks for easier command access"
        echo "  --help        Show this help message"
        echo ""
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac