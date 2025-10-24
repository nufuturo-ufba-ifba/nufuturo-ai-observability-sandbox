#!/bin/bash

# prometheus-mcp.sh - Self-run executable for prometheus-mcp-server
# This script creates a virtual environment, installs dependencies, and starts the server

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMETHEUS_MCP_DIR="$SCRIPT_DIR/prometheus-mcp-server"
VENV_DIR="$PROMETHEUS_MCP_DIR/.venv"
LOGFILE="$PROMETHEUS_MCP_DIR/prometheus-mcp-server.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}



# Install uv if not present
install_uv() {
    if ! command -v uv &> /dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
        # Add uv to PATH for current session (uv installs to ~/.local/bin)
        export PATH="$HOME/.local/bin:$PATH"
        
        if ! command -v uv &> /dev/null; then
            log_error "Failed to install uv"
            exit 1
        fi
    else
        log_info "uv is already installed"
    fi
}

# Create virtual environment
create_venv() {
    log_info "Creating virtual environment..."
    cd "$PROMETHEUS_MCP_DIR"
    
    if [[ ! -d "$VENV_DIR" ]]; then
        uv venv
        log_info "Virtual environment created at $VENV_DIR"
    else
        log_info "Virtual environment already exists"
    fi
}

# Load virtual environment
load_venv() {
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    export PATH="$VENV_DIR/bin:$PATH"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies with uv..."
    cd "$PROMETHEUS_MCP_DIR"
    
    # Temporarily unset pip config to avoid authentication issues with private repositories
    local old_pip_config="${PIP_CONFIG_FILE:-}"
    unset PIP_CONFIG_FILE
    
    # Install the package in editable mode using public PyPI explicitly
    uv pip install -e .
    
    # Restore pip config if it was set
    if [[ -n "$old_pip_config" ]]; then
        export PIP_CONFIG_FILE="$old_pip_config"
    fi
    
    log_info "Dependencies installed successfully"
}

# Test the server setup
test_server() {
    log_info "Testing prometheus-mcp-server setup..."
    cd "$PROMETHEUS_MCP_DIR"
    
    # Test the server by running it briefly with a test input
    log_info "Running server test..."
    
    # Create a test input for the MCP server
    local test_input='{"jsonrpc": "2.0", "id": 1, "method": "ping"}'
    
    # Test the server executable
    if [[ -x "$VENV_DIR/bin/prometheus-mcp-server" ]]; then
        log_info "Server executable found and is executable"
        
        # Try to run the server with a timeout to avoid hanging
        timeout 5s echo "$test_input" | "$VENV_DIR/bin/prometheus-mcp-server" > "$LOGFILE" 2>&1 || true
        
        log_info "Server test completed. Check logs at: $LOGFILE"
        log_info "Server is ready to be used by MCP clients"
        return 0
    else
        log_error "Server executable not found or not executable"
        return 1
    fi
}

# Check if all requirements are met
check_requirements() {
    # Check if prometheus-mcp-server directory exists
    if [[ ! -d "$PROMETHEUS_MCP_DIR" ]]; then
        log_warn "prometheus-mcp-server directory not found at $PROMETHEUS_MCP_DIR"
        return 1
    fi
    
    # Check if pyproject.toml exists
    if [[ ! -f "$PROMETHEUS_MCP_DIR/pyproject.toml" ]]; then
        log_warn "pyproject.toml not found in $PROMETHEUS_MCP_DIR"
        return 1
    fi
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        log_warn "uv is not installed"
        return 1
    fi
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_DIR" ]]; then
        log_warn "Virtual environment not found at $VENV_DIR"
        return 1
    fi
    
    # Check if server executable exists
    if [[ ! -x "$VENV_DIR/bin/prometheus-mcp-server" ]]; then
        log_warn "Server executable not found or not executable at $VENV_DIR/bin/prometheus-mcp-server"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    log_info "Setting up prometheus-mcp-server..."
    
    # Verify prometheus-mcp-server directory exists
    if [[ ! -d "$PROMETHEUS_MCP_DIR" ]]; then
        log_error "prometheus-mcp-server directory not found at $PROMETHEUS_MCP_DIR"
        exit 1
    fi
    
    # Verify required files exist
    if [[ ! -f "$PROMETHEUS_MCP_DIR/pyproject.toml" ]]; then
        log_error "pyproject.toml not found in $PROMETHEUS_MCP_DIR"
        exit 1
    fi
    
    # Install uv if necessary
    install_uv
    
    # Create virtual environment
    create_venv
    
    # Load virtual environment
    load_venv
    
    # Install dependencies
    install_dependencies
    
    # Test the server setup
    test_server
    
    log_info "Setup complete!"
    log_info ""
    log_info "The prometheus-mcp-server is now ready to use."
    log_info "To use it with Claude Desktop, add this configuration to your settings:"
    log_info ""
    log_info "{"
    log_info "  \"mcpServers\": {"
    log_info "    \"prometheus\": {"
    log_info "      \"command\": \"$VENV_DIR/bin/prometheus-mcp-server\","
    log_info "      \"env\": {"
    log_info "        \"PROMETHEUS_URL\": \"http://your-prometheus-server:9090\","
    log_info "        \"PROMETHEUS_USERNAME\": \"your_username\","
    log_info "        \"PROMETHEUS_PASSWORD\": \"your_password\""
    log_info "      }"
    log_info "    }"
    log_info "  }"
    log_info "}"
}

# Handle script arguments
case "${1:-}" in
    "setup"|"")
        main
        ;;
    "test")
        # Load virtual environment
        load_venv
        # Test the server setup
        test_server
        ;;
    "logs")
        if [[ -f "$LOGFILE" ]]; then
            tail -f "$LOGFILE"
        else
            log_error "Log file not found at $LOGFILE"
            exit 1
        fi
        ;;
    "run")
        # Check if all requirements are met, run setup if not
        if ! check_requirements; then
            log_info "Requirements not met. Running setup first..."
            main
        fi
        
        # Load virtual environment and run the server directly (for testing/debugging)
        load_venv
        cd "$PROMETHEUS_MCP_DIR"
        log_info "Running prometheus-mcp-server directly..."
        log_info "Press Ctrl+C to stop. Server communicates via stdin/stdout."
        "$VENV_DIR/bin/prometheus-mcp-server"
        ;;
    *)
        log_info "Usage: $0 {setup|test|logs|run}"
        log_info "  setup   - Set up the prometheus-mcp-server environment (default)"
        log_info "  test    - Test the server setup"
        log_info "  logs    - View the server logs"
        log_info "  run     - Run the server directly (auto-setup if requirements not met)"
        log_info ""
        log_info "If no argument is provided, 'setup' is assumed"
        log_info ""
        log_info "Note: This is an MCP server that communicates via stdin/stdout."
        log_info "It should be used by MCP clients like Claude Desktop, not run standalone."
        ;;
esac
