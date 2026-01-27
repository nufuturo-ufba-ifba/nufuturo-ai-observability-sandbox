# nufuturo-ai-observability-sandbox
NuFuturo program dedicated repository for AI-powered observability and monitoring tools.

## Repository Structure

This repository contains multiple components that work together to provide AI-powered observability and monitoring capabilities:


### **prometheus-mcp-server/**
A Model Context Protocol (MCP) server that enables AI assistants to interact with Prometheus metrics:
- Provides tools to query Prometheus metrics
- Executes PromQL queries (both instant and range queries)
- Lists available metrics and their metadata
- Retrieves scrape targets information
- Designed to work with Claude Desktop and other MCP clients

### **anomaly-detection-mcp/**
A Model Context Protocol (MCP) service for anomaly detection in time series data:
- Provides various anomaly detection algorithms (Isolation Forest, Matrix Profile, Prophet, Z-Score)
- Exposes endpoints for AI assistants to query anomaly detection results
- Designed for integration with Claude Desktop and other MCP clients

### **streamlit-dashboard/** (NuVision Tool)
A comprehensive Streamlit-based dashboard for monitoring and anomaly detection:
- Interactive web interface for metrics visualization
- Real-time anomaly detection using multiple algorithms
- Custom alerts and notifications
- Integration with Prometheus metrics
- Advanced analytics and reporting capabilities

### **logs-analyzer/**
An AI-equipped Streamlit dashboard for log file analysis and visualization:
- Interactive web interface for exploring and filtering logs
- Automatic grouping of similar error messages to identify root causes
- Visualization of log volume and error rates over time
### **logs-analyzer CLI/**
An AI-powered CLI tool for log file analysis using Ollama models:
- Leverages local Ollama models to analyze and interpret log files
- Provides command-line interface for querying and summarizing logs

### **prometheus-mcp.sh**
A self-contained shell script that sets up and runs the Prometheus MCP server automatically.

---

## Quick Setup - Prometheus MCP Server

### Using the Shell Script (Recommended)

The easiest way to set up the Prometheus MCP server is using the provided shell script from the root folder:

```bash
# From the root directory (nufuturo-ai-observability-sandbox/)
./prometheus-mcp.sh setup
```

This script will:
- Install `uv` (Python package manager) if not already installed
- Create a virtual environment
- Install all required dependencies
- Test the server setup
- Provide configuration instructions for Claude Desktop

### Available Commands

```bash
# Setup the server (default action)
./prometheus-mcp.sh setup

# Test the server setup
./prometheus-mcp.sh test

# View server logs
./prometheus-mcp.sh logs

# Run the server directly (for testing/debugging)
./prometheus-mcp.sh run
```

### Configuration for Claude Desktop

After running the setup, add this configuration to your Claude Desktop settings (`~/.cursor/mcp.json` or similar):

```json
{
   "mcpServers": {
      "prometheus": {
         "command": "/path/to/nufuturo-ai-observability-sandbox/prometheus-mcp.sh",
         "args": ["run"],
         "env": {
            "PROMETHEUS_URL": "http://your-prometheus-server:9090",
            "PROMETHEUS_USERNAME": "your_username",
            "PROMETHEUS_PASSWORD": "your_password"
         }
      }
   }
}
```

Replace the environment variables with your actual Prometheus server details.

---

## Detailed Setup Instructions

### How to run Prometheus MCP Server

1. **Option 1: Using the shell script (recommended)**
    ```bash
    ./prometheus-mcp.sh setup
    ```

2. **Option 2: Manual setup**
    ```bash
    cd prometheus-mcp-server
    ```
    Follow the detailed instructions in the [prometheus-mcp-server README](prometheus-mcp-server/README.md) to configure and run the service.

### How to run NuVision Dashboard

1. Go to the `streamlit-dashboard` directory:
    ```bash
    cd streamlit-dashboard
    ```
2. Follow the detailed instructions in the [streamlit-dashboard README](streamlit-dashboard/README.md) to configure and run the service.

### How to run Anomaly Detection MCP

1. Go to the `anomaly-detection-mcp` directory:
    ```bash
    cd anomaly-detection-mcp
    ```
2. Follow the detailed instructions in the [anomaly-detection-mcp README](anomaly-detection-mcp/README.md) to configure and run the service.


### How to run Logs Analyzer

1. Go to the `logs-analyzer` directory:
    ```bash
    cd logs-analyzer
    ```
2. Follow the detailed instructions in the [logs-analyzer README](logs-analyzer/README.md) to configure and run the service.

### How to run Logs Analyzer CLI

1. Go to the `logs-analyzer-cli` directory:
    ```bash
    cd logs-analyzer-cli
    ```
2. Follow the detailed instructions in the [logs-analyzer-cli README](logs-analyzer-cli/README.md) to configure and run the service.

---

## Integration

These components can work together to provide a complete observability solution:

1. **Prometheus MCP Server** provides AI assistants with access to your metrics
2. **NuVision Dashboard** offers interactive visualization and analysis
3. **Anomaly Detection MCP** provides an MCP interface for running anomaly detection algorithms on time series data
4. **Logs Analyzer** offers AI-powered analysis and visualization of log files
4. **Logs Analyzer CLI** offers AI-powered analysis and visualization of log files using Ollama models



The MCP server is particularly useful for integrating with AI assistants like Claude Desktop, allowing natural language queries about your infrastructure metrics.

