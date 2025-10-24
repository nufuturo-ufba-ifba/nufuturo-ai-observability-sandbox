# anomaly-detection-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server for anomaly detection.

## Overview

This project implements an MCP server for time series anomaly detection using multiple algorithms.

## Supported Tools

- **matrix_pro**: Matrix Profile for time series anomaly detection.
- **prophet**: Time series forecasting and anomaly detection using Prophet.
- **isolation_forest**: Isolation Forest algorithm for anomaly detection.
- **zscore**: Z-score based anomaly detection.

## Setup

1. Copy the sample environment file:

  ```bash
  cp .env.sample .env
  ```

2. Install [`uv`](https://github.com/astral-sh/uv) for dependency management:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

3. Create a virtual environment and install dependencies:

  ```bash
  uv venv
  source .venv/bin/activate      # On Unix/macOS
  .venv\Scripts\activate         # On Windows
  uv pip install -e .
  ```

## Usage

Run the MCP server:

```bash
uv run src/main.py
```

Or use a task runner with a script like:

```json
"mcp-anomaly-detection": {
  "command": "uv",
  "args": [
   "--directory",
   "<project-root>/anomaly-detection-mcp",
   "run",
   "src/main.py"
  ]
}
```

Replace `<project-root>` with your project directory path, or run from the project root.

## Project Structure

```
anomaly-detection-mcp/
├── src/
│   └── anomaly-detection-mcp/
│       ├── __init__.py
│       ├── logger.py
│       ├── main.py
│       ├── server.py
│       └── models/
│           ├── matrix_pro.py
│           ├── prophet.py
│           ├── isolation_forest.py
│           └── zscore.py
├── .env.sample
├── .gitignore
├── pyproject.toml
└── README.md
```

## License

GNU GENERAL PUBLIC LICENSE

---
