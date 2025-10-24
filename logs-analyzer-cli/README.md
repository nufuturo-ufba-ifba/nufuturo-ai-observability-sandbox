#!/bin/bash

cat > README.md << 'EOF'
# logs-analyzer-cli

This project is a CLI application called logs-analyzer-cli, which analyzes logs using Drain3 and AI-powered insights with Ollama.

## Prerequisites

- Python 3.x installed.
- Access to a terminal or command line.
- Ollama installed and running (optional, for AI analysis).

## How to run

Follow the steps below to set up and run the project:

1. **Clone the repository** (if applicable) or navigate to the project directory:
    ```
    cd logs-analyzer-cli
    ```

2. **Create a virtual environment** (optional, but recommended):
    ```
    python3 -m venv .venv
    ```

3. **Activate the virtual environment**:
    - On Linux/Mac:
      ```
      source .venv/bin/activate
      ```
    - On Windows:
      ```
      venv\Scripts\activate
      ```

4. **Install the dependencies**:
    ```
    pip install -r requirements.txt
    ```

5. **Configure environment variables**:
    - Copy `.env.example` to `.env`:
      ```
      cp .env.example .env
      ```
    - Edit `.env` with your settings (model name, Ollama URL, etc.):
      ```
      USE_IA=True
      OLLAMA_MODEL=deepseek-coder
      OLLAMA_URL=http://localhost:11434
      ```

6. **Run the analyzer**:
    ```
    python3 analyzer.py <log-file.json>
    ```

    **Example**:
    ```
    python3 analyzer.py logs.json
    ```

## Usage

### Basic usage:
```
python3 analyzer.py logs.json
```

## Notes

- Ensure that the `requirements.txt` and `analyzer.py` files are present in the directory.
- To disable AI analysis, set `USE_IA=False` in your `.env` file.
- Ollama must be running for AI analysis to work.
- To deactivate the virtual environment, run `deactivate`.
EOF