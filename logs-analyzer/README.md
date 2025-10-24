# LogsNubank

This project is a Streamlit application called LogsNubank, which analyzes logs.

## Prerequisites

- Python 3.x installed.
- Access to a terminal or command line.

## Running with Docker

If you prefer to use Docker, ensure Docker and Docker Compose are installed. Then, run:

```
docker-compose up -d
```

This will build and run the application in a container. Access it at `http://localhost:8501`.

## How to run

Follow the steps below to set up and run the project manually:

1. **Clone the repository** (if applicable) or navigate to the project directory:
    ```
    cd /workspaces/LOGS
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

5. **Run the Streamlit application**:
    ```
    streamlit run main.py
    ```

The application will open in the default browser. If it doesn't open automatically, use `"$BROWSER" http://localhost:8501` in the terminal.

## Notes

- Ensure that the `requirements.txt` and `main.py` files are present in the directory.
- To deactivate the virtual environment, run `deactivate`.