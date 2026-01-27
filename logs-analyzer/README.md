# Logs-Analy

This project is a Streamlit application called **Logs-Analy**, used to analyze application logs.

## Prerequisites

* Python 3.x installed
* **uv** installed

If you don’t have **uv** installed yet, follow the official installation guide:

[https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)

## Running with Docker

If you prefer to use Docker, make sure Docker and Docker Compose are installed. Then, simply run:

```
docker-compose up -d
```

This will build and run the application in a container.
Access it at: `http://localhost:8501`

## How to run (using uv)

Follow the steps below to set up and run the project manually using **uv**:

1. **Navigate to the project directory**:

    ```
    cd logs-analyzer
    ```

2. **Create a virtual environment with uv**:

    ```
    uv venv
    ```

    This command creates a `.venv` folder in the project root.

3. **Activate the virtual environment**:

    Activating the virtual environment ensures that all dependencies are installed and run in an isolated environment.

    * On **Linux / macOS**:

      ```
      source .venv/bin/activate
      ```
    - On Windows:
      ```
      .venv\Scripts\activate
      ```

4. **Install the project in editable mode**:

    ```
    uv pip install -e .
    ```

5. **Run the Streamlit application**:

    ```
    streamlit run main.py
    ```

The application will open in your default browser. If it doesn’t open automatically, access:

```
http://localhost:8501
```