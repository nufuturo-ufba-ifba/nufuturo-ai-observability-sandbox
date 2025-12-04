# logs-analyzer-cli: Analyzer for pipeline .txt logs

This is a simple CLI script to filter important error messages from unstructured `.txt` log files.

It works by grouping log lines into "blocks" (like an exception and its stack trace) and then checking those blocks against a list of keywords to find errors.

## Prerequisites

- Python 3.x installed.
- [uv](https://github.com/astral-sh/uv) installed (a fast Python installer).
- Access to a terminal or command line.

## How to Run

1.  **Navigate to the project directory**:
    ```bash
    cd logs-analyzer-cli
    ```

2.  **Create and activate the virtual environment** (optional, but good practice):
    ```bash
    uv venv
    source .venv/bin/activate
    ```
    *(On Windows, use: `.venv\Scripts\activate`)*

3.  **Install dependencies**:
    ```bash
    uv pip install -e .
    ```

4.  **Run the analyzer**:
    The script will print the filtered logs directly to your terminal AND save a copy to a new timestamped file (e.g., `logs-filtrado-20251031143000.txt`).

    ```bash
    python3 main.py <path_to_log_file.txt>
    ```

    **Example**:
    ```bash
    python3 main.py logs.txt
    ```

## Script Logic for Multi-Line Error Messages

The script's logic is designed to handle multi-line error messages (like stack traces):

### Read Line by Line
It reads the original log file one line at a time.

### Detect Blocks
It uses a Regular Expression (`block_start_regex`) to identify lines that start with a timestamp (e.g., `2025-08-13T...` or `00:00 +0...`). This line marks the beginning of a "log block".

### Group Stack Traces
Any line that does not start with a timestamp is considered a continuation (part of the stack trace) and is grouped with the previous block.

### Check for Errors
When the script finds the next timestamp, it stops and checks the entire block it just collected.

### Filter by Keyword
It checks if any line in that block contains one of the keywords from the `IMPORTANT_KEYWORDS` list (e.g., `ERROR`, `Exception`, `failed`).

### Print and Save
If an important keyword is found, the entire block (from the first timestamp to the last line of its stack trace) is printed to the terminal AND saved to the new `...-filtrado-....txt` file.

### Discard
If no keyword is found, the entire block is discarded.

## How to Customize

You can easily change what the script filters by editing the `IMPORTANT_KEYWORDS` list directly inside `main.py`.

```python
    # Keywords that mark a block as "important"
    IMPORTANT_KEYWORDS = [
        'clojure.lang.ExceptionInfo',   # Clojure errors/exceptions
        'ERROR',                        # Generic errors
        'Error:',                       # Dart/Flutter errors
        'Some tests failed.',           # Test failure summary
        'java.lang.ArithmeticException',# Other Java exceptions
        'Compilation failed',           # Flutter compilation error
        'FAIL in',                      # Clojure test failure
        'Tests failed.'                 # Clojure test failure summary
    ]

