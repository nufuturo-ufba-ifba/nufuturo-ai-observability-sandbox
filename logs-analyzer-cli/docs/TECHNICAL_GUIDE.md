# Technical Guide: Logs Analyzer CLI

The Logs Analyzer CLI is a high-performance utility for parsing and filtering unformatted text logs (`.txt`). It isolates critical errors and exceptions from raw output, with logic tailored for Java and Clojure environments. By stripping noise and grouping related log lines, developers can focus on actual system failures.

## Technical Overview

The analyzer uses **Block-Based Filtering** logic. Unlike simple line-by-line filters, it understands that errors (especially in Java or Clojure) often span multiple lines (e.g., stack traces).

### 1. ANSI De-coloring

Raw logs from CI/CD or terminals often contain ANSI color codes (e.g., `\x1B[31m`). The CLI automatically detects and strips these codes before analysis, ensuring keyword matching regardless of formatting.

### 2. Block Detection

The tool identifies the start of a new log entry using patterns:

- **ISO Timestamps**: Lines like `2026-01-27T16:00:00Z`
- **Test Runners**: Flutter/Dart indicators like `00:00 +0`
- Any line not starting with these patterns is considered a continuation of the previous block (e.g., stack trace or error body).

### Intelligence and Keywords

The CLI scans each block for **Important Keywords**. If any line in a block contains a keyword, the entire block is marked as important.

### Supported Error Patterns

Optimized for:

- **Clojure**: `clojure.lang.ExceptionInfo`, `FAIL in`, test failure summaries
- **Java**: Standard exceptions (`java.lang.ArithmeticException`), stack traces
- **Flutter/Dart**: `Error:`, `Compilation failed`, test failures
- **Generic**: Standard `ERROR` tags

## Operational Workflow

### Execution

Run the tool via command line, passing the path to a raw `.txt` log file:

```bash
python3 main.py <path_to_log_file>
```

### Analysis Process

- **Parsing**: Reads the input file, grouping lines into chronological blocks.
- **Filtering**: Discards `INFO` or `DEBUG` blocks without important keywords.
- **Terminal Output**: Prints important blocks in real-time for inspection.
- **Persistence**: Creates a new file with suffix `-filtrado-TIMESTAMP.txt` containing only critical blocks.

## Understanding the Output

- **Success with Errors**: Displays total lines written and the path to the filtered file—your "error-only" log view.
- **No Errors Found**: Reports no errors found and cleans up the empty output file, providing a quick "Go/No-Go" status.

## Usage Best Practices

- **Large Log Handling**: Uses streaming read for minimal memory consumption.
- **Technology Specificity**: Best for identifying multi-line error structures in Java and Clojure VMs, but works for generic logs.
