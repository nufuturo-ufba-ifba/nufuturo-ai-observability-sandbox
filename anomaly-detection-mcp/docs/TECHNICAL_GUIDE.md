# Technical Guide: Anomaly Detection MCP

This guide explains how to use an AI assistant (LLM) with the Anomaly Detection MCP server to analyze Prometheus metrics (e.g., GC cycles, memory leaks).

## Analysis Workflow

### 1. Context Discovery

```bash
# LLM calls list_models to discover available algorithms
```
- Selects method (e.g., Z-Score for spikes, Prophet for trends).

### 2. Parameter Mapping

```bash
# Example configuration for forced GC cycles metric
prom_query="go_gc_cycles_forced_gc_cycles_total"
query_duration_days=2
interval="1m"
end_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
```

### 3. Execution and Data Retrieval

- MCP queries Prometheus.
- Runs chosen algorithm (e.g., Z-Score, Isolation Forest).
- Returns JSON with:
    - **Anomaly Flags**: Boolean list for detected anomalies.
    - **Statistical Evidence**: Values vs. thresholds.
    - **Visual Confirmation**: Path to generated plot.

## Example: Forced GC Cycles Analysis

```json
{
    "prom_query": "go_gc_cycles_forced_gc_cycles_total",
    "query_duration_days": 2,
    "interval": "1m",
    "window": 5
}
```

- LLM interprets anomaly flags:
    - **Correlation**: Finds timestamps with anomaly=true.
    - **Severity**: Checks Z-Score (e.g., 10.0 = major event).
    - **Conclusion**: Explains findings (e.g., "Anomaly at 14:30 UTC, forced GC cycles spiked to 5, indicating memory pressure.").

## Technical Limitations

- **Metric Resolution**: Large intervals (e.g., 1h) may miss short spikes.
- **Data Freshness**: Always use latest UTC time for end_date.
- **Lookback Window**: Limited to last 7 days; older data unavailable.
