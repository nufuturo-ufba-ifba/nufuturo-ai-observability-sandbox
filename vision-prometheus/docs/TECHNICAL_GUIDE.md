## Vision Anomaly Detection Dashboard: Technical Guide

Vision is an analytical dashboard for detecting anomalies in Prometheus time-series data using statistical and machine learning algorithms. This guide outlines the workflow, configuration, and interpretation of results.

---

### Core Workflow

1. **Data Acquisition**  
    Define a PromQL query and timeframe on the Home page.

2. **Method Selection**  
    Choose one of four anomaly detection algorithms.

3. **Parameter Tuning**  
    Adjust algorithm-specific settings for sensitivity.

4. **Analysis**  
    Review results via interactive charts and anomaly tables.

---

## Home Page: Data Configuration

Before analysis, configure data retrieval from Prometheus:

### 1. Prometheus Query
- Enter a valid PromQL string (e.g., `rate(http_requests_total[5m])`).

### 2. End of Query Range
- Defaults to current system time (ISO 8601 format).

### 3. Length of Query Range (in days)
- Select 1–7 days using the slider.
- For Prophet, at least 2 days is recommended.

---

## Anomaly Detection Methods

### Isolation Forest
- **Contamination:** Proportion of expected anomalies (e.g., `0.01`).
- **Training Days:** Data window for model training.
- **Tree Count:** Number of trees in the ensemble.

### Z-Score
- **Threshold:** Points with Z-Score > ±4 are anomalies.
- **Time Window:** Rolling window for mean and standard deviation.

### Prophet
- **Confidence Interval:** Width of the normal band (e.g., 95%).
- **Seasonality:** Enable daily, weekly, or yearly cycles.
- **Trend Model:** Choose linear or flat trend.

### Matrix Profile
- **Window Size (m):** Pattern length to analyze.
- **Threshold (Sigma):** Sensitivity for discord detection.

---

## Interpreting the Results

Each analysis module provides:

### Summary Metrics
- Total anomalies detected
- Anomaly rate percentage

### Interactive Visualizations
- **Original Series:** Raw data with anomalies marked in red.
- **Algorithm Scores:** Underlying detection scores.

### Anomaly Details
- Searchable table with:
  - Timestamp
  - Metric value
  - Detection score

---

## Example: Running the Dashboard in Bash

```bash
# Clone the repository
git clone https://github.com/nufuturo-ufba-ifba/nufuturo-ai-observability-sandbox.git
cd nufuturo-ai-observability-sandbox

# Start the Streamlit dashboard
streamlit run streamlit-dashboard/app.py

# Open in the default browser (if $BROWSER is set)
"$BROWSER" http://localhost:8501
```

---

For further details, refer to the codebase and module documentation.