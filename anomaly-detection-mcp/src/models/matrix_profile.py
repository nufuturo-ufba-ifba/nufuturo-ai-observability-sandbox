import os
import pandas as pd
import numpy as np
import stumpy
from prometheus_pandas import query
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import logging
from dateutil.parser import parse as parse_datetime

from dotenv import load_dotenv
load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

logger = logging.getLogger("matrix_profile")

def run_matrix_profile(
    prom_query: str,
    end_date: str | datetime | None = None,
    query_duration_days: int = 1,
    interval: str = "1m",
    m: int = 2,
    std_multiplier: float = 3.0
) -> dict:
    try:
        if PROMETHEUS_URL is None:
            raise ValueError("PROMETHEUS_URL not set in environment variables")

        # Handle end_date
        if end_date is None:
            end_dt = datetime.now(timezone.utc)
        elif isinstance(end_date, str):
            end_dt = parse_datetime(end_date.replace("Z", "+00:00"))
        else:
            end_dt = end_date

        # Validate end_date
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        else:
            end_dt = end_dt.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        min_date = now - timedelta(days=7)
        if end_dt > now:
            end_dt = now
        if end_dt < min_date:
            raise ValueError("end_date cannot be older than 7 days ago")

        # Validate parameters
        if not (1 <= query_duration_days <= 7):
            raise ValueError("query_duration_days must be between 1 and 7")
        if not (2 <= m <= 8):
            raise ValueError("m must be between 2 and 8")
        if not (0.1 <= std_multiplier <= 10.0):
            raise ValueError("std_multiplier must be between 0.1 and 10.0")

        # Query Prometheus
        start_dt = end_dt - timedelta(days=query_duration_days)
        start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())

        prom_client = query.Prometheus(PROMETHEUS_URL)
        prom_data = prom_client.query_range(prom_query, start_ts, end_ts, interval)

        if prom_data is None or prom_data.empty or prom_data.shape[1] == 0:
            raise ValueError("No data returned from Prometheus")

        if isinstance(prom_data, pd.Series):
            prom_data = prom_data.to_frame(name="value")

        vals = prom_data.values
        if vals.ndim > 1:
            vals = vals[:, 0]

        df = pd.DataFrame({
            "time": prom_data.index.to_numpy(),
            "values": pd.to_numeric(vals, errors="coerce")
        }).dropna(subset=["values"])

        if len(df) < m:
            raise ValueError("Window size (m) is greater than available data")

        # Compute Matrix Profile
        values_float = df["values"].astype(float).values
        mp = stumpy.stump(values_float, m=m, normalize=True)

        mean_mp = np.mean(mp[:, 0])
        std_mp = np.std(mp[:, 0], dtype=np.float64)
        threshold = mean_mp + (std_multiplier * std_mp)

        anomaly_idx = np.where(mp[:, 0] > threshold)[0]
        df["anomaly"] = df.index.isin(anomaly_idx)

        # Plot
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"matrix_profile_{timestamp}.png"
        relative_path = os.path.join(STATIC_DIR, image_filename)
        image_path = os.path.abspath(relative_path)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time"], df["values"], label="Values", color="steelblue")
        ax.scatter(df[df["anomaly"]]["time"], df[df["anomaly"]]["values"],
                   color="crimson", label="Anomaly", zorder=5)
        ax.set_title("Matrix Profile Anomaly Detection")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        # Prepare output
        df["time"] = df["time"].map(lambda ts: pd.to_datetime(ts).isoformat())
        return {
            "data": df[["time", "values", "anomaly"]].to_dict(orient="records"),
            "image_path": image_path
        }

    except Exception as e:
        logger.error(f"Error running Matrix Profile: {str(e)}")
        raise e