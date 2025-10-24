import os
import pandas as pd
from prometheus_pandas import query
from scipy import stats
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from dateutil.parser import parse as parse_datetime

from dotenv import load_dotenv
load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

logger = logging.getLogger("zscore")

def run_zscore(
    prom_query: str,
    end_date: str | datetime,
    query_duration_days: int,
    interval: str,
    window: int
) -> dict:
    try:
        if PROMETHEUS_URL is None:
            raise ValueError("PROMETHEUS_URL not set in environment variables")

        # Converte string para datetime se necessário
        if isinstance(end_date, str):
            end_date = parse_datetime(end_date)

        now = datetime.now(timezone.utc)
        if end_date.tzinfo is None:
            end_dt = end_date.replace(tzinfo=timezone.utc)
        else:
            end_dt = end_date.astimezone(timezone.utc)

        min_date = now - timedelta(days=7)
        if end_dt > now:
            end_dt = now
        if end_dt < min_date:
            raise ValueError("end_date cannot be older than 7 days ago")

        start_dt = end_dt - timedelta(days=query_duration_days)
        start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())

        if not (2 <= window <= 12):
            raise ValueError("window must be between 2 and 12")

        prom_client = query.Prometheus(PROMETHEUS_URL)
        prom_data = prom_client.query_range(prom_query, start_ts, end_ts, interval)

        if isinstance(prom_data, pd.Series):
            prom_data = prom_data.to_frame(name="value")

        if prom_data.empty or prom_data.shape[1] == 0:
            raise ValueError("No data returned from Prometheus")

        vals = prom_data.values
        if vals.ndim > 1:
            vals = vals[:, 0]

        df = pd.DataFrame({
            "time": prom_data.index.to_numpy(),
            "values": pd.to_numeric(vals, errors="coerce")
        }).dropna(subset=["values"])

        if df.empty:
            raise ValueError("No numeric data to process")

        def trimmed_mean(x):
            return stats.trim_mean(x, 0.1)

        def trimmed_std(x):
            return stats.mstats.trimmed_std(x, limits=(0.1, 0.1))

        df["mean_values"] = df["values"] \
            .rolling(window=window, closed="left", min_periods=1) \
            .apply(trimmed_mean, raw=True)

        df["std_values"] = df["values"] \
            .rolling(window=window, closed="left", min_periods=1) \
            .apply(trimmed_std, raw=True)

        df["std_values"] = df["std_values"].replace(0, 1e-8)
        df["zscore"] = (df["values"] - df["mean_values"]) / df["std_values"]
        df["anomaly"] = df["zscore"].abs() > 4
        df["time"] = pd.to_datetime(df["time"])

        # Salvar imagem na pasta static
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"zscore_{timestamp}.png"
        relative_path = os.path.join(STATIC_DIR, image_filename)
        image_path = os.path.abspath(relative_path)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time"], df["values"], label="Values", color="steelblue")
        ax.scatter(df[df["anomaly"]]["time"], df[df["anomaly"]]["values"],
                   color="crimson", label="Anomaly", zorder=5)

        ax.set_title("Z-Score Anomaly Detection")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        df["time"] = df["time"].map(lambda x: x.isoformat())

        return {
            "data": df[["time", "values", "anomaly"]].to_dict(orient="records"),
            "image_path": image_path
        }

    except Exception as e:
        logger.error(f"Error running Z-Score: {e}")
        raise e
