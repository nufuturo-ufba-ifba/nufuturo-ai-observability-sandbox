import os
import pandas as pd
from prometheus_pandas import query
from prophet import Prophet
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import logging
from dateutil.parser import parse as parse_datetime

from dotenv import load_dotenv
load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

logger = logging.getLogger("prophet")

def run_prophet(
    prom_query: str,
    end_date: str | datetime | None = None,
    query_duration_days: int = 1,
    interval: str = "1m",
    days_train: int = 1
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

        # Validate days_train
        if not (1 <= days_train <= query_duration_days):
            raise ValueError("days_train must be between 1 and query_duration_days")

        # Query Prometheus
        start_dt = end_dt - timedelta(days=query_duration_days)
        start_ts, end_ts = start_dt.isoformat(), end_dt.isoformat()

        prom_client = query.Prometheus(PROMETHEUS_URL)
        prom_data = prom_client.query_range(prom_query, start_ts, end_ts, interval)

        if prom_data.empty:
            raise ValueError("No data returned from Prometheus")

        # Process data
        df = pd.DataFrame({
            "time": prom_data.index.to_numpy(),
            "values": pd.to_numeric(prom_data.values[:, 0] if prom_data.values.ndim > 1 else prom_data.values, errors="coerce")
        }).dropna()

        if df.empty:
            raise ValueError("Invalid or empty data after cleaning")

        # Split train and test
        train_size = int((days_train / query_duration_days) * len(df))
        train_size = min(train_size, len(df) - 1)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]

        if train.empty or test.empty:
            raise ValueError("Train or test set is empty")

        # Prophet model
        model = Prophet(interval_width=0.997, weekly_seasonality=False, yearly_seasonality=False, growth='flat')
        model.fit(train.rename(columns={"time": "ds", "values": "y"}))

        # Forecast
        forecast = model.predict(test.rename(columns={"time": "ds"}))
        forecast.rename(columns={"ds": "time"}, inplace=True)

        # Merge and detect anomalies
        merged = pd.merge(test, forecast[["time", "yhat_lower", "yhat_upper"]], on="time")
        merged["anomaly"] = (merged["values"] < merged["yhat_lower"]) | (merged["values"] > merged["yhat_upper"])

        # Plot
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"prophet_{timestamp}.png"
        relative_path = os.path.join(STATIC_DIR, image_filename)
        image_path = os.path.abspath(relative_path)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(merged["time"], merged["values"], label="Values", color="blue")
        ax.fill_between(merged["time"], merged["yhat_lower"], merged["yhat_upper"], color="gray", alpha=0.3, label="Prediction Interval")
        ax.scatter(merged[merged["anomaly"]]["time"], merged[merged["anomaly"]]["values"], color="red", label="Anomaly", zorder=5)
        ax.set_title("Prophet Anomaly Detection")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        # Prepare output
        merged["time"] = merged["time"].apply(lambda x: x.isoformat())
        return {
            "data": merged[["time", "values", "anomaly"]].to_dict(orient="records"),
            "image_path": image_path
        }

    except Exception as e:
        logger.error(f"Error running Prophet: {str(e)}")
        raise e