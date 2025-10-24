import os
import pandas as pd
from prometheus_pandas import query
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from dateutil.parser import parse as parse_datetime

from dotenv import load_dotenv
load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
os.makedirs(STATIC_DIR, exist_ok=True)

logger = logging.getLogger("isolation_forest")

def run_isolation_forest(
    prom_query: str,
    end_date: str | datetime | None = None,
    query_duration_days: int = 1,
    interval: str = "1m",
    days_train: int = 1,
    contamination_level: float = 0.1
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
        if not (1 <= days_train <= query_duration_days):
            raise ValueError("days_train must be between 1 and query_duration_days")
        if not (0.001 <= contamination_level <= 0.1):
            raise ValueError("contamination_level must be between 0.001 and 0.1")

        # Query Prometheus
        start_dt = end_dt - timedelta(days=query_duration_days)
        start_ts, end_ts = start_dt.isoformat(), end_dt.isoformat()

        prom_client = query.Prometheus(PROMETHEUS_URL)
        prom_data = prom_client.query_range(prom_query, start_ts, end_ts, interval)

        if prom_data is None or prom_data.empty:
            raise ValueError("No data returned from Prometheus")

        if isinstance(prom_data, pd.DataFrame):
            values = prom_data.iloc[:, 0].values
            times = prom_data.index.to_numpy()
        else:
            values = prom_data.values
            times = prom_data.index.to_numpy()

        df = pd.DataFrame({
            "time": times,
            "values": pd.to_numeric(values, errors="coerce")
        }).dropna(subset=["values"]).reset_index(drop=True)

        if df.empty:
            raise ValueError("No data returned from Prometheus after cleaning")

        # Train Isolation Forest
        train_size = int((days_train / query_duration_days) * len(df))
        train_size = min(train_size, len(df) - 1)
        X_train = df.iloc[:train_size]["values"].values.reshape(-1, 1)

        model = IsolationForest(contamination=contamination_level, random_state=42)
        model.fit(X_train)

        # Predict anomalies
        df["anomaly"] = False
        if train_size < len(df):
            X_test = df.iloc[train_size:]["values"].values.reshape(-1, 1)
            preds = model.predict(X_test)
            df.loc[train_size:, "anomaly"] = preds == -1

        # Plot
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"isolation_forest_{timestamp}.png"
        relative_path = os.path.join(STATIC_DIR, image_filename)
        image_path = os.path.abspath(relative_path)

        df["time"] = pd.to_datetime(df["time"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time"], df["values"], label="Values", color="steelblue")
        ax.scatter(df[df["anomaly"]]["time"], df[df["anomaly"]]["values"],
                   color="crimson", label="Anomaly", zorder=5)
        ax.set_title("Isolation Forest Anomaly Detection")
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

        # Prepare output
        df["time"] = df["time"].apply(lambda x: x.isoformat())
        return {
            "data": df[["time", "values", "anomaly"]].to_dict(orient="records"),
            "image_path": image_path
        }

    except Exception as e:
        logger.error(f"Error running Isolation Forest: {str(e)}")
        raise e