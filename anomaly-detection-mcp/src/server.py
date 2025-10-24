import datetime
from mcp.server.fastmcp import FastMCP
from logger import logger_config
from typing import Optional, Dict, Any
from pydantic import Field
import os
from dotenv import load_dotenv

# Importações das lógicas locais
from models.zscore import run_zscore
from models.isolation_forest import run_isolation_forest
from models.matrix_profile import run_matrix_profile
from models.prophet import run_prophet

load_dotenv()

logger = logger_config(process_name="server")
mcp = FastMCP("Anomaly Detection MCP")

@mcp.tool(description="List all available anomaly detection models.")
def list_models() -> Dict[str, Any]:
    return {
        "models": [
            "1 - isolation_forest",
            "2 - zscore",
            "3 - matrix_profile",
            "4 - prophet"
        ]
    }

@mcp.tool(description="Detect anomalies using Isolation Forest.")
def isolation_forest(
    prom_query: str = Field(..., description="Prometheus query string"),
    end_date: datetime = Field(..., description="End time in ISO 8601 UTC format (e.g., '2025-07-29T18:30:00Z'), most specific time based on the current date"),
    query_duration_days: int = Field(..., ge=1, description="How many days to query (must be >= 1)"),
    interval: str = Field(..., description="Prometheus query interval (e.g., 1m, 5m)"),
    days_train: int = Field(..., description="Number of days used for training (must be <= query_duration_days)"),
    contamination_level: float = Field(..., description="Contamination level (between 0.001 and 0.1)")
) -> Dict[str, Any]:
    try:
        result = run_isolation_forest(
            prom_query=prom_query,
            end_date=end_date,
            query_duration_days=query_duration_days,
            interval=interval,
            days_train=days_train,
            contamination_level=contamination_level
        )
        return {"result": result}
    except Exception as e:
        return {"error": f"Failed to run Isolation Forest detection: {str(e)}"}

@mcp.tool(description="Detect anomalies using Z-Score.")
def zscore(
   prom_query: str = Field(..., description="Prometheus query string"),
    end_date: datetime = Field(..., description="End time in ISO 8601 UTC format (e.g., '2025-07-29T18:30:00Z'), most specific time based on the current date"),
    query_duration_days: int = Field(..., ge=1, le=7, description="How many days back to query (1 to 7)"),
    interval: str = Field(..., description="Prometheus query interval (e.g., 1m, 5m)"),
    window: int = Field(..., ge=2, le=12, description="Rolling window size for Z-Score (2 to 12). Default value is 5")
) -> Dict[str, Any]:
    try:
        result = run_zscore(
            prom_query=prom_query,
            end_date=end_date,
            query_duration_days=query_duration_days,
            interval=interval,
            window=window
        )
        return {"result": result}
    except Exception as e:
        return {"error": f"Failed to run Z-Score detection: {str(e)}"}

@mcp.tool(description="Detect anomalies using Matrix Profile.")
def matrix_profile(
    prom_query: str = Field(..., description="Prometheus query string"),
    end_date: datetime = Field(..., description="End time in ISO 8601 UTC format (e.g., '2025-07-29T18:30:00Z'), most specific time based on the current date"),
    query_duration_days: int = Field(..., ge=1, le=7, description="How many days to query (1 to 7)"),
    interval: str = Field(..., description="Prometheus query interval (e.g., 1m, 5m)"),
    m: int = Field(..., ge=2, le=8, description="Window size for matrix profile (recommended: 2 to 8)"),
    std_multiplier: float = Field(..., ge=0.1, le=10.0, description="Standard deviation multiplier for threshold (e.g., 3.0)")
) -> Dict[str, Any]:
    try:
        result = run_matrix_profile(
            prom_query=prom_query,
            end_date=end_date,
            query_duration_days=query_duration_days,
            interval=interval,
            m=m,
            std_multiplier=std_multiplier
        )
        return {"result": result}
    except Exception as e:
        return {"error": f"Failed to run Matrix Profile detection: {str(e)}"}

@mcp.tool(description="Detect anomalies using Prophet.")
def prophet(
    prom_query: str = Field(..., description="Prometheus query string"),
    query_duration_days: int = Field(..., description="How many days to query (must be >= 1)"),
    interval: str = Field(..., description="Prometheus query interval (e.g., 1m, 5m)"),
    days_train: int = Field(..., description="Number of days used for training (must be <= query_duration_days)"),
    end_date: Optional[str] = Field(None, description="End time in ISO 8601 UTC format, optional")
) -> Dict[str, Any]:
    try:
        result = run_prophet(
            prom_query=prom_query,
            query_duration_days=query_duration_days,
            interval=interval,
            days_train=days_train,
            end_date=end_date
        )
        return {"result": result}
    except Exception as e:
        return {"error": f"Failed to run Prophet detection: {str(e)}"}

if __name__ == "__main__":
    logger.info("Starting Anomaly Detection MCP server")
    mcp.run()
