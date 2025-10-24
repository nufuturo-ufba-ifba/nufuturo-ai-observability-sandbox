from server import mcp
from logger import logger_config

def main():
    """
    Main function to start the MCP server.
    """
    logger = logger_config(process_name="main")
    
    # Start the MCP server
    logger.info("Starting Anomaly Detection MCP server...")
    mcp.run(transport="stdio")
    logger.info("Anomaly Detection MCP server started successfully.")

if __name__ == "__main__":
    main()