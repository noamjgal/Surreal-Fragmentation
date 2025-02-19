"""
Centralized path configuration for SURREAL project

Usage:
1. Default behavior: Data stored in SURREAL/data/
2. To override data location:
set os.environ['SURREAL_DATA_DIR'] = "/your/custom/path"
"""

# "/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main"
import os
from pathlib import Path

# Base data directory (override with SURREAL_DATA_DIR environment variable)
DATA_DIR = Path(os.environ.get("SURREAL_DATA_DIR", 
                Path(__file__).parent.parent / "data"))  # Default: SURREAL/data

# Raw data locations
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Standardized subdirectories
EPISODE_OUTPUT_DIR = PROCESSED_DATA_DIR / "episodes"
MAP_OUTPUT_DIR = PROCESSED_DATA_DIR / "maps"
GPS_PREP_DIR = PROCESSED_DATA_DIR / "gps_preprocessed"

# Create directories if they don't exist
_ = [d.mkdir(parents=True, exist_ok=True) for d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EPISODE_OUTPUT_DIR, 
    MAP_OUTPUT_DIR, GPS_PREP_DIR
]] 