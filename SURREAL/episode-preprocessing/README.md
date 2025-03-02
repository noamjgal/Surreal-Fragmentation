# SURREAL Episode Processing & Fragmentation Analysis Pipeline

A comprehensive pipeline for processing GPS and app usage data to detect behavioral episodes and analyze behavioral fragmentation patterns. These patterns are then associated with psychological data.

## Overview

This pipeline analyzes digital behavior and mobility patterns to detect episodes of activty and assess the level of  "fragmentation" - the degree to which activities are broken up into smaller, disconnected episodes throughout the day. It integrates data from:

- GPS tracking devices
- Smartphone app usage logs  
- Ecological Momentary Assessment (EMA) responses

## Folder Structure for relevant files in the SURREAL Repo

```
SURREAL/
├── data/
│   ├── raw/
│   │   └── Participants/
│   │       └── P*/9 - Smartphone Tracking App/*.csv
│   ├── processed/
│   │   ├── gps_prep/        # Preprocessed GPS data
│   │   ├── episodes/        # Detected behavioral episodes
│   │   ├── fragmentation/   # Fragmentation metrics
│   │   ├── ema/             # Processed EMA responses
│   │   ├── maps/            # Visualization outputs
│   │   └── combined/        # Integrated datasets
├── config/
│   └── paths.py             # Centralized path configuration
└── episode-preprocessing/
    ├── preprocess-gps.py    # GPS data preprocessing
    ├── detect_episodes.py   # Behavioral episode detection
    ├── daily_fragmentation.py # Fragmentation metrics calculation
    ├── window_fragmentation.py # EMA-linked fragmentation analysis
    ├── map-episodes.py      # Visualization generation
    ├── combine_metrics.py   # Data integration
    └── README.md
```

## Core Concepts

### Episodes
The project identifies three types of behavioral episodes:
- **Digital Episodes**: Periods of smartphone screen activity
- **Mobility Episodes**: Periods of physical movement between locations
- **Overlap Episodes**: Concurrent digital use and mobility

### Fragmentation
Fragmentation measures how dispersed behavior is throughout the day. This is calculated using:
- Entropy-based fragmentation index (higher values = more fragmented)
- Episode counts and durations
- Coefficient of variation (CV) in episode durations

## Processing Pipeline

### 1. GPS Preprocessing (`preprocess-gps.py`)
Processes raw GPS and smartphone app data:
- Cleans and standardizes data formats
- Uses Trackintel library for geospatial processing
- Outputs preprocessed data for episode detection

### 2. Episode Detection (`detect_episodes.py`)
Identifies behavioral episodes:
- Detects digital episodes from screen on/off events
- Identifies mobility (trips) and stationary periods using Trackintel
- Finds overlap between digital and mobility episodes
- Saves daily timelines of all episodes

### 3. Fragmentation Analysis (`daily_fragmentation.py`)
Calculates fragmentation metrics for each day:
- Entropy-based or HHI-based fragmentation indices
- Handles edge cases like insufficient episodes
- Generates data visualizations and summary statistics

### 4. Window Fragmentation (`window_fragmentation.py`)
Links fragmentation metrics to EMA responses:
- Analyzes episodes in time windows preceding EMA responses
- Supports psychological research questions about behavior and mood
- Outputs EMA-linked fragmentation datasets

### 5. Visualization (`map-episodes.py`)
Creates interactive maps of detected episodes:
- Shows GPS trajectories color-coded by episode type
- Provides detailed episode information on click
- Supports multiple base layers and customization

### 6. Data Integration (`combine_metrics.py`)
Merges EMA psychological data with fragmentation metrics:
- Standardizes participant IDs across datasets
- Creates daily averages for psychological scales
- Produces integrated datasets for statistical analysis

## Setup & Usage

### Dependencies
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- trackintel (for geospatial processing)
- folium (for map visualizations)
- geopandas, shapely (for geographic data handling)

### Execution Order
1. Configure paths in `config/paths.py`
2. Preprocess GPS data: `python preprocess-gps.py`
3. Detect episodes: `python detect_episodes.py`
4. Calculate fragmentation: `python daily_fragmentation.py`
5. Process EMA windows: `python window_fragmentation.py`
6. Generate visualizations: `python map-episodes.py`
7. Integrate datasets: `python combine_metrics.py`