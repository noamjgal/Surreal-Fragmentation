# SURREAL Episode Processing & Fragmentation Analysis Pipeline

A comprehensive pipeline for processing GPS and app usage data to detect behavioral episodes and analyze behavioral fragmentation patterns. These patterns are then associated with psychological data.

## Overview

This pipeline analyzes digital behavior and mobility patterns to detect episodes of activty and assess the level of  "fragmentation" - the degree to which activities are broken up into smaller, disconnected episodes throughout the day. It integrates data from:

- GPS tracking devices
- Smartphone app usage logs  
- Ecological Momentary Assessment (EMA) responses

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

### Data Standardization
To ensure consistency across the pipeline, all data processing follows standardized approaches:
- **Participant ID Standardization**: Consistent cleaning and normalization of IDs across datasets
- **Timestamp Handling**: Uniform datetime processing to avoid timezone and format inconsistencies
- **Missing Value Treatment**: Standardized approach to handling missing or invalid data
- **Data Validation**: Automatic validation of data ranges and integrity throughout processing

### Core Concepts

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

### Data Standardization
To ensure consistency across the pipeline, all data processing follows standardized approaches:
- **Participant ID Standardization**: Consistent cleaning and normalization of IDs across datasets
- **Timestamp Handling**: Uniform datetime processing to address timezone and format inconsistencies
- **Missing Value Treatment**: Standardized approach to handling missing or invalid data
- **Data Validation**: Automatic validation of data ranges and integrity throughout processing

## Processing Pipeline

### 1. Data Standardization (`data_utils.py`)
Provides utilities for consistent data handling:
- Standardizes participant IDs across different data sources
- Ensures consistent timestamp formats and timezone handling
- Applies uniform missing value treatment
- Validates data integrity with configurable validation rules
- Provides detailed logging for debugging data issues

### 2. GPS Preprocessing (`preprocess-gps.py`)
Processes both Qstarz and smartphone GPS data:
- Cleans and standardizes data formats
- Handles different data encodings and formats
- Uses Trackintel library for geospatial processing
- Generates quality reports for preprocessing steps
- Outputs preprocessed data for episode detection

### 3. Episode Detection (`detect_episodes.py`)
Identifies behavioral episodes:
- Detects digital episodes from screen on/off events
- Identifies mobility (trips) and stationary periods using Trackintel with fallback methods
- Finds overlap between digital and mobility episodes
- Saves daily timelines of all episodes
- Generates per-participant and overall summary statistics
- Uses standardized participant IDs for consistency

### 4. Fragmentation Analysis (`daily_fragmentation.py`)
Calculates fragmentation metrics for each day:
- Entropy-based or HHI-based fragmentation indices for both digital and mobility behaviors
- Handles edge cases like insufficient episodes
- Generates data visualizations and summary statistics
- Applies data validation to ensure metric quality
- Produces participant-level summaries and daily metrics

### 5. Data Integration (`combine_metrics.py` and `combine_metrics_raw.py`)
Merges EMA psychological data with fragmentation metrics:
- Uses standardized participant IDs for reliable merging
- Provides two approaches: standardized metrics and raw metrics
- Performs fuzzy date matching when exact matches fail
- Creates daily averages for psychological scales
- Provides detailed logging of match/mismatch statistics
- Applies data validation to ensure integrated dataset quality
- Produces integrated datasets for statistical analysis

### 6. Demographics Processing (`demographics.py`)
Processes and integrates demographic information:
- Standardizes demographic data formats
- Calculates derived variables (e.g., age from birth date)
- Merges demographics with combined datasets
- Handles different normalization types (raw/unstandardized, participant, population)
- Provides detailed reporting on data completeness and coverage

### 7. Visualization (`map-episodes.py`)
Creates interactive maps of detected episodes:
- Shows GPS trajectories color-coded by episode type
- Provides detailed episode information on click
- Handles timezone-aware and naive timestamps
- Generates daily episode maps with color-coded markers
- Includes detailed episode statistics
- Supports multiple base layers and customization

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
5. Integrate datasets: `python combine_metrics.py` or `python combine_metrics_raw.py`
6. Process demographics: `python demographics.py`
7. (Optional) Generate visualizations: `python map-episodes.py`

### Troubleshooting Data Issues
When encountering inconsistencies in data processing:
1. Check logs for validation warnings and errors
2. Verify participant ID standardization in outputs
3. Ensure timestamp formats are consistent across datasets
4. Review data validation rules in `data_utils.py` if needed
5. Check quality reports generated during preprocessing