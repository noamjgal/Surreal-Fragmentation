# SURREAL Episode Processing & Fragmentation Analysis Pipeline

A comprehensive pipeline for processing GPS and app usage data to detect behavioral episodes and analyze behavioral fragmentation patterns. These patterns are then associated with psychological data.

## Overview

This pipeline analyzes digital behavior and mobility patterns to detect episodes of activty and assess the level of  "fragmentation" - the degree to which activities are broken up into smaller, disconnected episodes throughout the day. It integrates data from:

- GPS tracking devices
- Smartphone app usage logs  
- Ecological Momentary Assessment (EMA) responses

## Core Concepts

### Data Standardization

To ensure consistency across the pipeline, all data processing follows standardized approaches:

- **Participant ID Standardization**: Consistent cleaning and normalization of IDs across datasets
- **Timestamp Handling**: Uniform datetime processing to avoid timezone and format inconsistencies
- **Missing Value Treatment**: Standardized approach to handling missing or invalid data
- **Data Validation**: Automatic validation of data ranges and integrity throughout processing

### Episodes

The project identifies three types of behavioral episodes:

- **Digital Episodes**: Periods of smartphone screen activity
- **Mobility Episodes**: Periods of physical movement between locations
- **Overlap Episodes**: Concurrent digital use and mobility

- Coefficient of variation (CV) in episode durations

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
5. Check quality reports generated during preprocessin
