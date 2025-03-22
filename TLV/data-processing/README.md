# Data Processing Pipeline Documentation

## Overview
The Data Processing pipeline analyzes GPS and digital activity data to study fragmentation patterns and their relationships with mobility, location, and emotional outcomes. The pipeline consists of modular scripts that handle different aspects of data processing, from raw data conversion to hypothesis testing preparation.

## Data Processing Pipeline

1. **Data Format Conversion** (`tocsv.py`)
   - Converts raw Excel files to CSV format
   - Handles GPS data and End-of-Day questionnaire responses
   - Performs initial data validation and logging
   - Maintains file modification timestamps for efficient processing

2. **Data Preprocessing** (`TLV-preprocess.py`)
   - Loads and standardizes GPS and app usage data
   - Combines date and time information
   - Creates participant info summaries
   - Identifies home locations based on location data
   - Handles early morning responses (before 5 AM)
   - Outputs preprocessed daily data files

3. **GPS Preprocessing** (`gps_preprocessing.py`)
   - Creates daily GPS summaries with quality metrics
   - Handles early morning responses and multiple responses per day
   - Processes home location flags for consistent identification
   - Generates comprehensive coverage analysis
   - Validates data quality (minimum 5 hours coverage)
   - Creates participant-level summaries
   - Outputs:
     - Preprocessed GPS files
     - Data quality summaries
     - Participant coverage statistics

4. **Episode Detection** (`episode-detection.py`)
   - Identifies multiple episode types:
     - Digital activity episodes
     - Mobility episodes
     - Home location episodes
     - Active transport episodes
     - Mechanized transport episodes
     - Digital-mobility overlap episodes
   - Processes episodes with configurable settings for each type
   - Calculates episode statistics, counts, and durations
   - Handles early morning data appropriately
   - Creates episode summaries with quality metrics
   - Outputs episode files per participant per day

5. **Fragmentation Analysis** (`TLV-fragment-3.0.py`)
   - Calculates fragmentation indices with improved validation:
     - Digital activity fragmentation
     - Movement pattern fragmentation
     - Digital activity during mobility
   - Handles outliers and invalid durations
   - Implements quality thresholds:
     - Minimum 3 episodes required
     - Maximum episode duration checks
     - Outlier detection (3 standard deviations)
   - Generates visualization plots for distributions
   - Outputs comprehensive fragmentation summaries

6. **Metrics Processing** (`metrics.py`)
   - Combines all processed data sources:
     - Fragmentation metrics
     - Episode statistics
     - EMA responses
     - GPS quality metrics
   - Calculates duration metrics:
     - Time spent at home (during waking hours)
     - Time spent in active transport
     - Time spent in mechanized transport
   - Calculates STAI-6 anxiety scores
   - Generates z-scores for key metrics
   - Creates weekday/weekend features
   - Outputs:
     - Combined metrics dataset
     - Summary statistics
     - Quality analysis reports

## Input Requirements

- Raw GPS and app usage data (Excel format)
- End-of-day questionnaire responses (Excel format)
- Participant demographic information
- Location data with home location indicators

## Output Structure

- **csv/**: Converted raw data files
- **preprocessed_data/**: Daily preprocessed files
- **preprocessed_summaries/**: 
  - GPS summaries
  - Data quality reports
  - Participant summaries
- **episodes/**: 
  - Episode detection results by type:
    - Digital episodes
    - Moving episodes
    - Home episodes
    - Active transport episodes
    - Mechanized transport episodes
    - Overlap episodes
  - Episode summary statistics
- **fragmentation/**: 
  - Fragmentation analysis results
  - Distribution plots
- **metrics/**: 
  - Combined metrics dataset with duration measures
  - Summary statistics
  - Analysis reports

## Quality Control

- Minimum 5 hours of GPS coverage required per day
- Early morning responses (before 5 AM) handled specially
- Multiple daily responses resolved using latest response
- Outlier detection in fragmentation calculations
- Comprehensive logging and error handling
- Data quality metrics and summaries generated at each stage
- Home location detection validated with detailed logging
