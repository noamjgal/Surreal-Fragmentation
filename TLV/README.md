# TLV Pipeline Documentation

## Overview
The TLV pipeline processes GPS and digital activity data to analyze fragmentation patterns and their relationships with mobility and emotional outcomes. The pipeline consists of several interconnected Python scripts that handle different aspects of data processing and analysis.

## Pipeline Flow

1. **Data Preprocessing** (`TLV-preprocess.py`)
   - Loads raw GPS and app usage data
   - Cleans and standardizes timestamps
   - Creates participant info summaries
   - Outputs preprocessed daily data files

2. **GPS Preprocessing** (`gps_preprocessing.py`)
   - Creates daily summaries of GPS data
   - Adds demographic information
   - Generates quality metrics for GPS coverage
   - Outputs enhanced GPS summaries

3. **Episode Detection** (`episode-detection.py`)
   - Identifies digital activity and mobility episodes during waking hours (7AM-7PM)
   - Applies duration thresholds and gap merging
   - Creates separate episode files for digital and moving activities
   - Outputs episode summaries per participant per day

4. **Fragmentation Analysis** (`TLV-fragment-3.0.py`)
   - Calculates fragmentation indices for:
     - Digital activity
     - Movement patterns
     - Digital activity during mobility
   - Computes Average Inter-episode Duration (AID)
   - Generates comprehensive fragmentation summaries

5. **Coverage Analysis** (`coverage_analysis.py`)
   - Analyzes data completeness
   - Identifies gaps in data collection
   - Provides demographic breakdowns of coverage
   - Outputs coverage quality metrics

6. **Statistical Analysis** (`hypothesis-testing/significance-combined.py`)
   - Performs statistical tests on relationships between:
     - Fragmentation indices
     - Emotional outcomes
     - Demographic factors
   - Conducts multilevel analyses
   - Generates detailed statistical reports

## Input Requirements

- Raw GPS and app usage data (Excel format)
- End-of-day questionnaire responses
- Participant demographic information


## Output Structure

- **preprocessed_data/**: Daily preprocessed files
- **preprocessed_summaries/**: GPS and coverage summaries
- **episodes/**: Episode detection results
- **fragmentation/**: Fragmentation analysis results
- **data_coverage/**: Coverage analysis reports
- **analysis_results/**: Statistical analysis outputs 