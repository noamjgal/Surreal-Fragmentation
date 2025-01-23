# Fragmentation Study Codebase

## Overview
This codebase is designed to process and compare participant data from two studies, the SURREAL study of adult participants in Jerusalem, and the TLV study of adolescents in Tel Aviv. For each dataset, the codebase contains an ETL pipeline, activity fragmentation and clustering analyses, and hypothesis testing.

## Core Components

### 1. Main Processing (@SURREAL/main-processing)
Core data extraction and processing pipeline:
- GPS trajectory cleaning and validation
- Accelerometer data processing
- Environmental sensor integration (Svantek, Hygrometer, Photometer)
- EEG data processing (Dreem headband)
- DBSCAN clustering of Moments of Stress

### 2. EMA Processing (@SURREAL/EMA-Processing)
Advanced ecological momentary assessment pipeline:
- Multi-language response standardization (Hebrew/English)
- Robust missing data handling with validation
- Cross-scale response normalization

### 3. Fragmentation Analysis (@SURREAL/Fragmentation)
Sophisticated mobility pattern analysis:
- Episode detection and classification
- Temporal fragmentation metrics
- Spatial clustering with DBSCAN
- Interactive visualization of movement patterns
- Automated quality control and validation


## TLV Dataset Overview (@TLV)
Tel Aviv dataset specific components:
- Data Processing Pipeline:
  - Raw data conversion and standardization
  - GPS preprocessing with coverage analysis
  - Episode detection and fragmentation analysis
  - Comprehensive metrics calculation
- Hypothesis Testing:
  - Multilevel EMA effects analysis
  - Population comparisons
  - T-tests for emotional outcomes
- Visualization:
  - Interactive dashboards
  - Population difference visualizations
  - Coverage analysis reports


## Acknowledgments
- Research supported in partby the European Union's Horizon2020 under grant agreement No 956780
- Special thanks to coauthors Li Min Wang, Amnon Franco, Basille Chaix, and Amit Birenboim for their supervision and contributions.