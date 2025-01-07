# EMA Data Processing Scripts

This repository contains Python scripts for processing Ecological Momentary Assessment (EMA) data, focusing on standardizing survey responses and generating participant-level analyses.

## Required Inputs
project_root/
├── data/
│   └── raw/
│       ├── comprehensive_ema_data.csv             # Raw EMA survey data
│       └── processed_dictionaries_merged.csv      # Processed translation dictionaries
│       └── Corrected-Response-Mappings.xlsx       # Response mapping for recoding


## Processing Pipeline

1. **Variable Preprocessing** (`variable_preprocessing.py`)
   - Assigns variables to raw EMA data based on existing mappings
   - Maps Hebrew questions to standardized variable names
   - Input: Raw EMA data, Response mapping definitions
   - Output: EMA data with Variable column

2. **Response Recoding** (`EMA_response_recoding.py`)
   - Standardizes responses using existing mappings
   - Handles special cases (traffic, calm questions)
   - Processes reverse coding
   - Input: EMA data with variables, Response dictionaries
   - Output: Recoded survey responses

3. **Participant Processing** (`EMA_participant_processing.py`)
   - Processes individual participant data
   - Generates daily summaries
   - Validates response completeness
   - Input: Recoded survey responses
   - Output: Individual participant files, Daily summaries

4. **Scale Standardization** (`std.py`)
   - Standardizes STAI (1-4) and CES-D (1-5) responses
   - Handles reverse scoring for positive items:
     - STAI: CALM, PEACE, SATISFACTION
     - CES-D: HAPPY, ENJOYMENT_RECENT
   - Performs z-score standardization
   - Generates scale and variable-level summaries
   - Input: Individual participant files
   - Output: 
     - Normalized participant data
     - Scale summaries
     - Variable summaries

## Output Directory Structure

```
output/
├── daily_summary.csv                    # Daily survey completion summary
├── normalized/
│   ├── normalized_participant_{id}.csv  # Normalized participant data
│   ├── overall_summary_by_scale.csv     # Scale-level summaries
│   └── overall_summary_by_variable.csv  # Variable-level summaries
├── {scale}_summary.csv                  # Scale-specific summaries
├── valid_days_analysis.csv             # Analysis of valid response days
└── valid_days_boolean_summary.csv      # Boolean summary of valid days
```

## Scale Notes

### STAI-Y-A-6
- 4-point scale (1-4)
- Reverse scoring for positive items (CALM, PEACE, SATISFACTION)
- Z-score standardization applied

### CES-D-8
- 5-point scale (1-5)
- Reverse scoring for positive items (HAPPY, ENJOYMENT_RECENT)
- Z-score standardization applied

## Key Features

- Standardization of survey responses
- Scale-specific processing (STAI, CES-D, etc.)
- Participant-level data processing
- Daily and scale-based summaries
- Validation of response completeness

## Data Processing Pipeline

1. Response Recoding:
   - Standardize responses using existing mappings
   - Apply scale-specific coding rules
   - Handle special cases (traffic, calm questions)

2. Participant Processing:
   - Generate individual participant files
   - Create daily summaries
   - Validate response completeness


## Notes

- Test participants are automatically filtered out (IDs containing 'test')
- A minimum of 3 responses across STAI and CES-D scales is required for a valid day
- Special handling is implemented for procrastination questions to standardize responses
- Hebrew responses are automatically translated to English using predefined mappings