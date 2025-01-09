# Digital Fragmentation Analysis Pipeline

## Overview
This pipeline analyzes the relationship between digital usage patterns, mobility, and emotional states. It processes raw data from multiple sources and performs statistical analysis using regression and t-tests.


## Input Files
1. **Fragmentation Summary**
   - Path: `/fragmentation/fragmentation_summary.csv`
   - Contains: Digital activity and mobility metrics per participant per day
   - Key columns: participant_id, date, digital_fragmentation_index, moving_fragmentation_index, digital_frag_during_mobility

2. **Survey Responses**
   - Path: `/Survey/End_of_the_day_questionnaire.xlsx`
   - Contains: Daily emotional state measurements
   - Key columns: Participant_ID, StartDate, TENSE, RELAXATION, WORRY, PEACE, IRRITATION, SATISFACTION, HAPPY

3. **Participant Information**
   - Path: `/participant_info.csv`
   - Contains: Demographic and background information
   - Key columns: user (matches participant_id), Gender, Class, School, School_location

## Data Matching Logic
- Survey responses are matched with fragmentation metrics using:
  - Participant ID (matching between 'participant_id', 'Participant_ID', and 'user')
  - Date (matching 'date' from fragmentation data with 'StartDate' from survey)
- Only exact matches are kept (inner join)
- All dates are converted to date objects before matching

## Analysis Types

### Between-Participant Analysis
- Uses all available data points
- No minimum days requirement
- Includes:
  - Population differences (Gender, Class, School)
  - Digital usage group comparisons
  - Overall fragmentation effects

### Within-Participant Analysis
- Requires minimum 3 days of data per participant
- Used for:
  - Individual variability in fragmentation
  - Daily digital usage patterns
  - Digital usage group assignment

## File Structure
```
.
├── config.py              # Configuration and paths
├── data_loader.py         # Data loading and merging
├── preprocessor.py        # Data cleaning and feature creation
├── digital_usage_processor.py  # Digital usage metrics
├── tests.py              # Statistical analysis
└── main.py              # Pipeline runner
```

## Analysis Process

1. **Data Loading** (`data_loader.py`)
   - Loads fragmentation summary from CSV
   - Loads survey responses from Excel
   - Loads participant info from CSV
   - Merges datasets using participant ID and date

2. **Data Preprocessing** (`preprocessor.py`)
   - Handles missing values
   - Calculates STAI6 scores
   - Creates derived features (e.g., is_weekend)
   - Standardizes mobility variables
   - Creates binary variables (e.g., Gender)

3. **Digital Usage Processing** (`digital_usage_processor.py`)
   - Calculates user-level digital metrics
   - Creates usage groups (low/medium/high)
   - Adds normalized metrics

4. **Statistical Analysis** (`tests.py`)
   - Runs regressions on fragmentation indices
   - Performs t-tests on population factors
   - Only keeps significant control variables (p < 0.05)
   - Calculates effect sizes

## Key Variables

### Fragmentation Indices
- digital_fragmentation_index
- moving_fragmentation_index
- digital_frag_during_mobility

### Emotional Outcomes
- TENSE, RELAXATION, WORRY, PEACE
- IRRITATION, SATISFACTION
- STAI6_score (composite)
- HAPPY

### Population Factors
- Gender (נקבה/זכר)
- Class
- School
- digital_usage_group (low/medium/high)

## Output Files
Results are saved in the output directory:
- analysis_results.csv: All statistical tests
- significant_results.csv: Tests with p < 0.05

1. **Primary Results**
   - Path: `{output_dir}/analysis_results.csv`
   - Contains: All statistical tests performed
   - Fields: test_type, predictor, outcome, coefficient, std_error, p_value, effect_size, n

2. **Significant Findings**
   - Path: `{output_dir}/significant_results.csv`
   - Contains: Tests with p < 0.05
   - Same structure as analysis_results.csv

3. **Analysis Log**
   - Path: `{output_dir}/analysis.log`
   - Contains: Pipeline execution details, warnings, errors

4. **Digital Usage Groups**
   - Path: `{output_dir}/digital_usage_groups.csv`
   - Contains: User classifications and metrics
   - Only for participants with 3+ days of data

