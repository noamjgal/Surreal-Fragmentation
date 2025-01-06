# Digital Technology & Mental Health Analysis Pipeline

## Overview
This analysis pipeline examines relationships between digital technology use, lifestyle fragmentation, and mental health outcomes using a comprehensive statistical approach that supports both cross-sectional and longitudinal analyses.

## Table of Contents
1. [Data Requirements](#data-requirements)
2. [Input Files](#input-files)
3. [Variable Transformations](#variable-transformations)
4. [Statistical Methods & Assumptions](#statistical-methods--assumptions)
5. [Output Files](#output-files)
6. [Analysis Components](#analysis-components)

## Data Requirements

### Minimum Requirements
- At least 3 participants
- At least 30 observations total
- For multilevel analyses: minimum of 2 observations per participant

### Required Variables
- participant_id: Unique identifier for each participant
- date: Date of observation
- digital_fragmentation_index: Measure of digital activity fragmentation
- moving_fragmentation_index: Measure of mobility fragmentation
- digital_frag_during_mobility: Combined digital-mobility fragmentation
- STAI6 items: TENSE, RELAXATION, WORRY, PEACE, IRRITATION, SATISFACTION

### Optional Control Variables
- Gender
- Class
- School_location
- total_duration_mobility
- avg_duration_mobility
- count_mobility
- is_weekend

## Input Files

### 1. Fragmentation Summary (CSV)
```
Location: fragmentation/fragmentation_summary.csv
Required columns:
- participant_id
- date
- digital_fragmentation_index
- moving_fragmentation_index
- digital_frag_during_mobility
```

### 2. Survey Responses (Excel)
```
Location: Survey/End_of_the_day_questionnaire.xlsx
Required columns:
- Participant_ID
- StartDate
- TENSE
- RELAXATION
- WORRY
- PEACE
- IRRITATION
- SATISFACTION
- HAPPY
```

### 3. Participant Info (CSV)
```
Location: participant_info.csv
Required columns:
- user (matches participant_id)
- Gender
- Class
- School_location
```

## Variable Transformations

### STAI6 Score Calculation
1. Reverse scoring for positive items:
   - RELAXATION_reversed = 5 - RELAXATION
   - PEACE_reversed = 5 - PEACE
   - SATISFACTION_reversed = 5 - SATISFACTION

2. STAI6 calculation:
   ```python
   STAI6_score = mean([TENSE, RELAXATION_reversed, WORRY, 
                       PEACE_reversed, IRRITATION, SATISFACTION_reversed]) * 20/6
   ```

### Standardized Variables
The following variables are z-standardized (mean=0, sd=1):
- total_duration_mobility_z
- avg_duration_mobility_z
- count_mobility_z
- days_with_data_z
- avg_fragmentation_z
- std_fragmentation_z

### Binary Variables
- Gender_binary: Mapped from Hebrew (נקבה=0, זכר=1)
- is_weekend: Derived from date (5,6 = 1; others = 0)

## Statistical Methods & Assumptions

### 1. Correlation Analysis
**Method**: Pearson and Spearman correlations
**Assumptions**:
- Pearson: Linear relationship, normality, homoscedasticity
- Spearman: Monotonic relationship
**Implementation**: Both methods used for robustness

### 2. Regression Analysis
**Method**: OLS with stepwise control variable selection
**Assumptions**:
- Linearity
- Independence of observations
- Homoscedasticity
- Normality of residuals
- No multicollinearity
**Control Selection**:
- Stepwise addition based on R² improvement
- Minimum improvement threshold: 0.01
- Maximum iterations: len(control_variables)

### 3. Multilevel Analysis
**Method**: Mixed Linear Models
**Assumptions**:
- Random effects normally distributed
- Independent residuals
- Homoscedasticity
**Fallback**: Robust regression if MLM fails to converge

### 4. Group Comparisons
**Method**: Welch's t-test (unequal variance) and ANCOVA
**Assumptions**:
- Normality within groups
- Independence of observations
**Effect Size**: Cohen's d for t-tests, η² for ANOVA

## Output Files

### Primary Results
1. mobility_analysis.csv
   - Correlations and regressions between fragmentation and mobility metrics

2. emotional_analysis.csv
   - Relationships between fragmentation indices and emotional outcomes

3. population_analysis.csv
   - Group differences in fragmentation patterns

4. usage_group_analysis.csv
   - Multilevel analysis of usage patterns and outcomes

### Supplementary Files
1. analysis_summary.csv
   - Overview of significant findings
   - Test counts and significance rates

2. *_significant_findings.csv
   - Filtered results with p < 0.05 for each analysis type

## Analysis Components

### DataPreprocessor
- Handles missing values
- Calculates derived variables
- Standardizes numeric features

### StatisticalAnalyzer
- Performs core statistical tests
- Implements control variable selection
- Calculates effect sizes

### MultilevelAnalyzer
- Handles repeated measures
- Implements robust fallbacks
- Manages longitudinal structure

### DigitalUsageProcessor
- Calculates usage metrics
- Creates user groups
- Processes temporal patterns

### ResultsManager
- Saves analysis outputs
- Generates summary statistics
- Organizes results by analysis type

## Usage Notes

1. Configure input/output paths in config.py
2. Ensure minimum data requirements are met
3. Run main.py to execute full analysis
4. Check analysis.log for execution details
5. Review output files for results

## Error Handling

1. Data Validation:
   - Checks minimum requirements
   - Validates variable presence
   - Verifies data types

2. Analysis Fallbacks:
   - MLM → Robust regression
   - Pearson → Spearman
   - Parametric → Non-parametric

3. Logging:
   - Execution progress
   - Warnings and errors
   - Analysis decisions
