# EMA Response Dictionary Construction Utilities

This folder contains the utilities used to construct and validate the response dictionaries and mappings used in the main EMA processing pipeline. These scripts document the process of creating standardized response mappings across different EMA versions and languages.

## Core Files

### EMA_dictionary_processing.py
Primary script for processing and standardizing response dictionaries. This script:
- Processes Hebrew and English response mappings
- Standardizes response values across different EMA versions
- Handles special cases like EFFORT questions
- Creates consistent numerical coding schemes

### utils.py
Contains core translation utilities, specifically:
- Hebrew to English translation mappings
- Standardized response term definitions
- Common phrase translations

## Dictionary Construction Process

1. **Initial Dictionary Creation**
   - Hebrew responses are mapped to standardized numerical codes
   - English translations are aligned with Hebrew responses
   - Special cases (EFFORT, PROCRASTINATION) are handled with specific mappings

2. **Standardization**
   - Response values are normalized across different EMA versions
   - Numerical codes are aligned for consistency
   - Special handling for reversed scales

3. **Validation**
   - Verification of complete mappings
   - Check for consistency across languages
   - Validation of special case handling

## Output Files

The dictionary construction process produces several key files used by the main pipeline:
- `processed_dictionaries.csv`: Core response mappings
- `processed_dictionaries_merged.csv`: Consolidated response dictionaries

## Usage Notes

These utilities were used to construct the initial response mappings and should be preserved for documentation and potential future dictionary updates.

## Special Cases

### EFFORT Questions
Special handling for effort-related questions across different EMA versions:
- EMA V1-V3: Standard 4-point scale
- EMA V4: Reversed 5-point scale

### PROCRASTINATION Questions
Standardized mapping for procrastination questions:
- Long form responses mapped to standard scale
- Consistent numerical coding across versions