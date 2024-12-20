#!/bin/bash

set -x  # This will print each command as it's executed
set -e  # This will cause the script to exit immediately if a command fails

# Change to the EMA directory
cd "$(dirname "$0")"

# Print current working directory
echo "Current working directory: $(pwd)"

# Check if the Python script exists
if [ ! -f "EMA_process.py" ]; then
    echo "Error: EMA_process.py not found"
    exit 1
fi
# Run the Python script with full error output
python EMA_process.py data/response_mapping_english_add.csv data/comprehensive_ema_data_eng.csv data/output 2>&1

echo "Script execution completed"
SURREAL/EMA-Round1/data/output/comprehensive_ema_data_eng_updated.csv
