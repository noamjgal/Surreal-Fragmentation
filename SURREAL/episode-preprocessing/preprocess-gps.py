#!/usr/bin/env python3
"""
Enhanced preprocessing script for GPS tracking data using Trackintel
This script processes raw GPS data and app usage data and prepares it for episode detection
"""
import pandas as pd
import numpy as np
import os
import glob
import traceback
from datetime import datetime
from pathlib import Path
import sys
import logging
import warnings
import geopandas as gpd
from shapely.geometry import Point
import trackintel as ti

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, MAP_OUTPUT_DIR, GPS_PREP_DIR

# Ensure output directories exist
GPS_PREP_DIR.mkdir(parents=True, exist_ok=True)
MAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize quality report metrics
quality_report = {
    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'participants_processed': 0,
    'missing_files': [],
    'encoding_issues': [],
    'date_errors': [],
    'failed_processing': [],
    'successful_participants': []
}

def load_qstarz_data(file_path):
    """Load and preprocess Qstarz GPS data, returning a Trackintel Positionfixes object"""
    logging.info(f"Loading Qstarz data from {file_path}")
    
    try:
        # Load the raw data
        df = pd.read_csv(file_path)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Check for different possible column names
        datetime_col = next((col for col in df.columns if 'DATE TIME' in col), None)
        latitude_col = next((col for col in df.columns if 'LATITUDE' in col), None)
        longitude_col = next((col for col in df.columns if 'LONGITUDE' in col), None)
        
        if datetime_col is None or latitude_col is None or longitude_col is None:
            raise ValueError(f"Required columns not found in {file_path}")
        
        # Convert to Trackintel positionfixes format
        participant_id = os.path.basename(file_path).split('_')[0]
        
        # Parse timestamp
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Create positionfixes dataframe
        positionfixes = pd.DataFrame({
            'user_id': participant_id,
            'tracked_at': df[datetime_col],
            'latitude': df[latitude_col],
            'longitude': df[longitude_col],
            'elevation': np.nan,  # Optional
            'accuracy': np.nan,   # Optional
        })
        
        # Make sure tracked_at is timezone aware (required by trackintel)
        positionfixes['tracked_at'] = positionfixes['tracked_at'].dt.tz_localize('UTC', ambiguous='raise')
        
        # Convert to GeoDataFrame and set as trackintel Positionfixes
        geometry = [Point(lon, lat) for lon, lat in zip(positionfixes['longitude'], positionfixes['latitude'])]
        gdf = gpd.GeoDataFrame(positionfixes, geometry=geometry, crs="EPSG:4326")
        
        # Drop rows with missing or invalid GPS coordinates
        gdf = gdf.dropna(subset=['latitude', 'longitude'])
        gdf = gdf[(gdf['latitude'] != 0) & (gdf['longitude'] != 0)]
        
        # Set as trackintel Positionfixes
        pfs = ti.Positionfixes(gdf)
        
        logging.info(f"Successfully loaded {len(pfs)} Qstarz points for participant {participant_id}")
        return pfs
        
    except Exception as e:
        logging.error(f"Error loading Qstarz data: {str(e)}")
        traceback.print_exc()
        return None

def load_app_data(file_path):
    """Load and preprocess app usage data"""
    logging.info(f"Loading app data from {file_path}")
    
    try:
        # Try different encodings and delimiters
        encodings = ['utf-8', 'latin-1', 'utf-16']
        for encoding in encodings:
            try:
                # First try with semicolon delimiter
                app_df = pd.read_csv(file_path, encoding=encoding, delimiter=';')
                
                # Fallback to comma if only 1 column found
                if len(app_df.columns) == 1:
                    app_df = pd.read_csv(file_path, encoding=encoding, delimiter=',')
                break
            except UnicodeDecodeError:
                if encoding == encodings[-1]:  # Last encoding failed
                    raise
                continue
        
        # Clean up column names
        app_df.columns = app_df.columns.str.replace(';', '').str.strip()
        
        # Find date and time columns
        date_candidates = ['date', 'timestamp', 'datetime']
        time_candidates = ['time', 'timestamp', 'datetime']
        
        date_col = next((col for col in app_df.columns if col.lower() in date_candidates), None)
        time_col = next((col for col in app_df.columns if col.lower() in time_candidates and col.lower() != date_col.lower()), None)
        
        if not date_col or not time_col:
            raise ValueError(f"Could not identify date and time columns. Found columns: {app_df.columns.tolist()}")
        
        # Convert date and time to timestamp
        app_df['timestamp'] = pd.to_datetime(app_df[date_col] + ' ' + app_df[time_col], errors='coerce')
        
        # Drop rows with invalid timestamps
        app_df = app_df.dropna(subset=['timestamp'])
        
        # Make sure there's an action column (for screen events)
        action_candidates = ['action', 'event', 'screen']
        action_col = next((col for col in app_df.columns if any(cand in col.lower() for cand in action_candidates)), None)
        
        if action_col:
            app_df['action'] = app_df[action_col]
        else:
            logging.warning(f"No action column found in {file_path}")
            app_df['action'] = 'UNKNOWN'
            
        # Keep only essential columns
        app_df = app_df[['timestamp', 'action']]
        
        # Add date column
        app_df['date'] = app_df['timestamp'].dt.date
        
        logging.info(f"Successfully loaded {len(app_df)} app events")
        return app_df
        
    except Exception as e:
        logging.error(f"Error loading app data: {str(e)}")
        traceback.print_exc()
        return None

def process_participant(participant_id, qstarz_file, app_file):
    """Process data for a single participant"""
    logging.info(f"Processing data for participant {participant_id}")
    
    try:
        # Load Qstarz data as Trackintel Positionfixes
        positionfixes = load_qstarz_data(qstarz_file)
        
        if positionfixes is None or positionfixes.empty:
            logging.error(f"No valid positionfixes for participant {participant_id}")
            return False
            
        # Load app data
        app_df = load_app_data(app_file)
        
        if app_df is None or app_df.empty:
            logging.error(f"No valid app data for participant {participant_id}")
            return False
            
        # Save preprocessed files for episode detection
        qstarz_csv_path = GPS_PREP_DIR / f'{participant_id}_qstarz_prep.csv'
        app_csv_path = GPS_PREP_DIR / f'{participant_id}_app_prep.csv'
        
        # Save positionfixes
        positionfixes.to_csv(qstarz_csv_path)
        
        # Save app data
        app_df.to_csv(app_csv_path, index=False)
        
        logging.info(f"Saved preprocessed data for participant {participant_id}")
        quality_report['successful_participants'].append(participant_id)
        
        # Generate optional staypoints for visualization
        try:
            pfs, staypoints = positionfixes.generate_staypoints(
                method='sliding',
                dist_threshold=100,  # meters
                time_threshold=5.0,  # minutes
                gap_threshold=15.0   # minutes
            )
            
            # Save staypoints for debugging
            staypoints_path = GPS_PREP_DIR / f'{participant_id}_staypoints.csv'
            staypoints.to_csv(staypoints_path)
            logging.info(f"Generated {len(staypoints)} staypoints for participant {participant_id}")
        except Exception as e:
            logging.warning(f"Could not generate staypoints: {str(e)}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {str(e)}")
        traceback.print_exc()
        quality_report['failed_processing'].append(participant_id)
        return False

def generate_quality_report():
    """Generate a quality report for the preprocessing"""
    report_path = GPS_PREP_DIR / "preprocessing_quality_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Preprocessing Quality Report ({quality_report['start_time']})\n")
        f.write("="*50 + "\n")
        f.write(f"Participants successfully processed: {len(quality_report['successful_participants'])}\n")
        f.write(f"Participants with errors: {len(quality_report['failed_processing'])}\n")
        
        f.write("\nEncoding Issues:\n")
        for issue in quality_report['encoding_issues']:
            f.write(f"- Participant {issue['participant']}: {issue['error']}\n")
            
        f.write("\nDate Parsing Errors:\n")
        for error in quality_report['date_errors']:
            f.write(f"- File: {error['file']}\n  Error: {error['error']}\n")
            
        f.write("\nMissing Files:\n")
        for files in quality_report['missing_files']:
            f.write(f"- {files}\n")
            
        f.write("\nFailed Processing:\n")
        f.write(", ".join(quality_report['failed_processing']))
            
        f.write("\nSuccessful Participants:\n")
        f.write(", ".join(quality_report['successful_participants']))
    
    logging.info(f"Quality report saved to {report_path}")

def main():
    """Main processing function"""
    logging.info("Starting preprocessing of GPS and app data")
    
    # Get Qstarz files
    qstarz_files = {f.stem.split('_')[0]: f 
                   for f in Path(PROCESSED_DATA_DIR).glob('*_1_Qstarz_processed.csv')
                   if not f.stem.startswith('._')}
    
    logging.info(f"Found {len(qstarz_files)} Qstarz files")
    
    # Get app files
    app_files = {}
    for participant_folder in (RAW_DATA_DIR / "Participants").glob("P*"):
        participant_id = participant_folder.name.split('_')[-1]
        app_folder = participant_folder / '9 - Smartphone Tracking App'
        
        # Skip if folder doesn't exist
        if not app_folder.exists():
            logging.warning(f"App folder not found for participant {participant_id}")
            continue
            
        app_file = next(app_folder.glob('*-apps.csv'), None)
        
        if app_file:
            app_files[participant_id] = app_file
    
    logging.info(f"Found {len(app_files)} app files")
    
    # Find common participants
    common_participants = set(qstarz_files.keys()) & set(app_files.keys())
    logging.info(f"Found {len(common_participants)} participants with both Qstarz and app data")
    
    # Process each participant
    successful = 0
    for participant_id in common_participants:
        if process_participant(participant_id, qstarz_files[participant_id], app_files[participant_id]):
            successful += 1
    
    logging.info(f"Successfully processed {successful}/{len(common_participants)} participants")
    
    # Generate quality report
    generate_quality_report()
    
    return successful

if __name__ == "__main__":
    main()