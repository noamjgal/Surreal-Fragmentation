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

def load_app_gps_data(file_path):
    """Load and preprocess smartphone GPS data, returning a Trackintel Positionfixes object"""
    logging.info(f"Loading smartphone GPS data from {file_path}")
    
    try:
        # Load the raw data
        df = pd.read_csv(file_path)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Expected column structure from sample
        # serial, id, date, time, long, lat, accuracy, provider
        
        # Check required columns are present
        required_cols = ['date', 'time', 'long', 'lat']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Required columns {missing_cols} not found in {file_path}")
        
        # Extract participant ID from filename
        participant_id = os.path.basename(file_path).split('-')[0]
        
        # Combine date and time into a timestamp
        df['tracked_at'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        
        # Create positionfixes dataframe
        positionfixes = pd.DataFrame({
            'user_id': participant_id,
            'tracked_at': df['tracked_at'],
            'latitude': df['lat'],
            'longitude': df['long'],
            'elevation': np.nan,
            'accuracy': df.get('accuracy', np.nan),  # Use accuracy if available
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
        
        logging.info(f"Successfully loaded {len(pfs)} smartphone GPS points for participant {participant_id}")
        return pfs
        
    except Exception as e:
        logging.error(f"Error loading smartphone GPS data: {str(e)}")
        traceback.print_exc()
        return None

def process_participant(participant_id, qstarz_file=None, app_file=None, app_gps_file=None):
    """Process data for a single participant"""
    logging.info(f"Processing data for participant {participant_id}")
    
    try:
        # Load Qstarz data as Trackintel Positionfixes if available
        positionfixes = None
        data_source = None
        
        if qstarz_file is not None:
            positionfixes = load_qstarz_data(qstarz_file)
            if positionfixes is not None and not positionfixes.empty:
                data_source = "qstarz"
                logging.info(f"Using Qstarz data for participant {participant_id}")
        
        # If Qstarz data is missing or invalid, try smartphone GPS data
        if (positionfixes is None or positionfixes.empty) and app_gps_file is not None:
            positionfixes = load_app_gps_data(app_gps_file)
            if positionfixes is not None and not positionfixes.empty:
                data_source = "smartphone"
                logging.info(f"Using smartphone GPS data for participant {participant_id}")
        
        # Check if we have valid GPS data from either source
        if positionfixes is None or positionfixes.empty:
            logging.error(f"No valid GPS data for participant {participant_id}")
            return False
            
        # Load app data
        if app_file is not None:
            app_df = load_app_data(app_file)
            if app_df is None or app_df.empty:
                logging.error(f"No valid app data for participant {participant_id}")
                return False
        else:
            logging.error(f"No app data file provided for participant {participant_id}")
            return False
            
        # Save preprocessed files for episode detection
        qstarz_csv_path = GPS_PREP_DIR / f'{participant_id}_gps_prep.csv'
        app_csv_path = GPS_PREP_DIR / f'{participant_id}_app_prep.csv'
        
        # Save positionfixes with data source metadata
        positionfixes['data_source'] = data_source
        positionfixes.to_csv(qstarz_csv_path)
        
        # Save app data
        app_df.to_csv(app_csv_path, index=False)
        
        logging.info(f"Saved preprocessed data for participant {participant_id} (GPS source: {data_source})")
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
    
    # Define Qstarz data path
    QSTARZ_DATA_DIR = Path("/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/qstarz")
    logging.info(f"Using Qstarz data from: {QSTARZ_DATA_DIR}")
    
    # Get Qstarz files with consistent ID format (keeping original format with leading zeros)
    qstarz_files = {}
    for f in QSTARZ_DATA_DIR.glob('*.csv'):
        if not f.stem.startswith('._'):
            # Extract ID and maintain original format (with leading zeros if present)
            participant_id = f.stem.split('_')[0]
            qstarz_files[participant_id] = f
    
    logging.info(f"Found {len(qstarz_files)} Qstarz files")
    logging.debug(f"Qstarz participant IDs: {sorted(qstarz_files.keys())}")
    
    # Get app files and smartphone GPS files
    app_files = {}
    app_gps_files = {}
    missing_app_folders = []
    missing_app_files = []
    
    for participant_folder in (RAW_DATA_DIR / "Participants").glob("Pilot_*"):
        # Extract participant ID correctly - keep leading zeros to match Qstarz files
        full_participant_id = participant_folder.name
        # Extract the part after "Pilot_" but keep leading zeros
        participant_id = full_participant_id.split('_')[-1]
        
        app_folder = participant_folder / '9 - Smartphone Tracking App'
        
        # Track missing folders
        if not app_folder.exists():
            logging.warning(f"App folder not found for participant {participant_id} (folder: {full_participant_id})")
            missing_app_folders.append(full_participant_id)
            continue
            
        # Look for app files with consistent naming
        app_file = next(app_folder.glob(f'{participant_id}-apps.csv'), None)
        
        # Look for smartphone GPS files
        app_gps_file = next(app_folder.glob(f'{participant_id}-gps.csv'), None)
        
        # Skip hidden files (those starting with ._)
        if app_file and not app_file.name.startswith('._'):
            app_files[participant_id] = app_file
        else:
            logging.warning(f"No app file found for participant {participant_id} (folder: {full_participant_id})")
            missing_app_files.append(full_participant_id)
        
        # Store GPS file if it exists
        if app_gps_file and not app_gps_file.name.startswith('._'):
            app_gps_files[participant_id] = app_gps_file
            logging.info(f"Found smartphone GPS data for participant {participant_id}")
    
    logging.info(f"Found {len(app_files)} app files")
    logging.info(f"Found {len(app_gps_files)} smartphone GPS files")
    
    # Find participants with data (now looking for either Qstarz OR smartphone GPS)
    participants_with_gps = set(qstarz_files.keys()) | set(app_gps_files.keys())
    participants_with_app = set(app_files.keys())
    
    # Participants with both types of data needed
    processable_participants = participants_with_gps & participants_with_app
    
    logging.info(f"Participants with some form of GPS data: {len(participants_with_gps)}")
    logging.info(f"Participants with app data: {len(participants_with_app)}")
    logging.info(f"Processable participants: {len(processable_participants)}")
    
    # Track what's missing
    missing_all_gps = set(app_files.keys()) - participants_with_gps
    missing_apps = participants_with_gps - set(app_files.keys())
    
    logging.info(f"Missing all GPS data for {len(missing_all_gps)} participants: {missing_all_gps}")
    logging.info(f"Missing app data for {len(missing_apps)} participants: {missing_apps}")
    
    # Update quality report with missing file information
    quality_report['missing_files'].extend([f"Missing app folder: {folder}" for folder in missing_app_folders])
    quality_report['missing_files'].extend([f"Missing app file: {folder}" for folder in missing_app_files])
    quality_report['missing_files'].extend([f"Missing all GPS data: Participant {p_id}" for p_id in missing_all_gps])
    quality_report['missing_files'].extend([f"Missing app data: Participant {p_id}" for p_id in missing_apps])
    
    # Process each participant
    successful = 0
    for participant_id in processable_participants:
        # Get the available data files
        qstarz_file = qstarz_files.get(participant_id)
        app_file = app_files.get(participant_id)
        app_gps_file = app_gps_files.get(participant_id)
        
        if process_participant(participant_id, qstarz_file, app_file, app_gps_file):
            successful += 1
    
    logging.info(f"Successfully processed {successful}/{len(processable_participants)} participants")
    
    # Generate quality report
    generate_quality_report()
    
    return successful

if __name__ == "__main__":
    main()

