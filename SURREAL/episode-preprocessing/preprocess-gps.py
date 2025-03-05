#!/usr/bin/env python3
"""
Enhanced GPS data preprocessing for both Qstarz and smartphone data
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
import geopandas as gpd
from shapely.geometry import Point
import trackintel as ti
import re
import pytz
from datetime import datetime
import sys

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'preprocessing.log'),
        logging.StreamHandler()
    ]
)

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, GPS_PREP_DIR

# Create output directories
GPS_PREP_DIR.mkdir(parents=True, exist_ok=True)
PROBLEM_FILES_DIR = GPS_PREP_DIR / "problem_files"
PROBLEM_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to track processing quality
quality_report = {
    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'participants_processed': 0,
    'successful_participants': [],
    'failed_processing': []
}

class GPSPreprocessor:
    """Handles preprocessing of GPS data from different sources"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
    def parse_timestamp(self, timestamp, formats=None):
        """Parse timestamps with flexible format detection"""
        if pd.isna(timestamp):
            return pd.NaT
            
        if isinstance(timestamp, pd.Timestamp):
            return timestamp
            
        # Default formats to try
        if formats is None:
            formats = [
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f',
                '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ'
            ]
            
        timestamp_str = str(timestamp).strip()
        
        # Try each format
        for fmt in formats:
            try:
                return pd.to_datetime(timestamp_str, format=fmt)
            except (ValueError, TypeError):
                continue
                
        # If none of the specific formats work, try pandas' flexible parser
        try:
            return pd.to_datetime(timestamp_str)
        except:
            return pd.NaT
            
    def fix_time_format(self, time_str):
        """Fix missing leading zeros in time strings"""
        if not isinstance(time_str, str):
            return time_str
            
        # Add leading zeros to seconds and minutes
        parts = time_str.split(':')
        if len(parts) == 3:
            hour, minute, second = parts
            if len(minute) == 1:
                minute = f"0{minute}"
            if len(second) == 1:
                second = f"0{second}"
            return f"{hour}:{minute}:{second}"
        return time_str
        
    def combine_date_time(self, date_val, time_val):
        """Combine date and time values into timestamp"""
        if pd.isna(date_val) or pd.isna(time_val):
            return pd.NaT
            
        # Convert to string and fix format
        date_str = str(date_val).strip()
        time_str = self.fix_time_format(str(time_val).strip())
        
        # Try to parse the combined string
        try:
            return pd.to_datetime(f"{date_str} {time_str}")
        except:
            # Try European format if default fails
            try:
                date_obj = pd.to_datetime(date_str, dayfirst=True)
                return pd.to_datetime(f"{date_obj.strftime('%Y-%m-%d')} {time_str}")
            except:
                return pd.NaT
                
    def ensure_tz_aware(self, dt_series, timezone='UTC'):
        """Ensure datetime series has timezone info"""
        if dt_series.empty:
            return dt_series
            
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(dt_series):
            dt_series = pd.to_datetime(dt_series, errors='coerce')
            
        # Add timezone if missing
        if dt_series.dt.tz is None:
            return dt_series.dt.tz_localize(timezone)
        return dt_series
        
    def detect_timezone_offset(self, df, utc_col, local_col):
        """Detect timezone offset between UTC and local time columns"""
        if utc_col not in df.columns or local_col not in df.columns:
            return None
        
        # Make sure both are datetime
        utc_time = pd.to_datetime(df[utc_col].iloc[0]) if not pd.api.types.is_datetime64_dtype(df[utc_col]) else df[utc_col].iloc[0]
        local_time = pd.to_datetime(df[local_col].iloc[0]) if not pd.api.types.is_datetime64_dtype(df[local_col]) else df[local_col].iloc[0]
        
        # Calculate offset in hours
        offset_seconds = (local_time - utc_time).total_seconds()
        offset_hours = round(offset_seconds / 3600)
        
        return offset_hours
        
    def process_qstarz_gps(self, file_path):
        """Process Qstarz GPS data"""
        if os.path.basename(file_path).startswith('._'):
            return None
            
        self.logger.info(f"Processing Qstarz data: {file_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    if encoding == 'ISO-8859-1':
                        raise
                    continue
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Find datetime and coordinate columns
            utc_datetime_col = next((col for col in df.columns if 'UTC DATE TIME' in col), None)
            local_datetime_col = next((col for col in df.columns if 'LOCAL DATE TIME' in col), None)
            datetime_col = utc_datetime_col or next((col for col in df.columns if 'DATE TIME' in col), None)
            lat_col = next((col for col in df.columns if 'LATITUDE' in col), None)
            lon_col = next((col for col in df.columns if 'LONGITUDE' in col), None)
            
            if not datetime_col or not lat_col or not lon_col:
                raise ValueError(f"Required columns not found in {file_path}")
                
            # Parse timestamps
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            
            # Detect timezone offset if local time is available
            tz_offset = None
            if local_datetime_col:
                df[local_datetime_col] = pd.to_datetime(df[local_datetime_col], errors='coerce')
                tz_offset = self.detect_timezone_offset(df, datetime_col, local_datetime_col)
                if tz_offset is not None:
                    self.logger.info(f"Detected timezone offset: UTC{'+' if tz_offset >= 0 else ''}{tz_offset}")
            
            # Extract participant ID from filename
            participant_id = os.path.basename(file_path).split('_')[0]
            
            # Create positionfixes dataframe
            positionfixes = pd.DataFrame({
                'user_id': participant_id,
                'tracked_at': df[datetime_col],
                'latitude': df[lat_col],
                'longitude': df[lon_col],
                'data_source': 'qstarz'
            })
            
            # Store timezone offset if available
            if tz_offset is not None:
                positionfixes['tz_offset'] = tz_offset
                
            # Ensure tracked_at is timezone aware
            positionfixes['tracked_at'] = self.ensure_tz_aware(positionfixes['tracked_at'])
            
            # Filter invalid points
            positionfixes = positionfixes.dropna(subset=['latitude', 'longitude'])
            positionfixes = positionfixes[(positionfixes['latitude'] != 0) & (positionfixes['longitude'] != 0)]
            
            self.logger.info(f"Processed {len(positionfixes)} Qstarz points for {participant_id}")
            return positionfixes
            
        except Exception as e:
            self.logger.error(f"Error processing Qstarz file {file_path}: {str(e)}")
            return None
            
    def process_smartphone_gps(self, file_path):
        """Process smartphone GPS data"""
        if os.path.basename(file_path).startswith('._'):
            return None
            
        self.logger.info(f"Processing smartphone GPS data: {file_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    if encoding == 'ISO-8859-1':
                        raise
                    continue
            
            # Find date, time and coordinate columns with flexible matching
            date_col = next((col for col in df.columns if col.lower() == 'date'), None)
            time_col = next((col for col in df.columns if col.lower() == 'time'), None)
            lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude']), None)
            lon_col = next((col for col in df.columns if col.lower() in ['lon', 'long', 'longitude']), None)
            
            if not date_col or not time_col or not lat_col or not lon_col:
                raise ValueError(f"Required columns not found in {file_path}")
            
            # Extract participant ID from filename
            participant_id = os.path.basename(file_path).split('-')[0]
            
            # Fix time format and combine with date
            df['fixed_time'] = df[time_col].astype(str).apply(self.fix_time_format)
            df['tracked_at'] = df.apply(lambda row: self.combine_date_time(row[date_col], row['fixed_time']), axis=1)
            
            # Create positionfixes dataframe
            positionfixes = pd.DataFrame({
                'user_id': participant_id,
                'tracked_at': df['tracked_at'],
                'latitude': df[lat_col],
                'longitude': df[lon_col],
                'data_source': 'smartphone'
            })
            
            # Ensure tracked_at is timezone aware
            positionfixes['tracked_at'] = self.ensure_tz_aware(positionfixes['tracked_at'])
            
            # Filter invalid points
            positionfixes = positionfixes.dropna(subset=['tracked_at', 'latitude', 'longitude'])
            positionfixes = positionfixes[(positionfixes['latitude'] != 0) & (positionfixes['longitude'] != 0)]
            
            self.logger.info(f"Processed {len(positionfixes)} smartphone GPS points for {participant_id}")
            return positionfixes
            
        except Exception as e:
            self.logger.error(f"Error processing smartphone GPS file {file_path}: {str(e)}")
            return None
            
    def process_app_data(self, file_path):
        """Process app usage data"""
        if os.path.basename(file_path).startswith('._'):
            return None
            
        self.logger.info(f"Processing app data: {file_path}")
        
        try:
            # Try different encodings and delimiters
            for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                try:
                    # Try semicolon delimiter first
                    app_df = pd.read_csv(file_path, encoding=encoding, delimiter=';')
                    
                    # If only one column, try comma
                    if len(app_df.columns) == 1 and ',' in app_df.iloc[0, 0]:
                        app_df = pd.read_csv(file_path, encoding=encoding, delimiter=',')
                        
                    # If still only one column, try tab
                    if len(app_df.columns) == 1 and '\t' in app_df.iloc[0, 0]:
                        app_df = pd.read_csv(file_path, encoding=encoding, delimiter='\t')
                        
                    break
                except:
                    if encoding == 'ISO-8859-1':
                        raise
                    continue
            
            # Clean column names
            app_df.columns = app_df.columns.str.replace(';', '').str.strip()
            
            # Find date and time columns
            date_col = next((col for col in app_df.columns if col.lower() in ['date', 'timestamp', 'datetime']), None)
            time_col = next((col for col in app_df.columns if col.lower() in ['time'] and (col.lower() != date_col.lower() if date_col else True)), None)
            
            if not date_col:
                raise ValueError(f"Date column not found in {file_path}")
                
            # Handle timestamp creation
            if time_col:
                # Fix time format with missing zeros
                app_df[time_col] = app_df[time_col].astype(str).apply(self.fix_time_format)
                
                # Create timestamp by combining date and time
                app_df['timestamp'] = app_df.apply(
                    lambda row: self.combine_date_time(row[date_col], row[time_col]), 
                    axis=1
                )
            else:
                # Single datetime column
                app_df['timestamp'] = pd.to_datetime(app_df[date_col], errors='coerce')
            
            # Filter out rows with invalid timestamps
            app_df = app_df.dropna(subset=['timestamp'])
            
            # Find action column for screen events
            action_col = next((col for col in app_df.columns if any(name in col.lower() for name in ['action', 'event', 'screen', 'status'])), None)
            
            if action_col:
                app_df['action'] = app_df[action_col]
            else:
                app_df['action'] = 'UNKNOWN'
                
            # Keep only essential columns
            app_df = app_df[['timestamp', 'action']]
            
            # Add date column
            app_df['date'] = app_df['timestamp'].dt.date
            
            self.logger.info(f"Processed {len(app_df)} app events")
            return app_df
            
        except Exception as e:
            self.logger.error(f"Error processing app data {file_path}: {str(e)}")
            return None
            
    def create_trackintel_positionfixes(self, gps_df):
        """Convert DataFrame to trackintel Positionfixes"""
        if gps_df is None or len(gps_df) == 0:
            return None
            
        # Create geometry points
        geometry = [Point(lon, lat) for lon, lat in zip(gps_df['longitude'], gps_df['latitude'])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(gps_df, geometry=geometry, crs="EPSG:4326")
        
        # Convert to trackintel Positionfixes
        return ti.Positionfixes(gdf)

def main():
    """Main processing function"""
    preprocessor = GPSPreprocessor()
    
    # Define Qstarz data path
    QSTARZ_DATA_DIR = Path("/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/qstarz")
    logging.info(f"Using Qstarz data from: {QSTARZ_DATA_DIR}")
    
    # Get Qstarz files (keeping original format with leading zeros)
    qstarz_files = {}
    for f in QSTARZ_DATA_DIR.glob('*_Qstarz_processed.csv'):
        if not os.path.basename(f).startswith('._'):
            participant_id = f.stem.split('_')[0]
            qstarz_files[participant_id] = f
    
    logging.info(f"Found {len(qstarz_files)} Qstarz files")
    
    # Get app files and smartphone GPS files
    app_files = {}
    app_gps_files = {}
    
    for participant_folder in (RAW_DATA_DIR / "Participants").glob("Pilot_*"):
        if os.path.basename(participant_folder).startswith('._'):
            continue
            
        # Extract participant ID
        full_participant_id = participant_folder.name
        participant_id = full_participant_id.split('_')[-1]
        
        app_folder = participant_folder / '9 - Smartphone Tracking App'
        
        if not app_folder.exists():
            continue
            
        # Look for app files with multiple possible patterns
        app_file = next((f for f in app_folder.glob(f'{participant_id.lstrip("0")}-apps.csv') 
                         if not os.path.basename(f).startswith('._')), None)
        if not app_file:
            app_file = next((f for f in app_folder.glob('*-apps.csv') 
                            if not os.path.basename(f).startswith('._')), None)
        
        # Look for smartphone GPS files
        app_gps_file = next((f for f in app_folder.glob(f'{participant_id.lstrip("0")}-gps.csv') 
                             if not os.path.basename(f).startswith('._')), None)
        if not app_gps_file:
            app_gps_file = next((f for f in app_folder.glob('*-gps.csv') 
                                if not os.path.basename(f).startswith('._')), None)
        
        # Store found files
        if app_file:
            app_files[participant_id] = app_file
            
        if app_gps_file:
            app_gps_files[participant_id] = app_gps_file
    
    logging.info(f"Found {len(app_files)} app files and {len(app_gps_files)} smartphone GPS files")
    
    # Participants with both types of data needed
    participants_with_gps = set(qstarz_files.keys()) | set(app_gps_files.keys())
    participants_with_app = set(app_files.keys())
    processable_participants = participants_with_gps & participants_with_app
    
    logging.info(f"Found {len(processable_participants)} processable participants")
    
    # Process each participant
    successful_count = 0
    
    for participant_id in processable_participants:
        # Skip macOS hidden files
        if participant_id.startswith('._'):
            continue
            
        logging.info(f"Processing participant {participant_id}")
        
        try:
            # Get the available data files
            qstarz_file = qstarz_files.get(participant_id)
            app_file = app_files.get(participant_id)
            app_gps_file = app_gps_files.get(participant_id)
            
            # Process GPS data (prefer Qstarz, fallback to smartphone)
            gps_data = None
            data_source = None
            
            if qstarz_file:
                gps_data = preprocessor.process_qstarz_gps(qstarz_file)
                if gps_data is not None and not gps_data.empty:
                    data_source = "qstarz"
            
            # Try smartphone GPS if Qstarz failed
            if (gps_data is None or gps_data.empty) and app_gps_file:
                gps_data = preprocessor.process_smartphone_gps(app_gps_file)
                if gps_data is not None and not gps_data.empty:
                    data_source = "smartphone"
            
            # Process app data
            app_data = None
            if app_file:
                app_data = preprocessor.process_app_data(app_file)
            
            # Skip if we're missing either GPS or app data
            if gps_data is None or gps_data.empty:
                logging.error(f"No valid GPS data for participant {participant_id}")
                quality_report['failed_processing'].append(f"{participant_id}: No valid GPS data")
                continue
                
            if app_data is None or app_data.empty:
                logging.error(f"No valid app data for participant {participant_id}")
                quality_report['failed_processing'].append(f"{participant_id}: No valid app data")
                continue
            
            # Extract any timezone information
            tz_offset = None
            if 'tz_offset' in gps_data.columns:
                tz_offset = gps_data['tz_offset'].iloc[0]
                gps_data = gps_data.drop(columns=['tz_offset'])
            
            # Add data source information
            gps_data['data_source'] = data_source
            
            # Save preprocessed data
            gps_csv_path = GPS_PREP_DIR / f'{participant_id}_gps_prep.csv'
            app_csv_path = GPS_PREP_DIR / f'{participant_id}_app_prep.csv'
            
            gps_data.to_csv(gps_csv_path, index=False)
            app_data.to_csv(app_csv_path, index=False)
            
            # Save timezone information separately if available
            if tz_offset is not None:
                tz_info_path = GPS_PREP_DIR / f'{participant_id}_timezone.txt'
                with open(tz_info_path, 'w') as f:
                    f.write(f"UTC{'+' if tz_offset >= 0 else ''}{tz_offset}")
                logging.info(f"Saved timezone info: UTC{'+' if tz_offset >= 0 else ''}{tz_offset}")
            
            logging.info(f"Saved preprocessed data for participant {participant_id} (source: {data_source})")
            quality_report['successful_participants'].append(participant_id)
            successful_count += 1
            
            # Generate optional staypoints for visualization
            try:
                positionfixes = preprocessor.create_trackintel_positionfixes(gps_data)
                if positionfixes is not None:
                    pfs, staypoints = positionfixes.generate_staypoints(
                        method='sliding',
                        dist_threshold=100,  # meters
                        time_threshold=5.0,  # minutes
                        gap_threshold=15.0   # minutes
                    )
                    
                    # Save staypoints for debugging
                    staypoints_path = GPS_PREP_DIR / f'{participant_id}_staypoints.csv'
                    staypoints.to_csv(staypoints_path)
                    logging.info(f"Generated {len(staypoints)} staypoints")
            except Exception as e:
                logging.warning(f"Could not generate staypoints: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error processing participant {participant_id}: {str(e)}")
            quality_report['failed_processing'].append(f"{participant_id}: {str(e)}")
    
    # Generate quality report
    quality_report['participants_processed'] = len(processable_participants)
    report_path = GPS_PREP_DIR / "preprocessing_quality_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Preprocessing Quality Report ({quality_report['start_time']})\n")
        f.write("="*50 + "\n")
        f.write(f"Participants processed: {quality_report['participants_processed']}\n")
        f.write(f"Participants successfully processed: {len(quality_report['successful_participants'])}\n")
        f.write(f"Participants with errors: {len(quality_report['failed_processing'])}\n\n")
        
        f.write("Failed Processing:\n")
        for reason in quality_report['failed_processing']:
            f.write(f"- {reason}\n")
            
        f.write("\nSuccessful Participants:\n")
        f.write(", ".join(quality_report['successful_participants']))
    
    logging.info(f"Quality report saved to {report_path}")
    logging.info(f"Successfully processed {successful_count}/{len(processable_participants)} participants")
    
    return successful_count

if __name__ == "__main__":
    main()