#!/usr/bin/env python3
"""
Simplified GPS data preprocessing script with improved outlier detection
"""
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import logging
import traceback
from datetime import datetime, timedelta
import sys
from typing import Dict, List, Tuple, Optional

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

# Constants
ISRAEL_BOUNDS = {
    'min_lat': 29.0,  # Southern border
    'max_lat': 34.0,  # Northern border
    'min_lon': 33.0,  # Western border
    'max_lon': 36.0   # Eastern border
}
MAX_EXPECTED_COVERAGE = 22000  # Israel's area is ~22,000 sq km
MAX_REASONABLE_COVERAGE = 200  # Maximum reasonable daily coverage in sq km
STUDY_YEARS = [2023, 2024]  # Known data collection years

# Quality report tracking
quality_report = {
    'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'participants_processed': 0,
    'successful': 0,
    'failed': 0,
    'failed_reasons': [],
    'qstarz_only_days': 0,
    'smartphone_only_days': 0,
    'merged_days': 0,
    'coordinate_fixes': 0,
    'date_fixes': 0,
    'outliers_removed': 0
}


class GPSPreprocessor:
    """Special class for handling problematic GPS data"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def load_smartphone_gps(self, file_path):
        """Load GPS data with special encoding and format handling for P18"""
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
            
        # Try multiple encodings with error handling
        df = None
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings:
            try:
                self.logger.info(f"Trying to read file with encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                self.logger.info(f"Successfully read file with encoding: {encoding}")
                break
            except Exception as e:
                self.logger.warning(f"Failed to read with encoding {encoding}: {str(e)}")
                continue
        
        if df is None or df.empty:
            self.logger.error("Failed to read file with any encoding")
            return pd.DataFrame()
        
        # Clean column names and handle P18's specific format
        df.columns = df.columns.str.strip().str.lower()
        self.logger.info(f"Columns found: {df.columns.tolist()}")
        
        # Filter out any completely empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Create cleaned dataframe with proper column mapping
        try:
            # Fix time formatting (add leading zeros)
            if 'time' in df.columns:
                df['time'] = df['time'].astype(str).apply(
                    lambda t: re.sub(r'^(\d{1,2}):(\d{1}):(\d{1,2})$', r'\1:0\2:\3', t) if re.match(r'^\d{1,2}:\d{1}:\d{1,2}$', t) else
                             re.sub(r'^(\d{1,2}):(\d{1,2}):(\d{1})$', r'\1:\2:0\3', t) if re.match(r'^\d{1,2}:\d{1,2}:\d{1}$', t) else t
                )
            
            # Combine date and time for tracked_at
            cleaned_df = pd.DataFrame()
            
            if 'date' in df.columns and 'time' in df.columns:
                cleaned_df['tracked_at'] = pd.to_datetime(
                    df['date'].astype(str) + ' ' + df['time'].astype(str),
                    errors='coerce'
                )
            
            # Map columns correctly, handling P18's reversed lat/long
            if 'long' in df.columns and 'lat' in df.columns:
                cleaned_df['latitude'] = pd.to_numeric(df['lat'], errors='coerce')
                cleaned_df['longitude'] = pd.to_numeric(df['long'], errors='coerce')
            elif 'longitude' in df.columns and 'latitude' in df.columns:
                cleaned_df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                cleaned_df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Add accuracy if available
            if 'accuracy' in df.columns:
                cleaned_df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
            
            # Add provider if available
            if 'provider' in df.columns:
                cleaned_df['provider'] = df['provider']
            
            # Add participant ID and source
            cleaned_df['user_id'] = '18'
            cleaned_df['data_source'] = 'smartphone'
            
            # Add date column
            cleaned_df['date'] = cleaned_df['tracked_at'].dt.date
            
            # Remove NaN values
            cleaned_df = cleaned_df.dropna(subset=['tracked_at', 'latitude', 'longitude'])
            
            self.logger.info(f"Successfully cleaned {len(cleaned_df)} GPS points")
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning P18 data: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def clean_coordinates(self, df):
        """Special coordinate cleaning for P18"""
        if df.empty:
            return df
            
        self.logger.info(f"Cleaning coordinates for {len(df)} points")
        
        # 1. Remove rows with NaN coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # 2. Remove rows with zero coordinates
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
        
        # 3. Check coordinates are within Israel bounds
        in_bounds = (
            (df['latitude'] >= ISRAEL_BOUNDS['min_lat']) & 
            (df['latitude'] <= ISRAEL_BOUNDS['max_lat']) & 
            (df['longitude'] >= ISRAEL_BOUNDS['min_lon']) & 
            (df['longitude'] <= ISRAEL_BOUNDS['max_lon'])
        )
        
        # Check if we need to swap coordinates
        out_of_bounds = df[~in_bounds]
        if len(out_of_bounds) > len(df) * 0.5:  # More than half out of bounds
            self.logger.warning("Most coordinates out of bounds, trying lat/long swap")
            
            # Try swapping lat/long
            df_swapped = df.copy()
            df_swapped['temp'] = df_swapped['latitude']
            df_swapped['latitude'] = df_swapped['longitude']
            df_swapped['longitude'] = df_swapped['temp']
            df_swapped = df_swapped.drop(columns=['temp'])
            
            # Check if swap improved things
            swapped_in_bounds = (
                (df_swapped['latitude'] >= ISRAEL_BOUNDS['min_lat']) & 
                (df_swapped['latitude'] <= ISRAEL_BOUNDS['max_lat']) & 
                (df_swapped['longitude'] >= ISRAEL_BOUNDS['min_lon']) & 
                (df_swapped['longitude'] <= ISRAEL_BOUNDS['max_lon'])
            )
            
            if swapped_in_bounds.mean() > in_bounds.mean():
                self.logger.info("Coordinate swap improved in-bounds ratio, using swapped coordinates")
                df = df_swapped
                in_bounds = swapped_in_bounds
        
        # 4. Filter to in-bounds coordinates
        df = df[in_bounds]
        
        # 5. Add timezone if not present
        if hasattr(df['tracked_at'].dt, 'tz') and df['tracked_at'].dt.tz is None:
            df['tracked_at'] = df['tracked_at'].dt.tz_localize('UTC')
        
        # 6. Sort by tracked_at
        df = df.sort_values('tracked_at')
        
        self.logger.info(f"Cleaning complete, {len(df)} valid points remaining")
        return df

# --------------- UTILITY FUNCTIONS ---------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between coordinates in meters"""
    R = 6371000  # Earth radius in meters
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def fix_smartphone_dates(df):
    """Fix common smartphone date issues"""
    if df is None or df.empty or 'tracked_at' not in df.columns:
        return df
    
    df = df.copy()
    original_dates = df['tracked_at'].copy()
    fixed_count = 0
    current_date = datetime.now().date()
    
    # 1. Fix the "2nd of month" pattern
    day_as_month_pattern = df['tracked_at'].dt.day == 2
    month_gt_2_pattern = df['tracked_at'].dt.month > 2
    future_date_pattern = df['tracked_at'].dt.date > current_date
    
    pattern_mask = day_as_month_pattern & month_gt_2_pattern & future_date_pattern
    if pattern_mask.any():
        df.loc[pattern_mask, 'tracked_at'] = df.loc[pattern_mask, 'tracked_at'].apply(
            lambda dt: dt.replace(month=2, day=dt.month)
        )
        fixed_pattern = pattern_mask.sum()
        if fixed_pattern > 0:
            logging.info(f"Fixed {fixed_pattern} dates with day-month confusion (day=2 pattern)")
            fixed_count += fixed_pattern
    
    # 2. Fix any remaining future dates by adjusting year
    future_date_mask = df['tracked_at'].dt.date > current_date
    if future_date_mask.any():
        future_years = df.loc[future_date_mask, 'tracked_at'].dt.year
        if not future_years.empty:
            most_common_year = future_years.value_counts().index[0]
            target_year = 2024 if most_common_year > 2024 else most_common_year - 1
            
            df.loc[future_date_mask, 'tracked_at'] = df.loc[future_date_mask, 'tracked_at'].apply(
                lambda dt: dt.replace(year=target_year)
            )
            fixed_future = future_date_mask.sum()
            logging.info(f"Fixed {fixed_future} future dates by setting year to {target_year}")
            fixed_count += fixed_future
    
    # Update date column if fixes were made
    if fixed_count > 0:
        df['date'] = df['tracked_at'].dt.date
        quality_report['date_fixes'] += fixed_count
        
        # Log a sample of fixes
        if fixed_count > 0 and len(original_dates) > 0:
            sample_idx = original_dates[pattern_mask | future_date_mask].index[:3]
            for idx in sample_idx:
                if idx in df.index:
                    logging.info(f"Date fix example: {original_dates[idx]} → {df.loc[idx, 'tracked_at']}")
    
    return df


def clean_coordinates(points, source='unknown'):
    """
    Clean and validate GPS coordinates with improved outlier detection
    """
    if points is None or points.empty:
        return points
    
    df = points.copy()
    original_len = len(df)
    
    # Log initial coordinate ranges
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    logging.info(f"{source} initial coordinate range: Lat [{lat_min:.6f}, {lat_max:.6f}], "
                f"Lon [{lon_min:.6f}, {lon_max:.6f}]")
    
    # 1. Basic cleaning - remove NaN and zeros
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
    
    # 2. Remove extreme outliers (beyond Earth's coordinates)
    earth_mask = (df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180))
    extreme_outliers = (~earth_mask).sum()
    if extreme_outliers > 0:
        # Log sample of extreme outliers for debugging
        extreme_sample = df[~earth_mask].head(3)
        logging.warning(f"Removed {extreme_outliers} impossible coordinates beyond Earth's bounds")
        logging.warning(f"Sample extreme outliers: \n{extreme_sample[['latitude', 'longitude']].to_string()}")
        df = df[earth_mask]
    
    # 3. Check if coordinates are within Israel bounds
    in_bounds = (
        (df['latitude'] >= ISRAEL_BOUNDS['min_lat']) & 
        (df['latitude'] <= ISRAEL_BOUNDS['max_lat']) & 
        (df['longitude'] >= ISRAEL_BOUNDS['min_lon']) & 
        (df['longitude'] <= ISRAEL_BOUNDS['max_lon'])
    )
    
    # Handle out-of-bounds points
    out_of_bounds = df[~in_bounds].copy()
    if not out_of_bounds.empty:
        # Try lat/long swap fix for out-of-bounds points
        swapped_df = out_of_bounds.copy()
        swapped_df['latitude'], swapped_df['longitude'] = swapped_df['longitude'], swapped_df['latitude']
        
        # Check if swapped coordinates are within bounds
        swapped_in_bounds = (
            (swapped_df['latitude'] >= ISRAEL_BOUNDS['min_lat']) & 
            (swapped_df['latitude'] <= ISRAEL_BOUNDS['max_lat']) & 
            (swapped_df['longitude'] >= ISRAEL_BOUNDS['min_lon']) & 
            (swapped_df['longitude'] <= ISRAEL_BOUNDS['max_lon'])
        )
        
        fixed_by_swap = swapped_in_bounds.sum()
        if fixed_by_swap > 0:
            logging.info(f"Fixed {fixed_by_swap} coordinates by swapping lat/long")
            quality_report['coordinate_fixes'] += fixed_by_swap
            
            # Log sample of fixed coordinates
            if fixed_by_swap > 0:
                sample_fixed = swapped_df[swapped_in_bounds].head(3)
                sample_original = out_of_bounds.loc[sample_fixed.index]
                for i, (_, orig) in enumerate(sample_original.iterrows()):
                    fixed = sample_fixed.iloc[i]
                    logging.info(f"Coordinate swap example: ({orig['latitude']:.6f}, {orig['longitude']:.6f}) → "
                                f"({fixed['latitude']:.6f}, {fixed['longitude']:.6f})")
            
            # Combine in-bounds original points with fixed swapped points
            in_bounds_points = df[in_bounds]
            fixed_points = swapped_df[swapped_in_bounds]
            df = pd.concat([in_bounds_points, fixed_points], ignore_index=True)
        else:
            # Log sample of removed out-of-bounds coordinates
            sample_removed = out_of_bounds.head(3)
            logging.warning(f"Removed {len(out_of_bounds)} coordinates outside Israel's bounds")
            logging.warning(f"Sample removed out-of-bounds points: \n{sample_removed[['latitude', 'longitude']].to_string()}")
            df = df[in_bounds]
    else:
        df = df[in_bounds]
    
    # 4. Improved outlier detection with adaptive thresholding
    if len(df) >= 10:
        # Calculate median location
        median_lat = df['latitude'].median()
        median_lon = df['longitude'].median()
        
        # Calculate distances from median location (in kilometers)
        df['dist_from_median'] = df.apply(
            lambda row: haversine_distance(median_lat, median_lon, row['latitude'], row['longitude'])/1000,
            axis=1
        )
        
        # Calculate IQR statistics
        q1 = df['dist_from_median'].quantile(0.25)
        q3 = df['dist_from_median'].quantile(0.75)
        iqr = q3 - q1
        
        # Use adaptive threshold with minimum distance guarantee
        # Ensure threshold is at least 1km to allow for normal movement patterns
        min_threshold = 1.0  # Minimum 1km threshold
        calculated_threshold = q3 + (2.0 * iqr)
        upper_bound = max(calculated_threshold, min_threshold)
        
        # For large datasets, use percentile-based approach as fallback
        if len(df) > 1000:
            percentile_threshold = df['dist_from_median'].quantile(0.95)  # Keep 95% of points
            upper_bound = max(upper_bound, percentile_threshold)
        
        # Identify outliers
        outliers = df['dist_from_median'] > upper_bound
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            # Log sample of detected outliers
            outlier_sample = df[outliers].sort_values('dist_from_median', ascending=False).head(3)
            logging.warning(f"Removed {outlier_count} spatial outliers (>{upper_bound:.1f}km from median)")
            logging.warning(f"Sample outliers: \n{outlier_sample[['latitude', 'longitude', 'dist_from_median']].to_string()}")
            
            # Special case: check if outliers form a separate valid cluster
            # (in case the dataset has multiple valid locations)
            if outlier_count >= 10 and outlier_count <= len(df) * 0.4:  # Increased from 0.3
                outlier_median_lat = df.loc[outliers, 'latitude'].median()
                outlier_median_lon = df.loc[outliers, 'longitude'].median()
                
                # Check if outlier cluster is within Israel
                if (ISRAEL_BOUNDS['min_lat'] <= outlier_median_lat <= ISRAEL_BOUNDS['max_lat'] and
                    ISRAEL_BOUNDS['min_lon'] <= outlier_median_lon <= ISRAEL_BOUNDS['max_lon']):
                    
                    # Calculate spread of outlier cluster
                    outlier_df = df[outliers]
                    lat_spread = outlier_df['latitude'].max() - outlier_df['latitude'].min()
                    lon_spread = outlier_df['longitude'].max() - outlier_df['longitude'].min()
                    
                    # If the outlier cluster is compact, it might be valid
                    if lat_spread < 0.5 and lon_spread < 0.5:
                        logging.info("Detected potential secondary location cluster - keeping these points")
                        outliers = pd.Series(False, index=df.index)
                        outlier_count = 0
            
            # Remove outliers from dataset
            df = df[~outliers]
            quality_report['outliers_removed'] += outlier_count
        
        # Drop distance column
        df = df.drop(columns=['dist_from_median'])
    
    # Log final coordinate ranges and changes
    if not df.empty:
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        logging.info(f"{source} final coordinate range: Lat [{lat_min:.6f}, {lat_max:.6f}], "
                    f"Lon [{lon_min:.6f}, {lon_max:.6f}]")
    
    # Log summary
    points_removed = original_len - len(df)
    if points_removed > 0:
        logging.info(f"Total coordinates filtered from {source}: {points_removed} ({points_removed/original_len:.1%})")
    
    return df


def filter_speed_outliers(points, max_speed_kph=150):
    """Remove points with unrealistic speeds between them"""
    if points is None or len(points) < 2:
        return points
    
    df = points.copy()
    original_len = len(df)
    
    try:
        # Sort by timestamp and reset index to avoid index-related errors
        df = df.sort_values('tracked_at').reset_index(drop=True)
        
        # Add previous point coordinates for distance calculation
        df['prev_lat'] = df['latitude'].shift(1)
        df['prev_lon'] = df['longitude'].shift(1)
        
        # Calculate time differences 
        df['time_diff'] = (df['tracked_at'] - df['tracked_at'].shift(1)).dt.total_seconds()
        valid_time = df['time_diff'] > 0  # Skip first point and negative time differences
        
        if valid_time.any():
            # Calculate distances between consecutive points using vectorized operation
            df.loc[valid_time, 'distance'] = df[valid_time].apply(
                lambda row: haversine_distance(
                    row['prev_lat'], row['prev_lon'],
                    row['latitude'], row['longitude']
                ), 
                axis=1
            )
            
            # Calculate speeds (m/s) and convert to km/h
            df.loc[valid_time, 'speed'] = df.loc[valid_time, 'distance'] / df.loc[valid_time, 'time_diff']
            df.loc[valid_time, 'speed_kmh'] = df.loc[valid_time, 'speed'] * 3.6
            
            # Filter by speed
            outliers = (df['speed_kmh'] > max_speed_kph) & valid_time
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                source = df['data_source'].iloc[0] if 'data_source' in df.columns else 'unknown'
                
                # Log sample of speed outliers
                if outlier_count > 0:
                    sample_outliers = df[outliers].sort_values('speed_kmh', ascending=False).head(3)
                    logging.info(f"Sample speed outliers: \n{sample_outliers[['tracked_at', 'latitude', 'longitude', 'speed_kmh']].to_string()}")
                
                logging.info(f"Removed {outlier_count} {source} points with speed > {max_speed_kph} km/h")
                df = df[~outliers]
        
        # Clean up temporary columns
        cols_to_drop = ['time_diff', 'distance', 'speed', 'speed_kmh', 'prev_lat', 'prev_lon']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
    except Exception as e:
        logging.warning(f"Error during speed outlier detection: {str(e)}")
        logging.warning("Skipping speed outlier filtering")
        return points
    
    # Log summary
    points_removed = original_len - len(df)
    if points_removed > 0:
        logging.info(f"Removed {points_removed} speed outliers ({points_removed/original_len:.1%} of data)")
    
    return df


# --------------- DATA LOADING FUNCTIONS ---------------

def read_qstarz_data(file_path):
    """Read and process Qstarz GPS data"""
    if not os.path.exists(file_path) or os.path.basename(file_path).startswith('._'):
        return None
        
    logging.info(f"Processing Qstarz data: {file_path}")
    
    try:
        # Read the CSV file with different encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except Exception:
                continue
        
        if df is None or df.empty:
            logging.error(f"Failed to read Qstarz file: {file_path}")
            return None
        
        # Extract participant ID from filename
        participant_id = os.path.basename(file_path).split('_')[0]
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Find key columns using patterns
        datetime_col = next((col for col in df.columns if 'UTC DATE TIME' in col.upper()), 
                           next((col for col in df.columns if 'DATE TIME' in col.upper()), None))
        
        lat_col = next((col for col in df.columns if 'LATITUDE' in col.upper()), None)
        lon_col = next((col for col in df.columns if 'LONGITUDE' in col.upper()), None)
        
        if not all([datetime_col, lat_col, lon_col]):
            logging.error(f"Required columns not found in {file_path}")
            return None
        
        # Check data types and log sample data
        logging.info(f"Sample data - Datetime: {df[datetime_col].head(1).iloc[0]}, " +
                    f"Lat: {df[lat_col].head(1).iloc[0]}, Lon: {df[lon_col].head(1).iloc[0]}")
        
        # Parse timestamps
        try:
            df['tracked_at'] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            logging.error(f"Error parsing datetime: {str(e)}")
            return None
        
        # Create output dataframe with explicit type conversion
        try:
            gps_data = pd.DataFrame({
                'user_id': participant_id,
                'tracked_at': df['tracked_at'],
                'latitude': pd.to_numeric(df[lat_col], errors='coerce'),
                'longitude': pd.to_numeric(df[lon_col], errors='coerce'),
                'data_source': 'qstarz'
            })
            
            # Log sample of converted data
            sample = gps_data.head(1)
            logging.info(f"Converted data sample: Lat={sample['latitude'].iloc[0]}, Lon={sample['longitude'].iloc[0]}")
            
        except Exception as e:
            logging.error(f"Error converting coordinate data: {str(e)}")
            return None
        
        # Add date column
        gps_data['date'] = gps_data['tracked_at'].dt.date
        
        # Add timezone if missing
        if gps_data['tracked_at'].dt.tz is None:
            gps_data['tracked_at'] = gps_data['tracked_at'].dt.tz_localize('UTC')
        
        # Clean coordinates and filter speed outliers
        gps_data = clean_coordinates(gps_data, source='qstarz')
        gps_data = filter_speed_outliers(gps_data, max_speed_kph=150)
        
        if not gps_data.empty:
            logging.info(f"Processed {len(gps_data)} Qstarz points for {participant_id}")
            logging.info(f"Processed {gps_data['date'].nunique()} days of Qstarz data")
        else:
            logging.warning(f"No valid Qstarz data after cleaning for {participant_id}")
            
        return gps_data
        
    except Exception as e:
        logging.error(f"Error processing Qstarz file {file_path}: {str(e)}")
        traceback.print_exc()
        return None


def read_smartphone_data(file_path):
    """Read and process smartphone GPS data with enhanced date handling and flexible column formats"""
    if not os.path.exists(file_path) or os.path.basename(file_path).startswith('._'):
        return None
        
    logging.info(f"Processing smartphone GPS data: {file_path}")
    
    try:
        # Read the file with different approaches
        df = None
        
        # First try with automatic delimiter detection
        try:
            df = pd.read_csv(file_path, engine='python', sep=None)
        except Exception as e:
            logging.warning(f"First attempt to read {file_path} failed: {str(e)}")
            
            # Try with common delimiters
            for sep in [',', '\t', ';']:
                for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        if df is not None and not df.empty:
                            logging.info(f"Successfully read file with delimiter '{sep}' and encoding '{encoding}'")
                            break
                    except Exception:
                        continue
                if df is not None and not df.empty:
                    break
        
        if df is None or df.empty:
            logging.error(f"Failed to read smartphone GPS file: {file_path}")
            return None
        
        # Extract participant ID from filename
        participant_id = os.path.basename(file_path).split('-')[0]
        
        # Log all columns for debugging
        logging.info(f"Available columns: {df.columns.tolist()}")
        
        # Find required columns with more flexible matching
        date_col = next((col for col in df.columns if col.lower() in ['date', 'ddate', 'datetime']), None)
        time_col = next((col for col in df.columns if col.lower() in ['time', 'ttime', 'timestamp']), None)
        lat_col = next((col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude', 'lati'])), None)
        lon_col = next((col for col in df.columns if any(term in col.lower() for term in ['lon', 'long', 'longitude', 'lng'])), None)
        
        # Log found columns
        logging.info(f"Identified columns - Date: {date_col}, Time: {time_col}, Lat: {lat_col}, Lon: {lon_col}")
        
        if not all([date_col, time_col, lat_col, lon_col]):
            logging.error(f"Required columns missing in {file_path}")
            return None
        
        # Log sample values for debugging
        logging.info(f"Sample data - Date: {df[date_col].head(1).iloc[0]}, " +
                    f"Time: {df[time_col].head(1).iloc[0]}, " +
                    f"Lat: {df[lat_col].head(1).iloc[0]}, Lon: {df[lon_col].head(1).iloc[0]}")
        
        # Fix time formats - ensure HH:MM:SS format with leading zeros
        df['fixed_time'] = df[time_col].astype(str).apply(
            lambda t: re.sub(r'^(\d):(\d\d?):(\d\d?)$', r'0\1:\2:\3', t) if re.match(r'^\d:\d\d?:\d\d?$', t) else 
                      re.sub(r'^(\d\d?):(\d):(\d\d?)$', r'\1:0\2:\3', t) if re.match(r'^\d\d?:\d:\d\d?$', t) else
                      re.sub(r'^(\d\d?):(\d\d?):(\d)$', r'\1:\2:0\3', t) if re.match(r'^\d\d?:\d\d?:\d$', t) else 
                      t
        )
        
        # Identify date format
        sample_date = str(df[date_col].iloc[0]) if not df.empty else ""
        date_format = None
        
        if re.match(r'\d{4}-\d{2}-\d{2}', sample_date):
            date_format = '%Y-%m-%d'
            logging.info("Detected YYYY-MM-DD date format")
        elif re.match(r'\d{2}/\d{2}/\d{4}', sample_date):
            date_format = '%d/%m/%Y'  # Assuming Israel format (day first)
            logging.info("Detected DD/MM/YYYY date format")
        elif re.match(r'\d{2}-\d{2}-\d{4}', sample_date):
            date_format = '%d-%m-%Y'  # Assuming Israel format
            logging.info("Detected DD-MM-YYYY date format")
        
        # Parse dates with appropriate format
        df['tracked_at'] = None
        try:
            if date_format:
                # Use detected format
                df['datetime_str'] = df[date_col].astype(str) + ' ' + df['fixed_time']
                df['tracked_at'] = pd.to_datetime(df['datetime_str'], format=f"{date_format} %H:%M:%S", errors='coerce')
            else:
                # Fall back to default parsing (day first for Israel data)
                df['datetime_str'] = df[date_col].astype(str) + ' ' + df['fixed_time']
                df['tracked_at'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
        except Exception as e:
            logging.warning(f"Datetime parsing error: {str(e)}, trying with dayfirst=True")
            # Fallback parsing
            df['datetime_str'] = df[date_col].astype(str) + ' ' + df['fixed_time']
            df['tracked_at'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
        
        # Check for NaT values
        nat_count = df['tracked_at'].isna().sum()
        if nat_count > 0:
            logging.warning(f"{nat_count} NaT values ({nat_count/len(df):.1%}) in datetime parsing")
            # Log sample of problematic dates
            if nat_count > 0:
                nat_samples = df[df['tracked_at'].isna()].head(3)
                logging.warning(f"Sample unparseable dates: \n{nat_samples[[date_col, time_col]].to_string()}")
        
        # Create output dataframe with explicit type conversion
        try:
            gps_data = pd.DataFrame({
                'user_id': participant_id,
                'tracked_at': df['tracked_at'],
                'latitude': pd.to_numeric(df[lat_col], errors='coerce'),
                'longitude': pd.to_numeric(df[lon_col], errors='coerce'),
                'data_source': 'smartphone'
            })
            
            # Log sample of converted coordinates
            valid_sample = gps_data.dropna(subset=['latitude', 'longitude']).head(1)
            if not valid_sample.empty:
                logging.info(f"Converted coordinate sample: Lat={valid_sample['latitude'].iloc[0]}, " +
                           f"Lon={valid_sample['longitude'].iloc[0]}")
                
        except Exception as e:
            logging.error(f"Error converting coordinate data: {str(e)}")
            return None
        
        # Add accuracy column if available
        if 'accuracy' in df.columns:
            gps_data['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
            
            # Filter out low-accuracy points
            before_acc = len(gps_data)
            gps_data = gps_data[gps_data['accuracy'] <= 100]  # Keep only points with better than 100m accuracy
            removed_acc = before_acc - len(gps_data)
            
            if removed_acc > 0:
                logging.info(f"Removed {removed_acc} points with accuracy worse than 100m")
        
        # Filter NaN values
        gps_data = gps_data.dropna(subset=['tracked_at', 'latitude', 'longitude'])
        
        # Add date column
        gps_data['date'] = gps_data['tracked_at'].dt.date
        
        # Fix date issues (day/month confusion and future dates)
        gps_data = fix_smartphone_dates(gps_data)
        
        # Add timezone if not present
        if gps_data['tracked_at'].dt.tz is None:
            gps_data['tracked_at'] = gps_data['tracked_at'].dt.tz_localize('UTC')
        
        # Clean coordinates and filter speed outliers
        gps_data = clean_coordinates(gps_data, source='smartphone')
        gps_data = filter_speed_outliers(gps_data, max_speed_kph=200)  # Higher threshold for smartphones
        
        if not gps_data.empty:
            days_covered = gps_data['date'].nunique()
            logging.info(f"Processed {len(gps_data)} smartphone GPS points for {participant_id}")
            logging.info(f"Processed {days_covered} days of smartphone GPS data")
            
            # Log date range
            min_date = gps_data['date'].min()
            max_date = gps_data['date'].max()
            logging.info(f"Date range: {min_date} to {max_date}")
        else:
            logging.warning(f"No valid smartphone data after cleaning for {participant_id}")
            
        return gps_data
        
    except Exception as e:
        logging.error(f"Error processing smartphone GPS file {file_path}: {str(e)}")
        traceback.print_exc()
        return None


def read_app_data(file_path):
    """Read app usage data with enhanced format handling"""
    if not os.path.exists(file_path) or os.path.basename(file_path).startswith('._'):
        return None
        
    logging.info(f"Processing app data: {file_path}")
    
    try:
        # Read the file with multiple approaches
        app_df = None
        
        # First try with automatic delimiter detection
        try:
            app_df = pd.read_csv(file_path, engine='python', sep=None)
        except Exception as e:
            logging.warning(f"First attempt to read app data failed: {str(e)}")
            
            # Try with common delimiters
            for sep in [',', '\t', ';']:
                for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                    try:
                        app_df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                        if app_df is not None and not app_df.empty:
                            logging.info(f"Successfully read app data with delimiter '{sep}' and encoding '{encoding}'")
                            break
                    except Exception:
                        continue
                if app_df is not None and not app_df.empty:
                    break
        
        if app_df is None or app_df.empty:
            logging.error(f"Failed to read app data file: {file_path}")
            return None
            
        # Log columns for debugging
        logging.info(f"App data columns: {app_df.columns.tolist()}")
        
        # Find date and time columns
        date_col = next((col for col in app_df.columns if col.lower() in 
                       ['date', 'timestamp', 'datetime']), None)
        time_col = next((col for col in app_df.columns if col.lower() == 'time' and 
                       (col.lower() != date_col.lower() if date_col else True)), None)
        
        if not date_col:
            logging.error(f"Date column not found in {file_path}")
            return None
        
        # Parse timestamps
        if time_col:
            # Combine date and time
            app_df['timestamp'] = pd.to_datetime(
                app_df[date_col].astype(str) + ' ' + app_df[time_col].astype(str),
                dayfirst=True,  # For Israel data
                errors='coerce'
            )
        else:
            # Single datetime column
            app_df['timestamp'] = pd.to_datetime(app_df[date_col], errors='coerce')
        
        # Drop rows with invalid timestamps
        app_df = app_df.dropna(subset=['timestamp'])
        
        # Find action column
        action_col = next((col for col in app_df.columns 
                         if any(term in col.lower() for term in 
                             ['action', 'event', 'screen', 'status'])), None)
        
        if action_col:
            app_df['action'] = app_df[action_col]
        else:
            app_df['action'] = 'UNKNOWN'
        
        # Keep only essential columns
        app_df = app_df[['timestamp', 'action']]
        
        # Add date column
        app_df['date'] = app_df['timestamp'].dt.date
        
        # Fix future dates
        current_date = datetime.now().date()
        future_mask = app_df['date'] > current_date
        if future_mask.any():
            future_dates = app_df.loc[future_mask, 'timestamp']
            for year in [2023, 2024]:
                app_df.loc[future_mask, 'timestamp'] = future_dates.apply(
                    lambda dt: dt.replace(year=year) if dt.date() > current_date else dt
                )
            app_df['date'] = app_df['timestamp'].dt.date
        
        logging.info(f"Processed {len(app_df)} app events")
        return app_df
        
    except Exception as e:
        logging.error(f"Error processing app data {file_path}: {str(e)}")
        traceback.print_exc()
        return None


# --------------- METRICS AND MERGING ---------------

def calculate_metrics(gps_df, source_name="unknown"):
    """Calculate quality metrics for GPS data"""
    if gps_df is None or gps_df.empty:
        return {
            'source': source_name,
            'quality_score': 0,
            'total_points': 0,
            'days': 0,
            'points_per_day': 0,
            'average_gap_seconds': float('inf'),
            'spatial_coverage': 0,
            'has_data': False
        }
    
    # Basic metrics
    days = gps_df['date'].nunique()
    total_points = len(gps_df)
    points_per_day = total_points / max(1, days)
    
    # Average gap
    if len(gps_df) > 1:
        sorted_df = gps_df.sort_values('tracked_at')
        time_diffs = (sorted_df['tracked_at'].shift(-1) - sorted_df['tracked_at']).dt.total_seconds()
        valid_diffs = time_diffs[(time_diffs > 0) & (time_diffs < 3600)]  # <1 hour
        average_gap = valid_diffs.mean() if not valid_diffs.empty else float('inf')
    else:
        average_gap = float('inf')
    
    # Spatial coverage with outlier trimming
    coordinate_range = {'lat': (0, 0), 'lon': (0, 0)}
    if 'latitude' in gps_df.columns and 'longitude' in gps_df.columns:
        # Store full range
        lat_min = gps_df['latitude'].min()
        lat_max = gps_df['latitude'].max()
        lon_min = gps_df['longitude'].min()
        lon_max = gps_df['longitude'].max()
        
        coordinate_range = {
            'lat': (lat_min, lat_max),
            'lon': (lon_min, lon_max)
        }
        
        # Use trimmed range (5-95%) for spatial coverage
        lat_5 = gps_df['latitude'].quantile(0.05)
        lat_95 = gps_df['latitude'].quantile(0.95)
        lon_5 = gps_df['longitude'].quantile(0.05)
        lon_95 = gps_df['longitude'].quantile(0.95)
        
        # Calculate coverage
        lat_range = lat_95 - lat_5
        lon_range = lon_95 - lon_5
        spatial_coverage = lat_range * lon_range * 111 * 111  # km²
        
        # Cap at reasonable value
        spatial_coverage = min(spatial_coverage, MAX_REASONABLE_COVERAGE)
    else:
        spatial_coverage = 0
    
    # Quality score (0-100)
    score = min(100, (
        min(50, points_per_day / 5) +        # Up to 50 points for density
        min(20, spatial_coverage / 10) +     # Up to 20 points for spatial coverage
        min(30, 30 * (1 - min(1, average_gap / 300)))  # Up to 30 points for low gaps
    ))
    
    return {
        'source': source_name,
        'quality_score': score,
        'total_points': total_points,
        'days': days,
        'points_per_day': points_per_day,
        'average_gap_seconds': average_gap,
        'spatial_coverage': spatial_coverage,
        'coordinate_range': coordinate_range,
        'has_data': True
    }


def merge_gps_sources(qstarz_data, smartphone_data, participant_id="unknown"):
    """Merge GPS data from Qstarz and smartphone sources"""
    # Handle missing source(s)
    if qstarz_data is None or qstarz_data.empty:
        if smartphone_data is None or smartphone_data.empty:
            logging.warning(f"No valid GPS data for participant {participant_id}")
            return pd.DataFrame()
        
        logging.info(f"Only smartphone GPS data available for {participant_id}")
        smartphone_copy = smartphone_data.copy()
        smartphone_copy['data_source'] = 'smartphone'
        return smartphone_copy
    
    if smartphone_data is None or smartphone_data.empty:
        logging.info(f"Only Qstarz GPS data available for {participant_id}")
        qstarz_copy = qstarz_data.copy()
        qstarz_copy['data_source'] = 'qstarz'
        return qstarz_copy
    
    # Ensure date columns exist
    for df in [qstarz_data, smartphone_data]:
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['tracked_at']).dt.date
    
    # Calculate metrics
    qstarz_metrics = calculate_metrics(qstarz_data, "qstarz")
    smartphone_metrics = calculate_metrics(smartphone_data, "smartphone")
    
    logging.info(f"Qstarz quality score: {qstarz_metrics['quality_score']:.1f}, "
               f"Smartphone quality score: {smartphone_metrics['quality_score']:.1f}")
    
    # Find dates with data in both sources
    qstarz_dates = set(qstarz_data['date'].unique())
    smartphone_dates = set(smartphone_data['date'].unique())
    common_dates = qstarz_dates.intersection(smartphone_dates)
    qstarz_only_dates = qstarz_dates - smartphone_dates
    smartphone_only_dates = smartphone_dates - qstarz_dates
    
    logging.info(f"Days with data in both sources: {len(common_dates)}")
    logging.info(f"Days with data only in Qstarz: {len(qstarz_only_dates)}")
    logging.info(f"Days with data only in smartphone: {len(smartphone_only_dates)}")
    
    # Initialize merged parts
    merged_parts = []
    
    # Process days with data in both sources
    for date in common_dates:
        qstarz_day = qstarz_data[qstarz_data['date'] == date].copy()
        smartphone_day = smartphone_data[smartphone_data['date'] == date].copy()
        
        # Skip empty days
        if qstarz_day.empty or smartphone_day.empty:
            if not qstarz_day.empty:
                qstarz_day['data_source'] = 'qstarz'
                merged_parts.append(qstarz_day)
                quality_report['qstarz_only_days'] += 1
            elif not smartphone_day.empty:
                smartphone_day['data_source'] = 'smartphone'
                merged_parts.append(smartphone_day)
                quality_report['smartphone_only_days'] += 1
            continue
        
        # Calculate day-specific metrics
        qstarz_day_metrics = calculate_metrics(qstarz_day, f"qstarz_{date}")
        smartphone_day_metrics = calculate_metrics(smartphone_day, f"smartphone_{date}")
        
        # Check for spatial coverage discrepancy on this day
        qstarz_daily_coverage = qstarz_day_metrics['spatial_coverage']
        smartphone_daily_coverage = smartphone_day_metrics['spatial_coverage']
        
        # Both sources have usable data for this day
        qstarz_quality = qstarz_day_metrics['quality_score']
        smartphone_quality = smartphone_day_metrics['quality_score']
        
        # Smart merging - prefer one source or intelligently combine both
        if qstarz_quality >= 60 and smartphone_quality >= 60:
            # Both are good quality - merge them intelligently
            merged_day = qstarz_day.copy()
            merged_day['data_source'] = 'qstarz'
            
            # Sort by time
            qstarz_day = qstarz_day.sort_values('tracked_at')
            smartphone_day = smartphone_day.sort_values('tracked_at')
            
            # Identify gaps in Qstarz data
            qstarz_day['time_diff'] = qstarz_day['tracked_at'].diff().dt.total_seconds()
            gaps = qstarz_day[qstarz_day['time_diff'] > 300]  # 5-min gaps
            
            # Fill gaps with smartphone data
            for _, gap_row in gaps.iterrows():
                gap_start = gap_row['tracked_at'] - timedelta(seconds=300)
                
                # Find smartphone points in this gap
                gap_points = smartphone_day[
                    (smartphone_day['tracked_at'] > gap_start) & 
                    (smartphone_day['tracked_at'] < gap_row['tracked_at'])
                ].copy()
                
                if not gap_points.empty:
                    gap_points['data_source'] = 'merged_high_conf'
                    merged_day = pd.concat([merged_day, gap_points], ignore_index=True)
            
            # Add smartphone data outside Qstarz timespan
            if not qstarz_day.empty and not smartphone_day.empty:
                earliest_qstarz = qstarz_day['tracked_at'].min()
                latest_qstarz = qstarz_day['tracked_at'].max()
                
                # Early morning smartphone data
                early_points = smartphone_day[smartphone_day['tracked_at'] < earliest_qstarz].copy()
                if not early_points.empty:
                    early_points['data_source'] = 'merged_high_conf'
                    merged_day = pd.concat([merged_day, early_points], ignore_index=True)
                
                # Late evening smartphone data
                late_points = smartphone_day[smartphone_day['tracked_at'] > latest_qstarz].copy()
                if not late_points.empty:
                    late_points['data_source'] = 'merged_high_conf'
                    merged_day = pd.concat([merged_day, late_points], ignore_index=True)
            
            # Sort merged data
            merged_day = merged_day.sort_values('tracked_at')
            merged_parts.append(merged_day)
            quality_report['merged_days'] += 1
            
        elif qstarz_quality >= smartphone_quality * 1.2:  # Qstarz significantly better
            qstarz_day['data_source'] = 'qstarz'
            merged_parts.append(qstarz_day)
            quality_report['qstarz_only_days'] += 1
        else:  # Smartphone is better or comparable
            smartphone_day['data_source'] = 'smartphone'
            merged_parts.append(smartphone_day)
            quality_report['smartphone_only_days'] += 1
    
    # Add days with data in only one source
    for date in qstarz_only_dates:
        qstarz_day = qstarz_data[qstarz_data['date'] == date].copy()
        qstarz_day['data_source'] = 'qstarz'
        merged_parts.append(qstarz_day)
        quality_report['qstarz_only_days'] += 1
    
    for date in smartphone_only_dates:
        smartphone_day = smartphone_data[smartphone_data['date'] == date].copy()
        smartphone_day['data_source'] = 'smartphone'
        merged_parts.append(smartphone_day)
        quality_report['smartphone_only_days'] += 1
    
    # Combine all parts
    if not merged_parts:
        return pd.DataFrame()
        
    merged_data = pd.concat(merged_parts, ignore_index=True)
    
    # Sort by time
    merged_data = merged_data.sort_values('tracked_at')
    
    return merged_data


# --------------- MAIN PROCESSING ---------------

def process_participant(participant_id, qstarz_files, app_files, app_gps_files):
    """Process data for a single participant"""
    result = {
        'participant_id': participant_id,
        'success': False,
        'qstarz_days': 0,
        'smartphone_days': 0,
        'common_days': 0,
        'reason': None
    }
    
    try:
        # Process Qstarz data if available
        qstarz_data = None
        if participant_id in qstarz_files:
            qstarz_data = read_qstarz_data(qstarz_files[participant_id])
            
            if qstarz_data is not None and not qstarz_data.empty:
                result['qstarz_days'] = len(qstarz_data['date'].unique())
        
        # Process smartphone GPS data if available
        smartphone_data = None
        if participant_id in app_gps_files:
            smartphone_data = read_smartphone_data(app_gps_files[participant_id])
            
            if smartphone_data is not None and not smartphone_data.empty:
                result['smartphone_days'] = len(smartphone_data['date'].unique())
        
        # Process app usage data if available
        app_data = None
        if participant_id in app_files:
            app_data = read_app_data(app_files[participant_id])
        
        # Merge GPS data sources
        merged_gps = merge_gps_sources(qstarz_data, smartphone_data, participant_id)
        
        if merged_gps is None or merged_gps.empty:
            result['reason'] = "No valid GPS data after processing"
            return result
        
        # Calculate overlapping days
        if qstarz_data is not None and smartphone_data is not None:
            qstarz_dates = set(qstarz_data['date'].unique())
            smartphone_dates = set(smartphone_data['date'].unique())
            common_dates = qstarz_dates.intersection(smartphone_dates)
            result['common_days'] = len(common_dates)
        
        # Save the processed data
        output_path = GPS_PREP_DIR / f"{participant_id}_processed_gps.csv"
        merged_gps.to_csv(output_path, index=False)
        logging.info(f"Saved processed GPS data to {output_path}")
        
        # Generate diagnostics report
        diagnostics_dir = GPS_PREP_DIR / participant_id
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = diagnostics_dir / f"{participant_id}_data_diagnostics.txt"
        
        with open(diagnostics_path, 'w') as f:
            f.write(f"Data Diagnostics for Participant {participant_id}\n")
            f.write("="*80 + "\n\n")
            
            f.write("DATA SOURCE QUALITY METRICS\n")
            f.write("-"*80 + "\n")
            
            if qstarz_data is not None and not qstarz_data.empty:
                qstarz_metrics = calculate_metrics(qstarz_data, "qstarz")
                f.write("Qstarz Data:\n")
                f.write(f"  - Quality Score: {qstarz_metrics['quality_score']:.1f}/100\n")
                f.write(f"  - Total Points: {qstarz_metrics['total_points']}\n")
                f.write(f"  - Days Covered: {qstarz_metrics['days']}\n")
                f.write(f"  - Points per Day: {qstarz_metrics['points_per_day']:.1f}\n")
                f.write(f"  - Average Gap: {qstarz_metrics['average_gap_seconds']:.1f} seconds\n")
                f.write(f"  - Spatial Coverage: {qstarz_metrics['spatial_coverage']:.2f} km²\n\n")
            
            if smartphone_data is not None and not smartphone_data.empty:
                smartphone_metrics = calculate_metrics(smartphone_data, "smartphone")
                f.write("Smartphone Data:\n")
                f.write(f"  - Quality Score: {smartphone_metrics['quality_score']:.1f}/100\n")
                f.write(f"  - Total Points: {smartphone_metrics['total_points']}\n")
                f.write(f"  - Days Covered: {smartphone_metrics['days']}\n")
                f.write(f"  - Points per Day: {smartphone_metrics['points_per_day']:.1f}\n")
                f.write(f"  - Average Gap: {smartphone_metrics['average_gap_seconds']:.1f} seconds\n")
                f.write(f"  - Spatial Coverage: {smartphone_metrics['spatial_coverage']:.2f} km²\n\n")
            
            # Merged data metrics
            merged_metrics = calculate_metrics(merged_gps, "merged")
            f.write("Merged Data:\n")
            f.write(f"  - Quality Score: {merged_metrics['quality_score']:.1f}/100\n")
            f.write(f"  - Total Points: {merged_metrics['total_points']}\n")
            f.write(f"  - Days Covered: {merged_metrics['days']}\n")
            f.write(f"  - Points per Day: {merged_metrics['points_per_day']:.1f}\n")
            f.write(f"  - Average Gap: {merged_metrics['average_gap_seconds']:.1f} seconds\n")
            f.write(f"  - Spatial Coverage: {merged_metrics['spatial_coverage']:.2f} km²\n\n")
            
            # Data source breakdown
            f.write("MERGED DATA SOURCE BREAKDOWN\n")
            f.write("-"*80 + "\n")
            source_counts = merged_gps['data_source'].value_counts()
            for source, count in source_counts.items():
                f.write(f"{source}: {count} points ({count/len(merged_gps)*100:.1f}%)\n")
        
        logging.info(f"Saved diagnostics to {diagnostics_path}")
        
        # Mark as success
        result['success'] = True
        quality_report['successful'] += 1
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {str(e)}")
        traceback.print_exc()
        result['reason'] = str(e)
        quality_report['failed'] += 1
        quality_report['failed_reasons'].append(f"{participant_id}: {str(e)}")
        return result


def main():
    """Main processing function"""
    start_time = datetime.now()
    
    # 1. Find data files with flexible path handling
    qstarz_files = {}
    
    # Look for Qstarz files in various locations
    qstarz_patterns = [
        '**/Qstarz*processed.csv',
        '**/qstarz*processed.csv',
        '**/Qstarz*.csv',
        '**/qstarz*.csv',
        '**/GPS/Q*.csv',
        '**/*_Qstarz_*.csv',     # Added pattern for files like 006_1_Qstarz_processed.csv
        '**/*_Qstarz*.csv',      # More general pattern with underscore
        '**/Q*.csv'              # Very general pattern for any Q files
    ]
    
    # Check in both raw data dir and processed data dir for Qstarz files
    search_dirs = [RAW_DATA_DIR, GPS_PREP_DIR]
    
    for search_dir in search_dirs:
        logging.info(f"Searching for Qstarz files in {search_dir}")
        for pattern in qstarz_patterns:
            for f in Path(search_dir).glob(pattern):
                if not os.path.basename(f).startswith('._'):
                    # Extract participant ID from filename
                    filename = os.path.basename(f)
                    if '_' in filename:
                        participant_id = filename.split('_')[0]
                    elif '-' in filename:
                        participant_id = filename.split('-')[0]
                    else:
                        # Try to extract numeric part from filename
                        match = re.search(r'(\d+)', filename)
                        if match:
                            participant_id = match.group(1)
                        else:
                            continue
                    
                    # Remove leading zeros but ensure it's a valid integer
                    try:
                        participant_id = str(int(participant_id))
                    except ValueError:
                        pass
                    
                    qstarz_files[participant_id] = f
    
    logging.info(f"Found {len(qstarz_files)} Qstarz files")
    
    # Get app files and smartphone GPS files with more flexible path handling
    app_files = {}
    app_gps_files = {}
    
    # Try multiple participant folder patterns
    participant_patterns = ["Pilot_*", "Participant_*", "Subject_*", "P_*", "P*"]
    
    for pattern in participant_patterns:
        for participant_folder in (RAW_DATA_DIR / "Participants").glob(pattern):
            if os.path.basename(participant_folder).startswith('._'):
                continue
                
            # Extract participant ID
            full_participant_id = participant_folder.name
            
            # Try different patterns to extract the participant ID
            if '_' in full_participant_id:
                participant_id = full_participant_id.split('_')[-1]
            else:
                # Try to extract numeric part
                match = re.search(r'(\d+)', full_participant_id)
                if match:
                    participant_id = match.group(1)
                else:
                    continue
            
            # Normalize participant ID (remove leading zeros but keep as string)
            try:
                participant_id = str(int(participant_id))
            except ValueError:
                pass
            
            # Look for app data in various potential locations
            app_folders = [
                participant_folder / '9 - Smartphone Tracking App',
                participant_folder / 'Smartphone App',
                participant_folder / 'App',
                participant_folder / 'GPS',
                participant_folder
            ]
            
            for app_folder in app_folders:
                if not app_folder.exists():
                    continue
                
                # Patterns for app files
                app_patterns = [
                    f'{participant_id.lstrip("0")}-apps.csv',
                    f'{participant_id}-apps.csv',
                    '*-apps.csv',
                    f'{participant_id.lstrip("0")}_apps.csv',
                    f'{participant_id}_apps.csv',
                    '*_apps.csv',
                    'apps.csv',
                    'app_usage.csv',
                    'app_data.csv'
                ]
                
                # Patterns for GPS files
                gps_patterns = [
                    f'{participant_id.lstrip("0")}-gps.csv',
                    f'{participant_id}-gps.csv',
                    '*-gps.csv',
                    f'{participant_id.lstrip("0")}_gps.csv',
                    f'{participant_id}_gps.csv',
                    '*_gps.csv',
                    'gps.csv',
                    'smartphone_gps.csv',
                    'phone_gps.csv'
                ]
                
                # Look for app files
                for pattern in app_patterns:
                    app_file = next((f for f in app_folder.glob(pattern) 
                                    if not os.path.basename(f).startswith('._')), None)
                    if app_file:
                        app_files[participant_id] = app_file
                        break
                
                # Look for smartphone GPS files
                for pattern in gps_patterns:
                    app_gps_file = next((f for f in app_folder.glob(pattern) 
                                        if not os.path.basename(f).startswith('._')), None)
                    if app_gps_file:
                        app_gps_files[participant_id] = app_gps_file
                        break
                
                # If we found files, no need to check other app folders
                if participant_id in app_files or participant_id in app_gps_files:
                    break
    
    logging.info(f"Found {len(app_files)} app files and {len(app_gps_files)} smartphone GPS files")
    
    # 2. Determine participants to process
    participants_with_gps = set(qstarz_files.keys()) | set(app_gps_files.keys())
    participants_with_app = set(app_files.keys())
    processable_participants = participants_with_gps & participants_with_app
    
    logging.info(f"Found {len(processable_participants)} processable participants")
    quality_report['participants_processed'] = len(processable_participants)

    # 3. Process participants
    results = []
    
    for participant_id in processable_participants:
        if participant_id.startswith('._'):
            continue
            
        logging.info(f"Processing participant {participant_id}...")
        result = process_participant(participant_id, qstarz_files, app_files, app_gps_files)
        results.append(result)
        
        if result['success']:
            logging.info(f"Successfully processed participant {participant_id}")
        else:
            logging.error(f"Failed to process {participant_id}: {result['reason']}")
    
    # 4. Generate quality report
    report_path = GPS_PREP_DIR / "preprocessing_report.txt"
    run_time = (datetime.now() - start_time).total_seconds()
    
    with open(report_path, 'w') as f:
        f.write(f"GPS Preprocessing Report ({quality_report['start_time']})\n")
        f.write("="*80 + "\n")
        f.write(f"Participants processed: {quality_report['participants_processed']}\n")
        f.write(f"Successfully processed: {quality_report['successful']}\n")
        f.write(f"Failed: {quality_report['failed']}\n")
        f.write(f"Processing time: {run_time:.1f} seconds\n\n")
        
        f.write("DATA FUSION STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Days with merged data from both sources: {quality_report['merged_days']}\n")
        f.write(f"Days with only Qstarz data: {quality_report['qstarz_only_days']}\n")
        f.write(f"Days with only smartphone data: {quality_report['smartphone_only_days']}\n")
        f.write(f"Coordinate fixes applied: {quality_report['coordinate_fixes']}\n")
        f.write(f"Date fixes applied: {quality_report['date_fixes']}\n")
        f.write(f"Outliers removed: {quality_report['outliers_removed']}\n\n")
        
        if quality_report['failed_reasons']:
            f.write("PROCESSING ERRORS\n")
            f.write("-"*80 + "\n")
            for reason in quality_report['failed_reasons']:
                f.write(f"- {reason}\n")
    
    logging.info(f"Quality report saved to {report_path}")
    logging.info(f"Successfully processed {quality_report['successful']}/{quality_report['participants_processed']} participants in {run_time:.1f} seconds")
    
    return quality_report['successful']


if __name__ == "__main__":
    main()