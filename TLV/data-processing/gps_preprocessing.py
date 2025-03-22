import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from pathlib import Path
import uuid

class GPSPreprocessor:
    def __init__(self, 
                 raw_gps_path: str,
                 ema_path: str,
                 output_dir: str,
                 early_morning_cutoff: int = 5):
        """
        Initialize GPS preprocessor with file paths
        
        Args:
            raw_gps_path: Path to raw GPS Excel file
            ema_path: Path to EMA questionnaire Excel file
            output_dir: Directory for output files
            early_morning_cutoff: Hour before which EMAs are considered early morning
        """
        self.raw_gps_path = Path(raw_gps_path)
        self.ema_path = Path(ema_path)
        self.output_dir = Path(output_dir)
        self.early_morning_cutoff = early_morning_cutoff
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define home location patterns - expand this list based on your data
        self.home_patterns = ['Home', '~Home', 'home', 'HOME', 'House', 'Residence']
        
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging with both console and file output"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / 'preprocessing.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load and prepare GPS and EMA datasets"""
        self.logger.info("Loading raw GPS data...")
        self.raw_gps = pd.read_csv(self.raw_gps_path, low_memory=False)
        
        self.logger.info("Loading EMA data...")
        self.ema_data = pd.read_csv(self.ema_path)
        
        # Log initial data shapes
        self.logger.info(f"Raw GPS data shape: {self.raw_gps.shape}")
        self.logger.info(f"EMA data shape: {self.ema_data.shape}")
        
        # Convert timestamps and standardize data types
        self.raw_gps['Timestamp'] = pd.to_datetime(self.raw_gps['Timestamp'])
        self.raw_gps['calendar_date'] = self.raw_gps['Timestamp'].dt.date
        self.raw_gps['user'] = self.raw_gps['user'].astype(str)
        
        # Sort GPS data by timestamp
        self.raw_gps = self.raw_gps.sort_values('Timestamp')
        
        # Add home location flag with broader pattern matching
        self._process_home_locations()
        
        # Prepare EMA data
        self.ema_data['timestamp'] = pd.to_datetime(self.ema_data['StartDate'])
        self.ema_data['response_time'] = pd.to_datetime(self.ema_data['EndDate'])
        self.ema_data['calendar_date'] = self.ema_data['response_time'].dt.date
        self.ema_data['user'] = self.ema_data['Participant_ID'].astype(str)
        
        # Add early morning flags and data associations
        self.ema_data['is_early_morning'] = self.ema_data['response_time'].dt.hour < self.early_morning_cutoff
        self.ema_data['associated_data_date'] = self.ema_data.apply(
            lambda row: (row['calendar_date'] - timedelta(days=1))
            if row['is_early_morning'] else row['calendar_date'],
            axis=1
        )
        
        # Generate unique IDs for each EMA response
        self.ema_data['ema_id'] = [str(uuid.uuid4()) for _ in range(len(self.ema_data))]
        
        # Log data ranges and early morning counts
        early_morning_count = self.ema_data['is_early_morning'].sum()
        self.logger.info(f"Found {early_morning_count} early morning responses")
        self.logger.info(f"GPS data date range: {min(self.raw_gps['calendar_date'])} to {max(self.raw_gps['calendar_date'])}")
        self.logger.info(f"EMA data date range: {min(self.ema_data['calendar_date'])} to {max(self.ema_data['calendar_date'])}")

    def _process_home_locations(self):
        """Process home location data and ensure proper flag is created"""
        # Check which columns contain location information
        location_cols = [col for col in self.raw_gps.columns if col in ['LU_na', 'place_type', 'location', 'type_of_place']]
        
        if 'LU_na' in location_cols:
            # Check unique values in location column to help with debugging
            unique_locations = self.raw_gps['LU_na'].dropna().unique()
            self.logger.info(f"Unique location values in LU_na: {unique_locations[:20]}")
            
            # Create is_home flag with broader pattern matching
            self.raw_gps['is_home'] = self.raw_gps['LU_na'].fillna('').astype(str).apply(
                lambda x: any(pattern.lower() in x.lower() for pattern in self.home_patterns)
            )
        else:
            self.logger.warning(f"No recognized location column found. Available columns: {self.raw_gps.columns.tolist()}")
            # Create a default is_home column
            self.raw_gps['is_home'] = False
        
        # Log statistics about home locations
        home_count = self.raw_gps['is_home'].sum()
        total_count = len(self.raw_gps)
        self.logger.info(f"Identified {home_count} points ({home_count/total_count*100:.1f}%) as home locations")

    def _create_ema_summary(self) -> pd.DataFrame:
        """Create summary of EMA responses with unique IDs"""
        ema_cols = ['ema_id', 'user', 'calendar_date', 'associated_data_date', 'timestamp', 
                   'response_time', 'is_early_morning', 'Gender', 'School', 'Class', 
                   'PEACE', 'TENSE', 'IRRITATION', 'RELAXATION', 'SATISFACTION', 
                   'WORRY', 'HAPPY']
        
        ema_summary = self.ema_data[ema_cols].copy()
        ema_summary['response_hour'] = ema_summary['response_time'].dt.hour
        ema_summary['response_weekday'] = ema_summary['response_time'].dt.day_name()
        
        return ema_summary

    def create_preprocessed_files(self):
        """Create preprocessed files with proper data windows and associations"""
        self.logger.info("Creating preprocessed files...")
        
        preprocess_dir = self.output_dir / 'preprocessed_data'
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle multiple responses per associated_data_date
        response_groups = self.ema_data.groupby(['user', 'associated_data_date'])
        
        latest_responses = []
        multiple_response_cases = []
        
        for (user, data_date), group in response_groups:
            # Convert all times to same day for proper temporal comparison
            normalized_times = group.apply(lambda row: 
                row['response_time'] - timedelta(days=1) if row['is_early_morning'] 
                else row['response_time'], axis=1)
            
            # Get index of temporally latest response
            latest_idx = normalized_times.idxmax()
            latest_response = group.loc[latest_idx]
            
            # Record multiple response cases for logging
            if len(group) > 1:
                response_info = {
                    'user': user,
                    'data_date': data_date,
                    'response_times': group['response_time'].tolist(),
                    'normalized_times': normalized_times.tolist(),
                    'selected_time': latest_response['response_time'],
                    'is_early_morning': latest_response['is_early_morning'],
                    'ema_ids': group['ema_id'].tolist()
                }
                multiple_response_cases.append(response_info)
            
            # Log warning if selecting early morning over same day
            same_day_responses = group[~group['is_early_morning']]
            if latest_response['is_early_morning'] and not same_day_responses.empty:
                self.logger.warning(
                    f"⚠️ Selected early morning response over same-day response(s)\n"
                    f"User: {user}\n"
                    f"Data date: {data_date}\n"
                    f"Selected: {latest_response['response_time']} (early morning)\n"
                    f"Other responses: {', '.join(same_day_responses['response_time'].dt.strftime('%Y-%m-%d %H:%M:%S'))}\n"
                    f"Normalized selected time: {normalized_times[latest_idx]}\n"
                    f"EMA IDs: Selected={latest_response['ema_id']}, "
                    f"Others={', '.join(same_day_responses['ema_id'])}"
                )
            
            latest_responses.append(latest_response)
        
        latest_responses_df = pd.DataFrame(latest_responses)
        self.logger.info(f"Processing {len(latest_responses_df)} days after resolving multiple responses")
        
        # Process GPS data for each user-day
        processed_count = 0
        skipped_count = 0
        insufficient_coverage_count = 0
        total_points = 0
        early_morning_count = 0
        
        for _, ema_row in tqdm(latest_responses_df.iterrows(), desc="Processing user-days"):
            # Set data window based on associated_data_date
            start_time = datetime.combine(ema_row['associated_data_date'], 
                                        datetime.min.time()) + timedelta(hours=self.early_morning_cutoff)
            end_time = ema_row['response_time']
            
            # Get GPS data for the time window
            day_data = self.raw_gps[
                (self.raw_gps['user'] == ema_row['user']) & 
                (self.raw_gps['Timestamp'] >= start_time) & 
                (self.raw_gps['Timestamp'] <= end_time)
            ].copy()
            
            if not day_data.empty:
                # Calculate coverage duration before processing further
                coverage_hours = (day_data['Timestamp'].max() - day_data['Timestamp'].min()).total_seconds() / 3600
                
                if coverage_hours >= 5:  # Only process days with 5+ hours of data
                    # Sort by timestamp
                    day_data = day_data.sort_values('Timestamp')
                    
                    # Ensure is_home flag exists and is boolean
                    if 'is_home' not in day_data.columns:
                        self.logger.warning(f"is_home flag missing for user {ema_row['user']} on {ema_row['associated_data_date']}")
                        day_data['is_home'] = False
                    
                    # Convert is_home to boolean if not already
                    day_data['is_home'] = day_data['is_home'].astype(bool)
                    
                    # Log home location statistics for this day
                    home_points = day_data['is_home'].sum()
                    day_points = len(day_data)
                    self.logger.info(f"User {ema_row['user']} on {ema_row['associated_data_date']}: "
                                   f"{home_points}/{day_points} points ({home_points/day_points*100:.1f}%) at home")
                    
                    # Add EMA linking information
                    day_data['ema_id'] = ema_row['ema_id']
                    day_data['ema_response_time'] = ema_row['response_time']
                    day_data['is_early_morning_response'] = ema_row['is_early_morning']
                    day_data['calendar_date'] = ema_row['calendar_date']
                    day_data['associated_data_date'] = ema_row['associated_data_date']
                    day_data['had_multiple_responses'] = any(
                        case['user'] == ema_row['user'] and 
                        case['data_date'] == ema_row['associated_data_date']
                        for case in multiple_response_cases
                    )
                    
                    # Add time period classification
                    day_data['time_period'] = day_data['Timestamp'].apply(
                        lambda x: 'early_morning' if x.hour < self.early_morning_cutoff else
                                'morning' if x.hour < 12 else
                                'afternoon' if x.hour < 17 else
                                'evening' if x.hour < 22 else 'night'
                    )
                    
                    # Calculate quality metrics
                    quality_metrics = {
                        'first_reading': day_data['Timestamp'].min(),
                        'last_reading': day_data['Timestamp'].max(),
                        'total_readings': len(day_data),
                        'time_periods_covered': day_data['time_period'].nunique(),
                        'max_gap_minutes': (
                            day_data['Timestamp'].diff().max().total_seconds() / 60 
                            if len(day_data) > 1 else np.nan
                        ),
                        'coverage_hours': (
                            day_data['Timestamp'].max() - day_data['Timestamp'].min()
                        ).total_seconds() / 3600,
                        'early_morning_points': sum(day_data['time_period'] == 'early_morning')
                    }
                    
                    # Add quality metrics to each row
                    for metric, value in quality_metrics.items():
                        day_data[metric] = value
                    
                    # Save file with clear naming
                    output_filename = (
                        f"{ema_row['associated_data_date']}_{ema_row['user']}_"
                        f"{'early_morning_' if ema_row['is_early_morning'] else ''}"
                        f"{ema_row['ema_id'][:8]}.csv"
                    )
                    output_file = preprocess_dir / output_filename
                    
                    # Ensure these essential columns are included
                    essential_columns = ['user', 'Timestamp', 'is_home', 'speed', 'Travel_mode']
                    missing_columns = [col for col in essential_columns if col not in day_data.columns]
                    if missing_columns:
                        for col in missing_columns:
                            self.logger.warning(f"Adding missing column {col}")
                            day_data[col] = None if col != 'is_home' else False
                    
                    day_data.to_csv(output_file, index=False)
                    
                    if ema_row['is_early_morning']:
                        early_morning_count += 1
                    
                    processed_count += 1
                    total_points += len(day_data)
                else:
                    insufficient_coverage_count += 1
                    self.logger.warning(
                        f"Insufficient GPS coverage for user {ema_row['user']}\n"
                        f"Calendar date: {ema_row['calendar_date']}\n"
                        f"Coverage hours: {coverage_hours:.2f}"
                    )
            else:
                skipped_count += 1
                self.logger.warning(
                    f"No GPS data found for user {ema_row['user']}\n"
                    f"Calendar date: {ema_row['calendar_date']}\n"
                    f"Associated data date: {ema_row['associated_data_date']}\n"
                    f"Time window: {start_time} to {end_time}"
                )
        
        self.logger.info(f"\nProcessing Summary:")
        self.logger.info(f"Created {processed_count} preprocessed files")
        self.logger.info(f"Processed {early_morning_count} early morning responses")
        self.logger.info(f"Skipped {skipped_count} user-days with no GPS data")
        self.logger.info(f"Dropped {insufficient_coverage_count} days with < 5 hours coverage")
        self.logger.info(f"Total GPS points processed: {total_points}")
        
        # Create and save data quality summary
        self._save_quality_summary()

    def _save_quality_summary(self):
        """Create and save summary of data quality"""
        quality_summary = []
        
        for _, ema_row in self.ema_data.iterrows():
            # Find corresponding GPS file using consistent naming pattern
            gps_file = next(self.output_dir.glob(
                f"preprocessed_data/{ema_row['associated_data_date']}_{ema_row['user']}_"
                f"{'early_morning_' if ema_row['is_early_morning'] else ''}"
                f"{ema_row['ema_id'][:8]}.csv"
            ), None)
            
            if gps_file:
                gps_data = pd.read_csv(gps_file)
                
                summary = {
                    'ema_id': ema_row['ema_id'],
                    'user': ema_row['user'],
                    'calendar_date': ema_row['calendar_date'],
                    'associated_data_date': ema_row['associated_data_date'],
                    'is_early_morning': ema_row['is_early_morning'],
                    'ema_timestamp': ema_row['timestamp'],
                    'ema_response_time': ema_row['response_time'],
                    'gps_file': gps_file.name,
                    'total_gps_points': len(gps_data),
                    'coverage_hours': gps_data['coverage_hours'].iloc[0],
                    'has_morning_data': any(pd.to_datetime(gps_data['Timestamp']).dt.hour < 12),
                    'has_evening_data': any(pd.to_datetime(gps_data['Timestamp']).dt.hour >= 17),
                    'has_early_morning_data': any(pd.to_datetime(gps_data['Timestamp']).dt.hour < self.early_morning_cutoff),
                    'max_gap_minutes': gps_data['max_gap_minutes'].iloc[0]
                }
                
                # Add data quality classification
                summary['data_quality'] = 'good' if (
                    summary['has_morning_data'] and 
                    summary['has_evening_data'] and 
                    summary['coverage_hours'] >= 5 and
                    summary['max_gap_minutes'] <= 240  # 4-hour max gap
                ) else 'partial'
                
                quality_summary.append(summary)
        
        # Convert to DataFrame and save
        quality_df = pd.DataFrame(quality_summary)
        quality_df.to_csv(self.output_dir / 'data_quality_summary.csv', index=False)
        
        # Log summary statistics
        self.logger.info("\nData Quality Summary:")
        self.logger.info(f"Total EMAs processed: {len(self.ema_data)}")
        self.logger.info(f"EMAs with matching GPS data: {len(quality_summary)}")
        self.logger.info(f"EMAs with good quality data: {(quality_df['data_quality'] == 'good').sum()}")
        self.logger.info(f"Early morning responses: {quality_df['is_early_morning'].sum()}")
        self.logger.info(f"Average GPS points per day: {quality_df['total_gps_points'].mean():.1f}")
        self.logger.info(f"Average coverage hours: {quality_df['coverage_hours'].mean():.1f}")
        
        # Create summary by participant
        participant_summary = quality_df.groupby('user').agg({
            'ema_id': 'count',
            'total_gps_points': 'mean',
            'coverage_hours': 'mean',
            'data_quality': lambda x: (x == 'good').mean() * 100,
            'is_early_morning': 'sum'
        }).round(2)
        
        participant_summary.columns = ['total_days', 'avg_gps_points', 
                                    'avg_coverage_hours', 'good_quality_percent',
                                    'early_morning_responses']
        
        # Add additional early morning metrics
        early_morning_stats = quality_df[quality_df['is_early_morning']].groupby('user').agg({
            'total_gps_points': 'mean',
            'coverage_hours': 'mean',
            'data_quality': lambda x: (x == 'good').mean() * 100
        }).round(2)
        
        if not early_morning_stats.empty:
            participant_summary['early_morning_avg_points'] = early_morning_stats['total_gps_points']
            participant_summary['early_morning_avg_coverage'] = early_morning_stats['coverage_hours']
            participant_summary['early_morning_good_quality_percent'] = early_morning_stats['data_quality']
        
        participant_summary.to_csv(self.output_dir / 'participant_summary.csv')
        
        self.logger.info("\nParticipant Summary:")
        self.logger.info(f"Total participants: {len(participant_summary)}")
        self.logger.info(f"Average days per participant: {participant_summary['total_days'].mean():.1f}")
        self.logger.info(f"Average good quality data percentage: {participant_summary['good_quality_percent'].mean():.1f}%")
        self.logger.info(f"Average early morning responses per participant: {participant_summary['early_morning_responses'].mean():.1f}")

    def process(self):
        """Run the complete preprocessing pipeline"""
        try:
            self.load_data()
            self.create_preprocessed_files()
            self.logger.info("Preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

def main():
    # Define paths (modify these as needed)
    RAW_GPS_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/csv/gpsappS_9.1_excel.csv'
    EMA_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/csv/End_of_the_day_questionnaire.csv'
    OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_summaries'
    
    # Initialize and run preprocessor
    preprocessor = GPSPreprocessor(
        raw_gps_path=RAW_GPS_PATH,
        ema_path=EMA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    preprocessor.process()

if __name__ == "__main__":
    main()



    