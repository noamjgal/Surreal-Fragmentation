import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from pathlib import Path

class GPSPreprocessor:
    def __init__(self, 
                 raw_gps_path: str,
                 ema_path: str,
                 output_dir: str,
                 early_morning_cutoff: int = 5,
                 afternoon_cutoff: int = 16):  # Add second cutoff parameter
        """
        Initialize GPS preprocessor with file paths
        
        Args:
            raw_gps_path: Path to raw GPS Excel file
            ema_path: Path to EMA questionnaire Excel file
            output_dir: Directory for output files
            early_morning_cutoff: Hour before which EMAs are assigned to previous day
            afternoon_cutoff: Hour before which EMAs are excluded (default 16:00)
        """
        self.raw_gps_path = Path(raw_gps_path)
        self.ema_path = Path(ema_path)
        self.output_dir = Path(output_dir)
        self.early_morning_cutoff = early_morning_cutoff
        self.afternoon_cutoff = afternoon_cutoff
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
            """Load and prepare GPS and EMA datasets from CSV files"""
            self.logger.info("Loading raw GPS data...")
            self.raw_gps = pd.read_csv(self.raw_gps_path)
            
            self.logger.info("Loading EMA data...")
            self.ema_data = pd.read_csv(self.ema_path)
            
            # Log columns for debugging
            self.logger.info("GPS columns: %s", self.raw_gps.columns.tolist())
            self.logger.info("EMA columns: %s", self.ema_data.columns.tolist())
            
            # Convert and standardize GPS data types
            self.raw_gps['date'] = pd.to_datetime(self.raw_gps['date']).dt.date
            self.raw_gps['user'] = self.raw_gps['user'].astype(str)
            self.raw_gps['Timestamp'] = pd.to_datetime(self.raw_gps['Timestamp'])
            
            # Prepare EMA data using StartDate for matching
            self.ema_data['timestamp'] = pd.to_datetime(self.ema_data['StartDate'])
            self.ema_data['date'] = self.ema_data['timestamp'].dt.date
            
            # Adjust dates for early morning responses
            early_morning_mask = self.ema_data['timestamp'].dt.hour < self.early_morning_cutoff
            self.ema_data.loc[early_morning_mask, 'date'] = self.ema_data.loc[
                early_morning_mask, 'timestamp'
            ].dt.date - timedelta(days=1)
            
            # Verify early morning adjustment
            verification_mask = early_morning_mask & (self.ema_data['timestamp'].dt.date != self.ema_data['date'])
            if not verification_mask.equals(early_morning_mask):
                mismatched = self.ema_data[early_morning_mask & ~verification_mask]
                self.logger.error(
                    f"Early morning adjustment verification failed for {len(mismatched)} records:"
                )
                for _, row in mismatched.iterrows():
                    self.logger.error(
                        f"User {row['user']}: Timestamp {row['timestamp']} not properly "
                        f"adjusted to date {row['date']}"
                    )
                raise ValueError("Early morning date adjustment verification failed")
            
            # Log early morning adjustments
            adjusted_count = early_morning_mask.sum()
            if adjusted_count > 0:
                self.logger.info(f"Verified {adjusted_count} early morning EMAs were correctly adjusted to previous day")
                self.logger.info("Sample of adjusted EMAs:")
                adjusted_emas = self.ema_data[early_morning_mask].head()
                for _, row in adjusted_emas.iterrows():
                    self.logger.info(
                        f"Survey at {row['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                        f"assigned to {row['date']}"
                    )
            
            # Standardize participant IDs
            self.ema_data['user'] = self.ema_data['Participant_ID'].astype(str)
            
            # Log data loading results
            self.logger.info(f"Loaded {len(self.raw_gps)} GPS records")
            self.logger.info(f"Loaded {len(self.ema_data)} EMA records")
            self.logger.info(f"GPS data date range: {self.raw_gps['date'].min()} to {self.raw_gps['date'].max()}")
            self.logger.info(f"EMA data date range: {min(self.ema_data['date'])} to {max(self.ema_data['date'])}")
            
            # Check for multiple EMAs per day after date adjustment
            ema_counts = self.ema_data.groupby(['user', 'date']).size()
            multiple_emas = ema_counts[ema_counts > 1]
            if not multiple_emas.empty:
                self.logger.warning(f"Found {len(multiple_emas)} user-days with multiple EMAs:")
                for (user, date), count in multiple_emas.items():
                    ema_times = self.ema_data[
                        (self.ema_data['user'] == user) & 
                        (self.ema_data['date'] == date)
                    ]['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                    self.logger.warning(f"User {user} on {date}: {count} EMAs at times: {ema_times}")

            # After the early morning adjustment, filter out responses between cutoffs
            invalid_time_mask = (
                (self.ema_data['timestamp'].dt.hour >= self.early_morning_cutoff) & 
                (self.ema_data['timestamp'].dt.hour < self.afternoon_cutoff)
            )
            
            if invalid_time_mask.any():
                excluded_count = invalid_time_mask.sum()
                self.logger.warning(
                    f"Excluding {excluded_count} EMAs between {self.early_morning_cutoff}:00 "
                    f"and {self.afternoon_cutoff}:00"
                )
                self.logger.info("Sample of excluded EMAs:")
                excluded_emas = self.ema_data[invalid_time_mask].head()
                for _, row in excluded_emas.iterrows():
                    self.logger.info(
                        f"Excluded survey at {row['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                        f"for user {row['user']}"
                    )
                
                # Remove the invalid responses
                self.ema_data = self.ema_data[~invalid_time_mask]

    def create_daily_summaries(self) -> pd.DataFrame:
        """Create daily summaries with quality metrics and EMA matching"""
        self.logger.info("Creating daily summaries with EMA matching...")
        daily_summaries = []
        
        # Create set of valid user-date combinations from EMA data
        valid_days = set(
            zip(self.ema_data['user'], self.ema_data['date'])
        )
        
        self.logger.info(f"Found {len(valid_days)} unique user-date combinations in EMA data")
        
        # Group by user and date
        skipped_count = 0
        for (user, date), day_data in tqdm(self.raw_gps.groupby(['user', 'date'])):
            # Skip if no matching EMA data
            if (user, date) not in valid_days:
                skipped_count += 1
                continue
                
            # Calculate day summary
            day_summary = {
                'user': user,
                'date': date,
                'first_reading': day_data['Timestamp'].min(),
                'last_reading': day_data['Timestamp'].max(),
                'total_readings': len(day_data),
                'has_morning_data': any(day_data['Timestamp'].dt.hour < 12),
                'has_evening_data': any(day_data['Timestamp'].dt.hour >= 17),
                'max_gap_minutes': (
                    day_data['Timestamp'].diff().max().total_seconds() / 60 
                    if len(day_data) > 1 else np.nan
                ),
                'coverage_hours': (
                    day_data['Timestamp'].max() - day_data['Timestamp'].min()
                ).total_seconds() / 3600
            }
            
            # Get matching EMA data
            ema_match = self.ema_data[
                (self.ema_data['user'] == user) & 
                (self.ema_data['date'] == date)
            ].iloc[0]
            
            # Add EMA timestamp information
            day_summary['ema_timestamp'] = ema_match['timestamp']
            day_summary['ema_hour'] = ema_match['timestamp'].hour
            
            # Add relevant EMA data
            ema_cols = ['Gender', 'School', 'Class', 'PEACE', 'TENSE', 'IRRITATION', 
                       'RELAXATION', 'SATISFACTION', 'WORRY', 'HAPPY']
            for col in ema_cols:
                if col in self.ema_data.columns:
                    day_summary[col] = ema_match[col]
            
            day_summary['has_ema'] = True
            
            daily_summaries.append(day_summary)
        
        self.logger.info(f"Skipped {skipped_count} days without matching EMA data")
        
        # Convert to DataFrame
        gps_summary_df = pd.DataFrame(daily_summaries)
        
        # Add quality metrics
        gps_summary_df['data_quality'] = np.where(
            (gps_summary_df['has_morning_data']) & 
            (gps_summary_df['has_evening_data']) & 
            (gps_summary_df['coverage_hours'] >= 5),
            'good',
            'partial'
        )
        
        return gps_summary_df

    def create_preprocessed_files(self, gps_summary_df: pd.DataFrame):
        """Create preprocessed GPS files for days with good data and EMA matches"""
        self.logger.info("Creating preprocessed GPS files...")
        
        # Create output directory for preprocessed files
        preprocess_dir = self.output_dir / 'preprocessed_data'
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        
        # Get valid user-date combinations
        valid_days = gps_summary_df[
            (gps_summary_df['data_quality'] == 'good') & 
            (gps_summary_df['has_ema'])
        ]
        
        processed_count = 0
        for _, row in tqdm(valid_days.iterrows()):
            # Get day's data
            day_data = self.raw_gps[
                (self.raw_gps['user'] == row['user']) & 
                (self.raw_gps['date'] == row['date'])
            ].copy()
            
            if not day_data.empty:
                # Add EMA data
                ema_cols = [col for col in gps_summary_df.columns 
                           if col not in day_data.columns and
                           col not in ['data_quality', 'has_morning_data', 'has_evening_data']]
                for col in ema_cols:
                    day_data[col] = row[col]
                
                # Save preprocessed file
                output_file = preprocess_dir / f"{row['date']}_{row['user']}.csv"
                day_data.to_csv(output_file, index=False)
                processed_count += 1
        
        self.logger.info(f"Created {processed_count} preprocessed files")

    def save_summaries(self, gps_summary_df: pd.DataFrame):
        """Save summary files"""
        # Save main summary
        summary_path = self.output_dir / 'gps_daily_summary.csv'
        gps_summary_df.to_csv(summary_path, index=False)
        
        # Log summary statistics
        self.logger.info("\nSummary Statistics:")
        self.logger.info(f"Total days: {len(gps_summary_df)}")
        self.logger.info(f"Days with good quality: {(gps_summary_df['data_quality'] == 'good').sum()}")
        self.logger.info(f"Days with EMA data: {gps_summary_df['has_ema'].sum()}")
        self.logger.info(f"Unique participants: {gps_summary_df['user'].nunique()}")
        
        # Add EMA timing statistics
        self.logger.info("\nEMA Timing Distribution:")
        hour_counts = gps_summary_df['ema_hour'].value_counts().sort_index()
        for hour, count in hour_counts.items():
            self.logger.info(f"  {hour:02d}:00-{hour:02d}:59: {count} responses")
        
        # Get demographic breakdowns
        for col in ['Gender', 'School', 'Class']:
            if col in gps_summary_df.columns:
                self.logger.info(f"\n{col} Distribution:")
                counts = gps_summary_df[col].value_counts()
                for val, count in counts.items():
                    self.logger.info(f"  {val}: {count}")

    def process(self):
        """Run the complete preprocessing pipeline"""
        try:
            # Load all required data
            self.load_data()
            
            # Create daily summaries
            gps_summary_df = self.create_daily_summaries()
            
            # Create preprocessed files
            self.create_preprocessed_files(gps_summary_df)
            
            # Save summary files
            self.save_summaries(gps_summary_df)
            
            return gps_summary_df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

def main():
    # Define paths
    RAW_GPS_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/csv/gpsappS_9.1_excel.csv'
    EMA_PATH = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/csv/End_of_the_day_questionnaire.csv'
    OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/preprocessed_summaries'
    
    # Initialize and run preprocessor
    preprocessor = GPSPreprocessor(
        raw_gps_path=RAW_GPS_PATH,
        ema_path=EMA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    gps_summary_df = preprocessor.process()
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    main()