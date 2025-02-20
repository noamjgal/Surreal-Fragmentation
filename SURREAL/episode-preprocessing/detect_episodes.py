#!/usr/bin/env python3
"""
Enhanced episode detection with separate processing for movement and digital states
Organized by day with comprehensive logging and validation
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import traceback
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('episode_detection.log'),
        logging.StreamHandler()
    ]
)

# Add parent directory to path to find config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import GPS_PREP_DIR, EPISODE_OUTPUT_DIR, PROCESSED_DATA_DIR

# Configuration
MOVEMENT_CUTOFF = 1.5  # m/s (stationary vs moving)
MIN_EPISODE_DURATION = '30s'
MAX_SCREEN_GAP = '5min'
DIGITAL_USE_COL = 'action'

class EpisodeProcessor:
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.logger = logging.getLogger(f"EpisodeProcessor_{participant_id}")
        self.output_dir = EPISODE_OUTPUT_DIR / participant_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_gps_data(self) -> pd.DataFrame:
        """Load GPS data with validation"""
        gps_path = GPS_PREP_DIR / f'{self.participant_id}_qstarz_prep.csv'
        self.logger.info(f"Loading GPS data from {gps_path}")
        
        try:
            gps_df = pd.read_csv(gps_path, parse_dates=['UTC DATE TIME'])
            gps_df['date'] = gps_df['UTC DATE TIME'].dt.date
            self.logger.info(f"Loaded {len(gps_df)} GPS points")
            return gps_df
        except Exception as e:
            self.logger.error(f"Failed to load GPS data: {str(e)}")
            raise

    def load_app_data(self) -> pd.DataFrame:
        """Load app data with validation"""
        app_path = GPS_PREP_DIR / f'{self.participant_id}_app_prep.csv'
        self.logger.info(f"Loading app data from {app_path}")
        
        try:
            app_df = pd.read_csv(app_path)
            
            # Handle timestamp column
            if 'Timestamp' in app_df.columns:
                app_df['timestamp'] = pd.to_datetime(app_df['Timestamp'])
            else:
                app_df['timestamp'] = pd.to_datetime(app_df['date'] + ' ' + app_df['time'], 
                                                   format='mixed', 
                                                   dayfirst=True)
            
            app_df['date'] = app_df['timestamp'].dt.date
            self.logger.info(f"Loaded {len(app_df)} app events")
            return app_df
        except Exception as e:
            self.logger.error(f"Failed to load app data: {str(e)}")
            raise

    def process_digital_episodes(self, app_df: pd.DataFrame) -> Dict[datetime.date, pd.DataFrame]:
        """Process digital episodes by day"""
        episodes_by_day = {}
        
        for date, day_data in app_df.groupby('date'):
            self.logger.info(f"Processing digital episodes for {date}")
            
            # Filter screen events
            screen_events = day_data[day_data[DIGITAL_USE_COL].isin(['SCREEN ON', 'SCREEN OFF'])].copy()
            screen_events = screen_events.sort_values('timestamp')
            
            if len(screen_events) == 0:
                self.logger.warning(f"No screen events found for {date}")
                continue
                
            # Process screen events into episodes
            episodes = []
            current_on = None
            
            for _, row in screen_events.iterrows():
                if row[DIGITAL_USE_COL] == 'SCREEN ON' and not current_on:
                    current_on = row['timestamp']
                elif row[DIGITAL_USE_COL] == 'SCREEN OFF' and current_on:
                    episodes.append({
                        'start_time': current_on,
                        'end_time': row['timestamp'],
                        'state': 'digital'
                    })
                    current_on = None
            
            if len(episodes) == 0:
                self.logger.warning(f"No digital episodes detected for {date}")
            else:
                episodes_by_day[date] = pd.DataFrame(episodes)
                self.logger.info(f"Detected {len(episodes)} digital episodes for {date}")
                
        return episodes_by_day

    def process_movement_episodes(self, gps_df: pd.DataFrame) -> Dict[datetime.date, pd.DataFrame]:
        """Process movement episodes by day"""
        episodes_by_day = {}
        
        for date, day_data in gps_df.groupby('date'):
            self.logger.info(f"Processing movement episodes for {date}")
            
            # Classify movement states
            day_data['state'] = np.where(day_data['SPEED_MS'] > MOVEMENT_CUTOFF, 
                                       'moving', 'stationary')
            
            # Detect state changes
            state_changes = day_data['state'].ne(day_data['state'].shift())
            day_data['episode_id'] = state_changes.cumsum()
            
            # Create episodes
            episodes = day_data.groupby('episode_id').agg({
                'UTC DATE TIME': ['min', 'max'],
                'state': 'first',
                'LATITUDE': 'mean',
                'LONGITUDE': 'mean',
                'episode_id': 'count'
            })
            
            episodes.columns = ['start_time', 'end_time', 'state', 'latitude', 'longitude', 'n_points']
            episodes = episodes.reset_index(drop=True)
            
            # Filter short episodes
            episodes['duration'] = episodes['end_time'] - episodes['start_time']
            episodes = episodes[episodes['duration'] >= pd.Timedelta(MIN_EPISODE_DURATION)]
            
            episodes_by_day[date] = episodes
            self.logger.info(f"Detected {len(episodes)} movement episodes for {date}")
            
        return episodes_by_day

    def save_episodes(self, digital_episodes: Dict, movement_episodes: Dict):
        """Save episodes by day"""
        for date in set(digital_episodes.keys()) | set(movement_episodes.keys()):
            # Save digital episodes
            if date in digital_episodes:
                digital_path = self.output_dir / f'{date}_digital_episodes.csv'
                digital_episodes[date].to_csv(digital_path, index=False)
                self.logger.info(f"Saved digital episodes to {digital_path}")
            
            # Save movement episodes
            if date in movement_episodes:
                movement_path = self.output_dir / f'{date}_movement_episodes.csv'
                movement_episodes[date].to_csv(movement_path, index=False)
                self.logger.info(f"Saved movement episodes to {movement_path}")

    def process(self):
        """Main processing pipeline"""
        try:
            # Load data
            gps_df = self.load_gps_data()
            app_df = self.load_app_data()
            
            # Process episodes
            digital_episodes = self.process_digital_episodes(app_df)
            movement_episodes = self.process_movement_episodes(gps_df)
            
            # Validate coverage
            dates_without_digital = set(movement_episodes.keys()) - set(digital_episodes.keys())
            if dates_without_digital:
                self.logger.warning(f"Dates missing digital episodes: {dates_without_digital}")
            
            # Save results
            self.save_episodes(digital_episodes, movement_episodes)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

def main():
    # Find valid participants
    qstarz_files = {f.stem.replace('_qstarz_prep', ''): f 
                    for f in GPS_PREP_DIR.glob('*_qstarz_prep.csv')}
    app_files = {f.stem.replace('_app_prep', ''): f 
                 for f in GPS_PREP_DIR.glob('*_app_prep.csv')}
    
    common_ids = set(qstarz_files.keys()) & set(app_files.keys())
    logging.info(f"Found {len(common_ids)} participants with complete data")
    
    for pid in common_ids:
        processor = EpisodeProcessor(pid)
        success = processor.process()
        if success:
            logging.info(f"Successfully processed participant {pid}")
        else:
            logging.error(f"Failed to process participant {pid}")

if __name__ == "__main__":
    main()