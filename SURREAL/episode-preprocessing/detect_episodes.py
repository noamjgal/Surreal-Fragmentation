#!/usr/bin/env python3
"""
Enhanced episode detection with separate processing for movement and digital states
Includes overlap detection and comprehensive statistics reporting
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
from collections import Counter
from tqdm import tqdm

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
MERGE_GAP = timedelta(minutes=1)

class EpisodeProcessor:
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.logger = logging.getLogger(f"EpisodeProcessor_{participant_id}")
        self.output_dir = EPISODE_OUTPUT_DIR / participant_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _merge_episodes(self, episodes: List[Tuple], merge_gap: timedelta) -> List[Tuple]:
        """Merge episodes that are close together"""
        if not episodes:
            return []
        
        episodes = sorted(episodes)
        merged = [episodes[0]]
        
        for current in episodes[1:]:
            prev = merged[-1]
            gap = current[0] - prev[1]
            
            if gap <= merge_gap:
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)
        
        return merged

    def _find_overlaps(self, digital_episodes: pd.DataFrame, 
                      movement_episodes: pd.DataFrame) -> pd.DataFrame:
        """Find temporal overlaps between digital and MOVING episodes only"""
        overlap_episodes = []
        
        # Filter movement episodes to only moving states
        moving_episodes = movement_episodes[movement_episodes['state'] == 'moving']
        
        for _, d_ep in digital_episodes.iterrows():
            for _, m_ep in moving_episodes.iterrows():  # Only consider moving episodes
                start = max(d_ep['start_time'], m_ep['start_time'])
                end = min(d_ep['end_time'], m_ep['end_time'])
                
                if start < end:  # There is an overlap
                    duration = end - start
                    if duration >= pd.Timedelta(minutes=1):
                        overlap_episodes.append({
                            'start_time': start,
                            'end_time': end,
                            'state': 'overlap',
                            'movement_state': m_ep['state'],
                            'latitude': m_ep['latitude'],
                            'longitude': m_ep['longitude'],
                            'duration': duration
                        })
        
        if overlap_episodes:
            return pd.DataFrame(overlap_episodes)
        return pd.DataFrame()

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
            
            screen_events = day_data[day_data[DIGITAL_USE_COL].isin(['SCREEN ON', 'SCREEN OFF'])].copy()
            screen_events = screen_events.sort_values('timestamp')
            
            if len(screen_events) == 0:
                self.logger.warning(f"No screen events found for {date}")
                continue
                
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
                episodes_df = pd.DataFrame(episodes)
                episodes_df['duration'] = episodes_df['end_time'] - episodes_df['start_time']
                episodes_by_day[date] = episodes_df
                self.logger.info(f"Detected {len(episodes)} digital episodes for {date}")
                
        return episodes_by_day

    def process_movement_episodes(self, gps_df: pd.DataFrame) -> Dict[datetime.date, pd.DataFrame]:
        """Process movement episodes by day"""
        episodes_by_day = {}
        
        for date, day_data in gps_df.groupby('date'):
            self.logger.info(f"Processing movement episodes for {date}")
            
            day_data['state'] = np.where(day_data['SPEED_MS'] > MOVEMENT_CUTOFF, 
                                       'moving', 'stationary')
            
            state_changes = day_data['state'].ne(day_data['state'].shift())
            day_data['episode_id'] = state_changes.cumsum()
            
            episodes = day_data.groupby('episode_id').agg({
                'UTC DATE TIME': ['min', 'max'],
                'state': 'first',
                'LATITUDE': 'mean',
                'LONGITUDE': 'mean',
                'episode_id': 'count'
            })
            
            episodes.columns = ['start_time', 'end_time', 'state', 'latitude', 'longitude', 'n_points']
            episodes = episodes.reset_index(drop=True)
            
            episodes['duration'] = episodes['end_time'] - episodes['start_time']
            episodes = episodes[episodes['duration'] >= pd.Timedelta(MIN_EPISODE_DURATION)]
            
            episodes_by_day[date] = episodes
            self.logger.info(f"Detected {len(episodes)} movement episodes for {date}")
            
        return episodes_by_day

    def create_daily_timeline(self, digital_episodes: pd.DataFrame, 
                          movement_episodes: pd.DataFrame,
                          overlap_episodes: pd.DataFrame) -> pd.DataFrame:
        """Create a chronological timeline of all episodes for a day"""
        # Add episode type column to each DataFrame
        if not digital_episodes.empty:
            digital_episodes = digital_episodes.copy()
            digital_episodes['episode_type'] = 'digital'
            digital_episodes['movement_state'] = None
        
        if not movement_episodes.empty:
            movement_episodes = movement_episodes.copy()
            movement_episodes['episode_type'] = 'movement'
            movement_episodes['movement_state'] = movement_episodes['state']
            movement_episodes = movement_episodes.drop(columns=['state'])
        
        if not overlap_episodes.empty:
            overlap_episodes = overlap_episodes.copy()
            overlap_episodes['episode_type'] = 'overlap'
        
        # Combine all episodes
        all_episodes = pd.concat([digital_episodes, movement_episodes, overlap_episodes], 
                               ignore_index=True)
        
        # Sort chronologically
        if not all_episodes.empty:
            all_episodes = all_episodes.sort_values('start_time')
            
            # Add sequential episode number
            all_episodes['episode_number'] = range(1, len(all_episodes) + 1)
            
            # Calculate time since previous episode
            all_episodes['time_since_prev'] = all_episodes['start_time'].diff()
            
            # Ensure consistent column order
            columns = ['episode_number', 'episode_type', 'movement_state', 
                      'start_time', 'end_time', 'duration', 'time_since_prev',
                      'latitude', 'longitude']
            all_episodes = all_episodes[columns]
        
        return all_episodes

    def process_day(self, date: datetime.date, digital_episodes: pd.DataFrame, 
                   movement_episodes: pd.DataFrame) -> dict:
        """Process a single day and generate statistics"""
        overlap_episodes = self._find_overlaps(digital_episodes, movement_episodes)
        
        # Create daily timeline
        daily_timeline = self.create_daily_timeline(digital_episodes, movement_episodes, overlap_episodes)
        
        # Save daily timeline
        if not daily_timeline.empty:
            timeline_file = self.output_dir / f"{date}_daily_timeline.csv"
            daily_timeline.to_csv(timeline_file, index=False)
            self.logger.info(f"Saved daily timeline to {timeline_file}")
        
        # Calculate statistics
        day_stats = {
            'user': self.participant_id,
            'date': date,
            'digital_episodes': len(digital_episodes),
            'movement_episodes': len(movement_episodes),
            'overlap_episodes': len(overlap_episodes),
            'digital_duration': digital_episodes['duration'].sum().total_seconds() / 60,
            'movement_duration': movement_episodes['duration'].sum().total_seconds() / 60,
            'overlap_duration': overlap_episodes['duration'].sum().total_seconds() / 60 if not overlap_episodes.empty else 0,
        }
        
        # Save episodes
        for ep_type, episodes in [
            ('digital', digital_episodes),
            ('movement', movement_episodes),
            ('overlap', overlap_episodes)
        ]:
            if len(episodes) > 0:
                output_file = self.output_dir / f"{date}_{ep_type}_episodes.csv"
                episodes.to_csv(output_file, index=False)
                self.logger.info(f"Saved {ep_type} episodes to {output_file}")
        
        return day_stats

    def process(self) -> List[dict]:
        """Main processing pipeline"""
        try:
            # Load data
            gps_df = self.load_gps_data()
            app_df = self.load_app_data()
            
            # Process episodes
            digital_episodes = self.process_digital_episodes(app_df)
            movement_episodes = self.process_movement_episodes(gps_df)
            
            # Process each day
            all_stats = []
            all_dates = sorted(set(digital_episodes.keys()) | set(movement_episodes.keys()))
            
            for date in all_dates:
                digital_eps = digital_episodes.get(date, pd.DataFrame())
                movement_eps = movement_episodes.get(date, pd.DataFrame())
                
                if len(digital_eps) == 0:
                    self.logger.warning(f"No digital episodes for {date}")
                if len(movement_eps) == 0:
                    self.logger.warning(f"No movement episodes for {date}")
                
                day_stats = self.process_day(date, digital_eps, movement_eps)
                all_stats.append(day_stats)
            
            # Save summary statistics
            summary_df = pd.DataFrame(all_stats)
            summary_file = self.output_dir / 'episode_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            self.logger.info(f"Saved summary statistics to {summary_file}")
            
            return all_stats
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

def main():
    # Find valid participants
    qstarz_files = {f.stem.replace('_qstarz_prep', ''): f 
                    for f in GPS_PREP_DIR.glob('*_qstarz_prep.csv')}
    app_files = {f.stem.replace('_app_prep', ''): f 
                 for f in GPS_PREP_DIR.glob('*_app_prep.csv')}
    
    common_ids = set(qstarz_files.keys()) & set(app_files.keys())
    logging.info(f"Found {len(common_ids)} participants with complete data")
    
    all_stats = []
    
    for pid in tqdm(common_ids, desc="Processing participants"):
        processor = EpisodeProcessor(pid)
        participant_stats = processor.process()
        all_stats.extend(participant_stats)
    
    if all_stats:
        # Create overall summary
        all_summary = pd.DataFrame(all_stats)
        summary_file = EPISODE_OUTPUT_DIR / 'all_participants_summary.csv'
        all_summary.to_csv(summary_file, index=False)
        
        # Log overall statistics
        logging.info("\nOverall Processing Summary:")
        logging.info(f"Total participants processed: {len(common_ids)}")
        logging.info(f"Total days processed: {len(all_summary)}")
        logging.info("\nEpisode Statistics:")
        logging.info(f"Total digital episodes: {all_summary['digital_episodes'].sum()}")
        logging.info(f"Total movement episodes: {all_summary['movement_episodes'].sum()}")
        logging.info(f"Total overlap episodes: {all_summary['overlap_episodes'].sum()}")
        logging.info("\nDuration Statistics (minutes):")
        logging.info(f"Total digital duration: {all_summary['digital_duration'].sum():.1f}")
        logging.info(f"Total movement duration: {all_summary['movement_duration'].sum():.1f}")
        logging.info(f"Total overlap duration: {all_summary['overlap_duration'].sum():.1f}")
        
        # Per-participant statistics
        for pid in common_ids:
            participant_data = all_summary[all_summary['user'] == pid]
            logging.info(f"\nParticipant {pid}:")
            logging.info(f"Days of data: {len(participant_data)}")
            logging.info(f"Average daily digital episodes: {participant_data['digital_episodes'].mean():.1f}")
            logging.info(f"Average daily movement episodes: {participant_data['movement_episodes'].mean():.1f}")
            logging.info(f"Average daily overlap episodes: {participant_data['overlap_episodes'].mean():.1f}")

if __name__ == "__main__":
    main()