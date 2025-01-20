import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import os
from collections import Counter

class EpisodeDetector:
    def __init__(self, 
                 digital_settings: Dict = None,
                 moving_settings: Dict = None):
        """Initialize episode detector with configurable settings"""
        self.digital_settings = digital_settings or {
            'min_duration': timedelta(seconds=20),
            'merge_gap': timedelta(minutes=1),
            'max_gap': timedelta(minutes=5)
        }
        
        self.moving_settings = moving_settings or {
            'min_duration': timedelta(minutes=2),
            'merge_gap': timedelta(minutes=1),
            'max_gap': timedelta(minutes=5)
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def process_user_day(self, file_path: str, verbose: bool = False) -> Dict:
        """Process a single user-day from preprocessed GPS data"""
        try:
            # Load and sort data
            df = pd.read_csv(file_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp')
            
            # Extract digital episodes from StartEnd
            digital_episodes = self._extract_digital_episodes(df)
            
            # Extract moving episodes from Travel_mode
            moving_episodes, travel_modes = self._extract_moving_episodes(df)
            
            # Find overlaps
            overlap_episodes = self._find_overlaps(digital_episodes, moving_episodes)
            
            # Get user and date from filename
            filename = Path(file_path).name
            date_str, user_id = filename.split('_')[:2]
            
            if verbose:
                self._log_processing_stats(filename, df, digital_episodes, 
                                        moving_episodes, overlap_episodes,
                                        travel_modes)
            
            return {
                'user': user_id,
                'date': date_str,
                'digital_episodes': digital_episodes,
                'moving_episodes': moving_episodes,
                'overlap_episodes': overlap_episodes,
                'first_timestamp': df['Timestamp'].min(),
                'last_timestamp': df['Timestamp'].max(),
                'total_points': len(df),
                'travel_modes': travel_modes
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def _extract_digital_episodes(self, df: pd.DataFrame) -> List[Tuple]:
        """Extract digital episodes from StartEnd markers and App usage"""
        episodes = []
        start_time = None
        in_digital = False
        
        # Find Start and End events or App usage
        for idx, row in df.iterrows():
            timestamp = row['Timestamp']
            
            # Check primary StartEnd marker
            is_digital = False
            if isinstance(row.get('StartEnd'), str):
                if 'Start' in row['StartEnd']:
                    is_digital = True
                    start_time = timestamp
                elif 'End' in row['StartEnd']:
                    is_digital = False
                    if start_time is not None:
                        duration = timestamp - start_time
                        if duration >= self.digital_settings['min_duration']:
                            episodes.append((start_time, timestamp))
                    start_time = None
            
            # Backup check: App usage
            if isinstance(row.get('App'), str) and row['App'] != 'No use':
                is_digital = True
            
            # Handle transitions based on App usage
            if is_digital and not in_digital:
                start_time = timestamp
            elif not is_digital and in_digital and start_time is not None:
                duration = timestamp - start_time
                if duration >= self.digital_settings['min_duration']:
                    episodes.append((start_time, timestamp))
                start_time = None
            
            in_digital = is_digital
        
        # Handle any unclosed episode
        if start_time is not None:
            duration = df['Timestamp'].iloc[-1] - start_time
            if duration >= self.digital_settings['min_duration']:
                episodes.append((start_time, df['Timestamp'].iloc[-1]))
        
        return self._merge_episodes(episodes, self.digital_settings['merge_gap'])

    def _extract_moving_episodes(self, df: pd.DataFrame) -> Tuple[List[Tuple], Counter]:
        """Extract moving episodes from Travel_mode and speed"""
        episodes = []
        start_time = None
        travel_modes = Counter()
        prev_mode = None
        
        for idx, row in df.iterrows():
            timestamp = row['Timestamp']
            
            # Primary check: Travel_mode
            is_moving = False
            if row['Travel_mode'] != 'Missing':
                is_moving = row['Travel_mode'] != 'Staying'
                if is_moving:
                    travel_modes[row['Travel_mode']] += 1
            
            # Backup checks
            if not is_moving:
                # Check speed > 3 (more conservative threshold)
                if pd.notna(row['speed']) and row['speed'] > 3.0:
                    is_moving = True
                    travel_modes['Speed > 3'] += 1
                # If speed > 2 and Travel_mode was Missing (changed from > 1)
                elif row['Travel_mode'] == 'Missing' and pd.notna(row['speed']) and row['speed'] > 2.0:
                    is_moving = True
                    travel_modes['Speed > 2 (Missing)'] += 1  # Updated counter label
            
            if is_moving and prev_mode != 'moving':
                start_time = timestamp
            elif not is_moving and prev_mode == 'moving' and start_time is not None:
                duration = timestamp - start_time
                if duration >= self.moving_settings['min_duration']:
                    episodes.append((start_time, timestamp))
                start_time = None
            
            prev_mode = 'moving' if is_moving else 'staying'
        
        # Handle ongoing episode at end of day
        if start_time is not None:
            duration = df['Timestamp'].iloc[-1] - start_time
            if duration >= self.moving_settings['min_duration']:
                episodes.append((start_time, df['Timestamp'].iloc[-1]))
        
        return self._merge_episodes(episodes, self.moving_settings['merge_gap']), travel_modes

    def _find_overlaps(self, digital_episodes: List[Tuple], 
                      moving_episodes: List[Tuple]) -> List[Tuple]:
        """Find temporal overlaps between digital and moving episodes"""
        overlap_episodes = []
        
        for d_start, d_end in digital_episodes:
            for m_start, m_end in moving_episodes:
                # Find overlap
                start = max(d_start, m_start)
                end = min(d_end, m_end)
                
                if start < end:  # There is an overlap
                    duration = end - start
                    if duration >= timedelta(minutes=1):
                        overlap_episodes.append((start, end))
        
        return self._merge_episodes(overlap_episodes, timedelta(minutes=1))

    def _merge_episodes(self, episodes: List[Tuple], merge_gap: timedelta) -> List[Tuple]:
        """Merge episodes that are close together"""
        if not episodes:
            return []
        
        # Sort episodes by start time
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

    def _log_processing_stats(self, filename: str, df: pd.DataFrame, 
                            digital_episodes: List, moving_episodes: List,
                            overlap_episodes: List, travel_modes: Counter):
        """Log detailed processing statistics"""
        self.logger.info(f"\nProcessing file: {filename}")
        self.logger.info(f"Time range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        self.logger.info(f"Total points: {len(df)}")
        self.logger.info(f"Digital episodes: {len(digital_episodes)}")
        self.logger.info(f"Moving episodes: {len(moving_episodes)}")
        self.logger.info(f"Overlap episodes: {len(overlap_episodes)}")
        
        # Log travel modes found in mobility episodes
        self.logger.info("\nTravel modes in mobility episodes:")
        for mode, count in travel_modes.most_common():
            self.logger.info(f"  {mode}: {count} points")
        
        # Calculate and log episode statistics
        for ep_type, episodes in [
            ('Digital', digital_episodes),
            ('Moving', moving_episodes),
            ('Overlap', overlap_episodes)
        ]:
            if episodes:
                durations = [(end - start).total_seconds() / 60 for start, end in episodes]
                self.logger.info(f"\n{ep_type} Episode Stats:")
                self.logger.info(f"  Average duration: {np.mean(durations):.1f} minutes")
                self.logger.info(f"  Median duration: {np.median(durations):.1f} minutes")
                self.logger.info(f"  Min duration: {min(durations):.1f} minutes")
                self.logger.info(f"  Max duration: {max(durations):.1f} minutes")

def main():
    # Configure paths
    base_dir = Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon')
    input_dir = base_dir / 'preprocessed_summaries/preprocessed_data'
    output_dir = base_dir / 'episodes'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = EpisodeDetector()
    
    # Get list of all preprocessed files
    preprocessed_files = list(input_dir.glob('*.csv'))
    
    if not preprocessed_files:
        logging.error(f"No CSV files found in {input_dir}")
        return
    
    logging.info(f"Found {len(preprocessed_files)} files to process")
    
    # Process all files
    episode_summaries = []
    all_travel_modes = Counter()
    
    for file_path in tqdm(preprocessed_files, desc="Processing files"):
        day_results = detector.process_user_day(file_path, verbose=True)
        
        if day_results:
            # Update travel modes counter
            all_travel_modes.update(day_results['travel_modes'])
            
            # Process all episode types
            for episode_type in ['digital', 'moving', 'overlap']:
                episodes = day_results[f'{episode_type}_episodes']
                if episodes:
                    episodes_df = pd.DataFrame(episodes, columns=['start_time', 'end_time'])
                    episodes_df['start_time'] = pd.to_datetime(episodes_df['start_time'])
                    episodes_df['end_time'] = pd.to_datetime(episodes_df['end_time'])
                    episodes_df['duration_minutes'] = (
                        episodes_df['end_time'] - episodes_df['start_time']
                    ).dt.total_seconds() / 60
                    
                    episodes_df['user'] = day_results['user']
                    episodes_df['date'] = day_results['date']
                    
                    # Add to summary
                    episode_summaries.append({
                        'user': day_results['user'],
                        'date': day_results['date'],
                        'episode_type': episode_type,
                        'num_episodes': len(episodes_df),
                        'total_duration_minutes': episodes_df['duration_minutes'].sum(),
                        'mean_duration_minutes': episodes_df['duration_minutes'].mean(),
                        'first_timestamp': day_results['first_timestamp'],
                        'last_timestamp': day_results['last_timestamp'],
                        'total_points': day_results['total_points']
                    })
                    
                    # Save episode file
                    output_file = output_dir / f"{episode_type}_episodes_{day_results['date']}_{day_results['user']}.csv"
                    episodes_df.to_csv(output_file, index=False)
    
    # Save summary statistics
    if episode_summaries:
        summary_df = pd.DataFrame(episode_summaries)
        summary_df.to_csv(output_dir / 'episode_summary.csv', index=False)
        
        # Log statistics
        logging.info("\nProcessing Summary:")
        logging.info(f"Total files processed: {len(preprocessed_files)}")
        logging.info(f"Total episodes detected: {len(summary_df)}")
        
        # Log overall travel mode statistics
        logging.info("\nOverall Travel Modes in Mobility Episodes:")
        for mode, count in all_travel_modes.most_common():
            logging.info(f"  {mode}: {count} points")
        
        for episode_type in ['digital', 'moving', 'overlap']:
            type_stats = summary_df[summary_df['episode_type'] == episode_type]
            logging.info(f"\n{episode_type.title()} Episodes:")
            logging.info(f"  Total episodes: {type_stats['num_episodes'].sum()}")
            logging.info(f"  Mean episodes per day: {type_stats['num_episodes'].mean():.1f}")
            logging.info(f"  Mean duration: {type_stats['mean_duration_minutes'].mean():.1f} minutes")

if __name__ == "__main__":
    main()



