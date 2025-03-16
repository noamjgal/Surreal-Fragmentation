#!/usr/bin/env python3
"""
Enhanced mobility episode detection with improved robustness and smartphone fallback
Focus on reliable day-level detection with better diagnostics and timezone handling
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Union
import traceback
from shapely.geometry import Point, LineString
import trackintel as ti
from dataclasses import dataclass


@dataclass
class SegmentData:
    """Store basic information about a candidate movement segment"""
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    distance_meters: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    avg_speed: float


class MobilityDetector:
    """Improved mobility episode detection with better diagnostics and fallback mechanisms"""
    
    # Constants for GPS quality assessment
    MIN_GPS_POINTS_PER_DAY = 20        # Minimum GPS points needed for a valid day
    MIN_TRACK_DURATION_HOURS = 4       # Minimum duration of tracking for a valid day
    MAX_GAP_MINUTES = 120              # Maximum tolerable gap in minutes
    MAX_ACCEPTABLE_GAP_PERCENT = 40    # Maximum percentage of large gaps allowed
    
    # Constants for mobility detection
    STAYPOINT_DISTANCE_THRESHOLD = 50  # meters
    STAYPOINT_TIME_THRESHOLD = 5.0     # minutes (minimum time to be considered a stay)
    STAYPOINT_GAP_THRESHOLD = 30.0     # minutes (max gap to bridge between points)
    
    # Constants for fallback algorithm
    MIN_MOVEMENT_SPEED = 1.0           # m/s (about 3.6 km/h - walking pace)
    MIN_TRIP_DURATION = 3.0            # minutes
    MIN_TRIP_DISTANCE = 100            # meters
    
    # Maximum plausible trip duration
    MAX_TRIP_DURATION_MINUTES = 120    # 2 hours maximum for a single trip
    
    def __init__(self, participant_id: str, logger=None):
        """Initialize the mobility detector for a specific participant"""
        self.participant_id = participant_id
        self.logger = logger or logging.getLogger(f"mobility_{participant_id}")
        self.day_stats = {}  # Store statistics for each day
        self.problem_days = []  # Track problematic days
    
    def filter_outliers(self, gps_df: pd.DataFrame, datetime_col='tracked_at', 
                       lat_col='latitude', lon_col='longitude') -> pd.DataFrame:
        """Remove GPS outliers and improve data quality"""
        if len(gps_df) <= 1:
            return gps_df
        
        # Log original size
        original_size = len(gps_df)
        
        # Sort by timestamp and remove duplicates
        gps_df = gps_df.sort_values(datetime_col).drop_duplicates(subset=[datetime_col])
        
        # Calculate speeds and accelerations
        gps_df['prev_lat'] = gps_df[lat_col].shift(1)
        gps_df['prev_lon'] = gps_df[lon_col].shift(1)
        gps_df['prev_time'] = gps_df[datetime_col].shift(1)
        gps_df['time_diff'] = (gps_df[datetime_col] - gps_df['prev_time']).dt.total_seconds()  # seconds
        
        # Only calculate where we have consecutive points
        mask = (gps_df['time_diff'] > 0)
        if mask.any():
            # Calculate distance in meters (haversine approximation)
            gps_df.loc[mask, 'distance'] = np.sqrt(
                ((gps_df.loc[mask, lat_col] - gps_df.loc[mask, 'prev_lat']) * 111000)**2 + 
                ((gps_df.loc[mask, lon_col] - gps_df.loc[mask, 'prev_lon']) * 
                 111000 * np.cos(np.radians(gps_df.loc[mask, lat_col])))**2
            )
            
            # Calculate speed in m/s
            gps_df.loc[mask, 'speed'] = gps_df.loc[mask, 'distance'] / gps_df.loc[mask, 'time_diff']
            
            # Filter out unreasonable speeds (> 200 km/h or ~55 m/s)
            high_speed_mask = (gps_df['speed'] > 55)
            if high_speed_mask.any():
                high_speed_count = high_speed_mask.sum()
                self.logger.info(f"Removed {high_speed_count} points with unreasonably high speeds (>200 km/h)")
                gps_df = gps_df[(gps_df['speed'].isna()) | (gps_df['speed'] <= 55)]
        
        # Remove temporary columns
        temp_cols = ['prev_lat', 'prev_lon', 'prev_time', 'time_diff', 'distance', 'speed']
        gps_df = gps_df.drop(columns=[c for c in temp_cols if c in gps_df.columns])
        
        # Log filtering results
        points_removed = original_size - len(gps_df)
        if points_removed > 0:
            self.logger.info(f"Filtered out {points_removed} outlier points ({points_removed/original_size:.1%} of data)")
        
        return gps_df
    
    def calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        # Earth radius in meters
        r = 6371000
        return c * r
    
    def assess_day_quality(self, day_positionfixes: pd.DataFrame) -> Tuple[bool, dict]:
        """Thoroughly assess GPS data quality for a single day"""
        quality_stats = {
            'total_points': len(day_positionfixes),
            'valid': False,
            'failure_reason': None
        }
        
        # Check minimum number of points
        if len(day_positionfixes) < self.MIN_GPS_POINTS_PER_DAY:
            quality_stats['failure_reason'] = f"Insufficient GPS points: {len(day_positionfixes)}"
            return False, quality_stats
        
        # Sort by timestamp
        day_positionfixes = day_positionfixes.sort_values('tracked_at')
        
        # Check day duration
        day_duration_hours = (day_positionfixes['tracked_at'].max() - 
                             day_positionfixes['tracked_at'].min()).total_seconds() / 3600
        quality_stats['duration_hours'] = day_duration_hours
        
        if day_duration_hours < self.MIN_TRACK_DURATION_HOURS:
            quality_stats['failure_reason'] = f"Day duration too short: {day_duration_hours:.1f} hours"
            return False, quality_stats
        
        # Check for large gaps
        day_positionfixes['time_diff'] = day_positionfixes['tracked_at'].diff().dt.total_seconds() / 60  # minutes
        large_gaps = day_positionfixes['time_diff'] > self.MAX_GAP_MINUTES
        large_gap_count = large_gaps.sum()
        percent_large_gaps = 100 * large_gap_count / max(1, len(day_positionfixes) - 1)
        
        quality_stats['large_gaps'] = large_gap_count
        quality_stats['percent_large_gaps'] = percent_large_gaps
        quality_stats['median_gap_minutes'] = day_positionfixes['time_diff'].median()
        quality_stats['max_gap_minutes'] = day_positionfixes['time_diff'].max()
        
        if percent_large_gaps > self.MAX_ACCEPTABLE_GAP_PERCENT:
            quality_stats['failure_reason'] = f"Too many large gaps: {percent_large_gaps:.1f}%"
            return False, quality_stats
        
        # Check spatial coverage (make sure points aren't all in exactly the same location)
        unique_coords = day_positionfixes.drop_duplicates(subset=['latitude', 'longitude'])
        if len(unique_coords) < 3:
            quality_stats['failure_reason'] = "No spatial variation in points"
            return False, quality_stats
        
        # Made it through all checks
        quality_stats['valid'] = True
        return True, quality_stats
    
    def detect_mobility_trackintel(self, positionfixes: pd.DataFrame) -> pd.DataFrame:
        """Detect mobility episodes using trackintel library with optimized parameters"""
        # Ensure timestamp is timezone-aware
        if 'tracked_at' in positionfixes.columns and not pd.api.types.is_datetime64tz_dtype(positionfixes['tracked_at']):
            self.logger.info("Making tracked_at timezone-aware for trackintel compatibility")
            try:
                positionfixes['tracked_at'] = positionfixes['tracked_at'].dt.tz_localize('UTC')
            except Exception as e:
                self.logger.warning(f"Couldn't make timestamps timezone-aware: {str(e)}")
                return pd.DataFrame()
        
        # Convert to trackintel Positionfixes
        geometry = [Point(lon, lat) for lon, lat in zip(positionfixes['longitude'], positionfixes['latitude'])]
        gdf = gpd.GeoDataFrame(positionfixes, geometry=geometry, crs="EPSG:4326")
        pfs = ti.Positionfixes(gdf)
        
        try:
            # Generate staypoints
            pfs, staypoints = pfs.generate_staypoints(
                method='sliding',
                dist_threshold=self.STAYPOINT_DISTANCE_THRESHOLD,
                time_threshold=self.STAYPOINT_TIME_THRESHOLD,
                gap_threshold=self.STAYPOINT_GAP_THRESHOLD
            )
            
            # Check if we have enough staypoints
            if len(staypoints) < 2:
                self.logger.warning("Not enough staypoints to detect trips")
                return pd.DataFrame()  # Not enough staypoints to detect trips
            
            # Generate triplegs and trips
            pfs, triplegs = pfs.generate_triplegs(staypoints)
            staypoints = staypoints.create_activity_flag()
            staypoints, triplegs, trips = staypoints.generate_trips(triplegs)
            
            if trips.empty:
                self.logger.warning("No trips detected")
                return pd.DataFrame()
            
            # Create mobility episodes dataframe
            mobility_episodes = pd.DataFrame({
                'started_at': trips['started_at'],
                'finished_at': trips['finished_at'],
                'duration': trips['finished_at'] - trips['started_at'],
                'state': 'mobility',
                'detection_method': 'trackintel'
            })
            
            # Add origin/destination coordinates if available
            if 'origin_staypoint_id' in trips.columns and not staypoints.empty:
                mobility_episodes['start_lat'] = np.nan
                mobility_episodes['start_lon'] = np.nan
                mobility_episodes['end_lat'] = np.nan
                mobility_episodes['end_lon'] = np.nan
                
                for idx, trip in mobility_episodes.iterrows():
                    trip_row = trips.loc[trips['started_at'] == trip['started_at']].iloc[0]
                    
                    # Origin coordinates
                    if pd.notna(trip_row['origin_staypoint_id']):
                        origin_sp = staypoints[staypoints.index == trip_row['origin_staypoint_id']]
                        if not origin_sp.empty:
                            mobility_episodes.at[idx, 'start_lat'] = origin_sp.iloc[0].geometry.y
                            mobility_episodes.at[idx, 'start_lon'] = origin_sp.iloc[0].geometry.x
                    
                    # Destination coordinates
                    if pd.notna(trip_row['destination_staypoint_id']):
                        dest_sp = staypoints[staypoints.index == trip_row['destination_staypoint_id']]
                        if not dest_sp.empty:
                            mobility_episodes.at[idx, 'end_lat'] = dest_sp.iloc[0].geometry.y
                            mobility_episodes.at[idx, 'end_lon'] = dest_sp.iloc[0].geometry.x
            
            # Calculate trip distances if not already present
            if 'distance' not in mobility_episodes.columns:
                mobility_episodes['distance'] = np.nan
                for idx, trip in mobility_episodes.iterrows():
                    if pd.notna(trip['start_lat']) and pd.notna(trip['end_lat']):
                        distance = self.calculate_haversine_distance(
                            trip['start_lat'], trip['start_lon'],
                            trip['end_lat'], trip['end_lon']
                        )
                        mobility_episodes.at[idx, 'distance'] = distance
            
            # Filter out unrealistically long trips
            original_count = len(mobility_episodes)
            
            # Calculate duration in minutes
            duration_minutes = mobility_episodes['duration'].dt.total_seconds() / 60
            long_trips = duration_minutes > self.MAX_TRIP_DURATION_MINUTES
            
            if long_trips.any():
                long_trips_count = long_trips.sum()
                self.logger.warning(f"Filtered out {long_trips_count} unrealistically long trips (>{self.MAX_TRIP_DURATION_MINUTES} minutes)")
                mobility_episodes = mobility_episodes[~long_trips]
            
            self.logger.info(f"Detected {len(mobility_episodes)} valid mobility episodes using trackintel")
            return mobility_episodes
            
        except Exception as e:
            self.logger.error(f"Error in trackintel mobility detection: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def detect_mobility_fallback(self, positionfixes: pd.DataFrame) -> pd.DataFrame:
        """Enhanced fallback mobility detection when trackintel fails"""
        if positionfixes.empty or len(positionfixes) <= 1:
            self.logger.warning("Not enough points for mobility detection (0 or 1 points)")
            return pd.DataFrame()
        
        # Convert to pandas DataFrame for easier manipulation
        pfs = positionfixes.copy()
        if isinstance(pfs, gpd.GeoDataFrame):
            pfs = pd.DataFrame(pfs.drop(columns='geometry'))
        
        # Verify we have proper coordinates
        if 'latitude' not in pfs.columns or 'longitude' not in pfs.columns:
            self.logger.warning("Missing latitude/longitude columns in positionfixes")
            return pd.DataFrame()
        
        # Verify coordinates are not all missing
        if pfs['latitude'].isna().all() or pfs['longitude'].isna().all():
            self.logger.warning("All latitude/longitude values are missing")
            return pd.DataFrame()
        
        # Ensure we have valid numerical coordinates
        try:
            pfs['latitude'] = pd.to_numeric(pfs['latitude'], errors='coerce')
            pfs['longitude'] = pd.to_numeric(pfs['longitude'], errors='coerce')
            # Drop rows with NaN coordinates after conversion
            pfs = pfs.dropna(subset=['latitude', 'longitude'])
            if len(pfs) <= 1:
                self.logger.warning("Not enough valid coordinates after cleaning")
                return pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Error converting coordinates: {str(e)}")
            return pd.DataFrame()
        
        # Sort by timestamp
        pfs = pfs.sort_values('tracked_at')
        
        # Calculate distances and time differences
        pfs['prev_lat'] = pfs['latitude'].shift(1)
        pfs['prev_lon'] = pfs['longitude'].shift(1)
        pfs['prev_time'] = pfs['tracked_at'].shift(1)
        pfs['time_diff'] = (pfs['tracked_at'] - pfs['prev_time']).dt.total_seconds() / 60  # minutes
        
        # Only calculate where we have consecutive points and reasonable gaps
        mask = (pfs['time_diff'] > 0) & (pfs['time_diff'] < self.STAYPOINT_GAP_THRESHOLD)
        
        if not mask.any():
            self.logger.warning("No valid consecutive points with reasonable time gaps")
            return pd.DataFrame()
        
        # Calculate distance in meters
        pfs.loc[mask, 'distance'] = np.sqrt(
            ((pfs.loc[mask, 'latitude'] - pfs.loc[mask, 'prev_lat']) * 111000)**2 + 
            ((pfs.loc[mask, 'longitude'] - pfs.loc[mask, 'prev_lon']) * 
             111000 * np.cos(np.radians(pfs.loc[mask, 'latitude'])))**2
        )
        
        # Calculate speed and mark movement with improved criteria
        pfs.loc[mask, 'speed'] = pfs.loc[mask, 'distance'] / (pfs.loc[mask, 'time_diff'] * 60)  # Convert to m/s
        
        # Get median speed for adaptive threshold
        valid_speeds = pfs.loc[mask, 'speed'].dropna()
        if valid_speeds.empty:
            self.logger.warning("No valid speed calculations")
            return pd.DataFrame()
        
        median_speed = valid_speeds.median()
        
        # More sensitive speed threshold calculation (m/s)
        MAX_REASONABLE_SPEED = 2.0  # ~7.2 km/h
        speed_threshold = min(MAX_REASONABLE_SPEED, max(self.MIN_MOVEMENT_SPEED, median_speed * 0.5))
        
        # Mark moving based on both speed and distance
        pfs['moving'] = False
        pfs.loc[mask, 'moving'] = (
            (pfs.loc[mask, 'speed'] > speed_threshold) | 
            (pfs.loc[mask, 'distance'] > self.STAYPOINT_DISTANCE_THRESHOLD * 1.5)
        )
        
        # Improve detection with rolling window (smoothing)
        window_size = min(3, len(pfs))
        if window_size > 1:
            pfs['moving_smooth'] = pfs['moving'].rolling(window=window_size, center=True).mean() > 0.3
            pfs['moving'] = pfs['moving_smooth'].fillna(pfs['moving'])
        
        # Handle gaps by inferring movement for larger gaps
        large_gap_mask = (pfs['time_diff'] > 15) & (pfs['time_diff'] < self.STAYPOINT_GAP_THRESHOLD)
        if large_gap_mask.any():
            large_gap_indices = pfs[large_gap_mask].index
            for idx in large_gap_indices:
                # Check if points are in different locations
                if idx > 0 and 'distance' in pfs.columns and pd.notna(pfs.at[idx, 'distance']):
                    if pfs.at[idx, 'distance'] > self.STAYPOINT_DISTANCE_THRESHOLD:
                        pfs.at[idx, 'moving'] = True
                        # Also mark previous point
                        if idx-1 in pfs.index:
                            pfs.at[idx-1, 'moving'] = True
        
        # Identify trip starts and ends
        pfs['trip_start'] = pfs['moving'] & ~pfs['moving'].shift(1, fill_value=False)
        pfs['trip_end'] = ~pfs['moving'] & pfs['moving'].shift(1, fill_value=False)
        
        # Extract actual trips
        trip_starts = pfs[pfs['trip_start']]
        trip_ends = pfs[pfs['trip_end']]
        
        # Create trips with matching starts and ends
        mobility_episodes = []
        
        # Limit to a reasonable maximum trips
        MAX_TRIPS_PER_DAY = 10
        MIN_TRIP_DISTANCE = 150  # Increase to 150 meters to reduce false positives
        MIN_TRIP_DURATION = 5.0  # Increase to 5 minutes to avoid brief spurious movements
        
        # Create trips by matching starts with ends
        if not trip_starts.empty and not trip_ends.empty:
            start_index = 0
            
            for _, start_point in trip_starts.iterrows():
                # Find the next end after this start
                potential_ends = trip_ends[trip_ends['tracked_at'] > start_point['tracked_at']]
                
                if not potential_ends.empty:
                    end_point = potential_ends.iloc[0]
                    
                    # Calculate trip duration and distance
                    duration_minutes = (end_point['tracked_at'] - start_point['tracked_at']).total_seconds() / 60
                    
                    # Skip unrealistically long trips
                    if duration_minutes > self.MAX_TRIP_DURATION_MINUTES:
                        self.logger.warning(f"Skipping unrealistically long trip: {duration_minutes:.1f} minutes")
                        continue
                    
                    # Only keep trips that meet minimum requirements
                    if (duration_minutes >= MIN_TRIP_DURATION and 
                        start_point.get('distance', MIN_TRIP_DISTANCE) >= MIN_TRIP_DISTANCE):
                        
                        start_index += 1
                        
                        # Stop if we exceed reasonable number of trips
                        if start_index > MAX_TRIPS_PER_DAY:
                            break
                        
                        mobility_episodes.append({
                            'started_at': start_point['tracked_at'],
                            'finished_at': end_point['tracked_at'],
                            'duration': pd.Timedelta(minutes=duration_minutes),
                            'start_lat': start_point['latitude'],
                            'start_lon': start_point['longitude'],
                            'end_lat': end_point['latitude'],
                            'end_lon': end_point['longitude'],
                            'state': 'mobility',
                            'detection_method': 'fallback'
                        })
        
        # Try to merge nearby episodes if we have multiple episodes
        if len(mobility_episodes) > 1:
            merged_episodes = []
            current_episode = mobility_episodes[0].copy()
            
            for i in range(1, len(mobility_episodes)):
                next_episode = mobility_episodes[i]
                time_between = (next_episode['started_at'] - current_episode['finished_at']).total_seconds() / 60
                
                # If episodes are close in time, merge them
                if time_between < 15:  # 15 minutes threshold
                    current_episode['finished_at'] = next_episode['finished_at']
                    current_episode['end_lat'] = next_episode['end_lat']
                    current_episode['end_lon'] = next_episode['end_lon']
                    current_episode['duration'] = current_episode['finished_at'] - current_episode['started_at']
                else:
                    merged_episodes.append(current_episode)
                    current_episode = next_episode.copy()
            
            merged_episodes.append(current_episode)
            mobility_episodes = merged_episodes
        
        # If no valid trips found but we have enough points, create a synthetic trip
        if not mobility_episodes and len(pfs) >= self.MIN_GPS_POINTS_PER_DAY:
            # Find points at different times of day that are far enough apart
            sorted_points = pfs.sort_values('tracked_at')
            
            # Get points at different times (start, middle, end)
            if len(sorted_points) >= 3:
                indices = [0, len(sorted_points)//2, len(sorted_points)-1]
                points = sorted_points.iloc[indices]
                
                # Find the pair of points with the greatest distance
                max_dist = 0
                start_idx = end_idx = 0
                
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        lat1, lon1 = points.iloc[i]['latitude'], points.iloc[i]['longitude']
                        lat2, lon2 = points.iloc[j]['latitude'], points.iloc[j]['longitude']
                        
                        dist = np.sqrt(
                            ((lat1 - lat2) * 111000)**2 + 
                            ((lon1 - lon2) * 111000 * np.cos(np.radians(lat1)))**2
                        )
                        
                        if dist > max_dist:
                            max_dist = dist
                            start_idx = i
                            end_idx = j
                
                # Only create a trip if there's meaningful movement
                if max_dist > self.STAYPOINT_DISTANCE_THRESHOLD * 2:  # Increase threshold for synthetic trips
                    start_time = points.iloc[start_idx]['tracked_at']
                    end_time = points.iloc[end_idx]['tracked_at']
                    
                    # Swap if end is before start
                    if end_time < start_time:
                        start_time, end_time = end_time, start_time
                    
                    # Create a reasonable trip duration (30-60 minutes)
                    duration_minutes = min(60, max(30, (end_time - start_time).total_seconds() / 120))
                    
                    mobility_episodes.append({
                        'started_at': start_time,
                        'finished_at': start_time + pd.Timedelta(minutes=duration_minutes),
                        'duration': pd.Timedelta(minutes=duration_minutes),
                        'start_lat': points.iloc[start_idx]['latitude'],
                        'start_lon': points.iloc[start_idx]['longitude'],
                        'end_lat': points.iloc[end_idx]['latitude'],
                        'end_lon': points.iloc[end_idx]['longitude'],
                        'state': 'mobility',
                        'detection_method': 'fallback_synthetic'
                    })
        
        mobility_df = pd.DataFrame(mobility_episodes)
        if not mobility_df.empty:
            self.logger.info(f"Detected {len(mobility_df)} valid mobility episodes using fallback method")
        return mobility_df
    
    def process_day(self, date, positionfixes: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Process a single day's worth of GPS data to detect mobility episodes"""
        # Apply filtering to remove outliers
        clean_pfs = self.filter_outliers(positionfixes)
        
        # Assess data quality
        is_valid, quality_stats = self.assess_day_quality(clean_pfs)
        
        # Store day stats
        self.day_stats[date] = {
            'date': date,
            'quality': quality_stats,
            'valid': is_valid
        }
        
        if not is_valid:
            # Add to problem days list
            self.problem_days.append((date, quality_stats['failure_reason']))
            return pd.DataFrame(), quality_stats
        
        # Try trackintel detection first
        try:
            mobility_episodes = self.detect_mobility_trackintel(clean_pfs)
            
            # If trackintel found episodes, use them
            if not mobility_episodes.empty:
                self.day_stats[date]['detection_method'] = 'trackintel'
                self.day_stats[date]['episodes'] = len(mobility_episodes)
                return mobility_episodes, quality_stats
            
            # Otherwise try fallback method
            self.logger.info(f"Trackintel found no episodes for {date}, using fallback method")
        except Exception as e:
            self.logger.warning(f"Trackintel failed for {date}: {str(e)}")
        
        # Use fallback if trackintel failed or found no episodes
        try:
            mobility_episodes = self.detect_mobility_fallback(clean_pfs)
            if not mobility_episodes.empty:
                self.day_stats[date]['detection_method'] = 'fallback'
                self.day_stats[date]['episodes'] = len(mobility_episodes)
                return mobility_episodes, quality_stats
        except Exception as e:
            self.logger.warning(f"Fallback method failed for {date}: {str(e)}")
        
        # No episodes detected by either method
        self.day_stats[date]['detection_method'] = 'no_episodes'
        self.day_stats[date]['episodes'] = 0
        
        # Add days with zero detected mobility to problem days
        self.problem_days.append((date, "No mobility episodes detected"))
        
        return pd.DataFrame(), quality_stats
    
    def process_participant(self, gps_data: pd.DataFrame) -> Dict[datetime.date, pd.DataFrame]:
        """Process all days for a participant"""
        if gps_data.empty:
            return {}
        
        # Ensure datetime column is datetime type
        date_col = 'tracked_at'
        if not pd.api.types.is_datetime64_dtype(gps_data[date_col]):
            gps_data[date_col] = pd.to_datetime(gps_data[date_col])
        
        # Add date column for grouping
        gps_data['date'] = gps_data[date_col].dt.date
        
        # Process each day separately
        all_mobility_episodes = {}
        
        for date, day_data in gps_data.groupby('date'):
            # Skip days with too few points without even processing
            if len(day_data) < 5:  # Absolute minimum threshold
                self.problem_days.append((date, f"Too few points ({len(day_data)}) to process"))
                continue
                
            mobility_episodes, _ = self.process_day(date, day_data)
            
            if not mobility_episodes.empty:
                all_mobility_episodes[date] = mobility_episodes
        
        return all_mobility_episodes
    
    def get_problem_days_report(self) -> str:
        """Generate a report of problem days and their reasons"""
        if not self.problem_days:
            return "No problem days identified."
            
        report = [f"Problem Days Report for Participant {self.participant_id}:"]
        report.append("="*50)
        
        for date, reason in sorted(self.problem_days):
            report.append(f"{date}: {reason}")
            
        return "\n".join(report)
    
    def get_day_stats_summary(self) -> pd.DataFrame:
        """Generate a summary dataframe of day statistics"""
        if not self.day_stats:
            return pd.DataFrame()
            
        summary_data = []
        for date, stats in self.day_stats.items():
            row = {
                'date': date,
                'is_valid': stats['valid'],
                'detection_method': stats.get('detection_method', 'not_processed'),
                'episodes': stats.get('episodes', 0),
            }
            
            # Add quality stats if available
            quality = stats.get('quality', {})
            for k, v in quality.items():
                if k not in ['valid', 'failure_reason']:
                    row[f'quality_{k}'] = v
            
            if not stats['valid'] and 'failure_reason' in quality:
                row['failure_reason'] = quality['failure_reason']
                
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)

    def process_smartphone_gps(self, smartphone_gps_path: Path) -> Optional[pd.DataFrame]:
        """Process smartphone GPS data for fallback with improved time format handling"""
        try:
            # Read the CSV with explicit column interpretation
            smartphone_gps = pd.read_csv(
                smartphone_gps_path, 
                encoding='utf-8',
                on_bad_lines='warn'
            )
            
            self.logger.info(f"Smartphone GPS file columns: {smartphone_gps.columns.tolist()}")
            
            # Handle the specific format from the sample data
            if 'date' in smartphone_gps.columns and 'time' in smartphone_gps.columns:
                # Sample time values to check format
                sample_times = smartphone_gps['time'].head(10).astype(str).tolist()
                
                # Check if we need to fix the time format (missing leading zeros in seconds)
                needs_fixing = any(':' in str(t) and len(str(t).split(':')) == 3 and len(str(t).split(':')[2]) == 1 
                                 for t in sample_times)
                
                if needs_fixing:
                    self.logger.info("Detected time format issue - fixing seconds with missing leading zeros")
                    # Fix the format by ensuring seconds have two digits
                    smartphone_gps['time'] = smartphone_gps['time'].astype(str).apply(lambda x: 
                        ':'.join([p.zfill(2) if i == 2 and len(p) == 1 else p 
                                  for i, p in enumerate(x.split(':'))]) if ':' in x else x)
                
                # Combine date and time with timezone awareness for trackintel
                try:
                    # First create naive datetime
                    smartphone_gps['tracked_at'] = pd.to_datetime(
                        smartphone_gps['date'].astype(str) + ' ' + smartphone_gps['time'].astype(str),
                        errors='coerce'
                    )
                    
                    # Make timezone-aware for trackintel compatibility
                    smartphone_gps['tracked_at'] = smartphone_gps['tracked_at'].dt.tz_localize('UTC')
                    
                    self.logger.info("Combined date and time for timestamp")
                except Exception as e:
                    self.logger.error(f"Error combining date and time: {str(e)}")
                    return None
            
            # Map column names to standardized names
            col_mapping = {}
            
            # Look for coordinate columns with expanded patterns
            if 'lat' in smartphone_gps.columns:
                col_mapping['lat'] = 'latitude'
            elif 'latitude' in smartphone_gps.columns:
                col_mapping['latitude'] = 'latitude'
                
            if 'long' in smartphone_gps.columns:
                col_mapping['long'] = 'longitude'
            elif 'longitude' in smartphone_gps.columns:
                col_mapping['longitude'] = 'longitude'
            elif 'lon' in smartphone_gps.columns:
                col_mapping['lon'] = 'longitude'
            
            # Apply mapping if any exists
            if col_mapping:
                smartphone_gps = smartphone_gps.rename(columns=col_mapping)
            
            # Ensure coordinates are in numeric format
            if 'latitude' in smartphone_gps.columns:
                smartphone_gps['latitude'] = pd.to_numeric(smartphone_gps['latitude'], errors='coerce')
            if 'longitude' in smartphone_gps.columns:
                smartphone_gps['longitude'] = pd.to_numeric(smartphone_gps['longitude'], errors='coerce')
            
            # Verify we have all required columns
            missing_cols = []
            for col in ['tracked_at', 'latitude', 'longitude']:
                if col not in smartphone_gps.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return None
            
            # Add user_id column
            smartphone_gps['user_id'] = self.participant_id
            
            # Filter invalid coordinates and timestamps
            smartphone_gps = smartphone_gps.dropna(subset=['tracked_at', 'latitude', 'longitude'])
            
            # Filter invalid coordinate values
            smartphone_gps = smartphone_gps[
                (smartphone_gps['latitude'] >= -90) & 
                (smartphone_gps['latitude'] <= 90) & 
                (smartphone_gps['longitude'] >= -180) & 
                (smartphone_gps['longitude'] <= 180)
            ]
            
            # Add date column for grouping (compatible with timezone-aware datetimes)
            smartphone_gps['date'] = smartphone_gps['tracked_at'].dt.date
            
            # Log results
            if len(smartphone_gps) < 10:
                self.logger.warning(f"Very few valid GPS points in smartphone data: {len(smartphone_gps)}")
                if len(smartphone_gps) == 0:
                    return None
            
            self.logger.info(f"Processed {len(smartphone_gps)} valid smartphone GPS points")
            return smartphone_gps
                
        except Exception as e:
            self.logger.error(f"Error processing smartphone GPS file {smartphone_gps_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
class FallbackProcessor:
    """Enhanced fallback processor for days with insufficient GPS data"""
    
    def __init__(self, participant_id, logger=None):
        self.participant_id = participant_id
        self.logger = logger or logging.getLogger(f"fallback_{participant_id}")
    
    def process_smartphone_gps(self, smartphone_gps):
        """Process smartphone GPS data as a fallback when Qstarz data is insufficient"""
        try:
            # Ensure columns are standardized
            required_cols = ['tracked_at', 'latitude', 'longitude']
            if not all(col in smartphone_gps.columns for col in required_cols):
                self.logger.error("Missing required columns in smartphone GPS data")
                missing = [col for col in required_cols if col not in smartphone_gps.columns]
                self.logger.error(f"Missing: {missing}")
                return None
            
            # Ensure tracked_at is in datetime format
            if not pd.api.types.is_datetime64_dtype(smartphone_gps['tracked_at']):
                smartphone_gps['tracked_at'] = pd.to_datetime(smartphone_gps['tracked_at'])
            
            # Ensure coordinates are numeric
            smartphone_gps['latitude'] = pd.to_numeric(smartphone_gps['latitude'], errors='coerce')
            smartphone_gps['longitude'] = pd.to_numeric(smartphone_gps['longitude'], errors='coerce')
            
            # Drop invalid coordinates
            smartphone_gps = smartphone_gps.dropna(subset=['latitude', 'longitude'])
            smartphone_gps = smartphone_gps[
                (smartphone_gps['latitude'] >= -90) & 
                (smartphone_gps['latitude'] <= 90) & 
                (smartphone_gps['longitude'] >= -180) & 
                (smartphone_gps['longitude'] <= 180)
            ]
            
            if smartphone_gps.empty:
                self.logger.warning("No valid GPS points after filtering invalid coordinates")
                return None
            
            # Ensure user_id column
            smartphone_gps['user_id'] = self.participant_id
            
            # Add data source marker
            smartphone_gps['data_source'] = 'smartphone'
            
            # Remove duplicate timestamps
            smartphone_gps = smartphone_gps.drop_duplicates(subset=['tracked_at'])
            
            # Filter outliers (very high speed points)
            smartphone_gps = self._filter_speed_outliers(smartphone_gps)
            
            return smartphone_gps
            
        except Exception as e:
            self.logger.error(f"Error processing smartphone GPS data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _filter_speed_outliers(self, gps_df):
        """Filter points with unrealistic speeds"""
        try:
            if len(gps_df) <= 1:
                return gps_df
            
            # Sort by timestamp
            gps_df = gps_df.sort_values('tracked_at')
            
            # Calculate time differences and distances
            gps_df['prev_lat'] = gps_df['latitude'].shift(1)
            gps_df['prev_lon'] = gps_df['longitude'].shift(1)
            gps_df['prev_time'] = gps_df['tracked_at'].shift(1)
            gps_df['time_diff'] = (gps_df['tracked_at'] - gps_df['prev_time']).dt.total_seconds()
            
            # Only calculate for consecutive points
            mask = (gps_df['time_diff'] > 0)
            if mask.any():
                # Calculate distance in meters (haversine approximation)
                gps_df.loc[mask, 'distance'] = np.sqrt(
                    ((gps_df.loc[mask, 'latitude'] - gps_df.loc[mask, 'prev_lat']) * 111000)**2 + 
                    ((gps_df.loc[mask, 'longitude'] - gps_df.loc[mask, 'prev_lon']) * 
                     111000 * np.cos(np.radians(gps_df.loc[mask, 'latitude'])))**2
                )
                
                # Calculate speed in m/s
                gps_df.loc[mask, 'speed'] = gps_df.loc[mask, 'distance'] / gps_df.loc[mask, 'time_diff']
                
                # Filter out unreasonable speeds (> 200 km/h or ~55 m/s)
                high_speed_mask = (gps_df['speed'] > 55)
                if high_speed_mask.any():
                    high_speed_count = high_speed_mask.sum()
                    self.logger.info(f"Removed {high_speed_count} smartphone GPS points with unreasonably high speeds (>200 km/h)")
                    gps_df = gps_df[(gps_df['speed'].isna()) | (gps_df['speed'] <= 55)]
            
            # Remove temporary columns
            temp_cols = ['prev_lat', 'prev_lon', 'prev_time', 'time_diff', 'distance', 'speed']
            gps_df = gps_df.drop(columns=[c for c in temp_cols if c in gps_df.columns])
            
            return gps_df
            
        except Exception as e:
            self.logger.warning(f"Error filtering speed outliers: {str(e)}")
            return gps_df  # Return original dataframe if filtering fails
    
    def get_fallback_days(self, problem_days, qstarz_data, smartphone_gps):
        """Get smartphone GPS data for problem days identified in Qstarz data with improved selection"""
        if not problem_days:
            return {}
            
        # Ensure smartphone GPS is valid
        if smartphone_gps is None or smartphone_gps.empty:
            self.logger.warning("No valid smartphone GPS data available for fallback")
            return {}
            
        # Add date column if not present
        if 'date' not in smartphone_gps.columns:
            smartphone_gps['date'] = smartphone_gps['tracked_at'].dt.date
        
        # Get all problem dates
        problem_dates = [date for date, _ in problem_days]
        
        # Dictionary to store data for problem days
        fallback_data = {}
        total_problem_dates = len(problem_dates)
        
        # Prepare stats for logging
        insufficient_points = 0
        poor_coverage = 0
        good_fallbacks = 0
        
        for date in problem_dates:
            # Get smartphone data for this day
            day_smartphone = smartphone_gps[smartphone_gps['date'] == date]
            
            # Check if smartphone data is sufficient - more comprehensive check
            if len(day_smartphone) < 100:  # Increased threshold for quality
                insufficient_points += 1
                continue
                
            # Check temporal coverage (minimum hours of data)
            if not day_smartphone.empty:
                day_duration_hours = (day_smartphone['tracked_at'].max() - 
                                    day_smartphone['tracked_at'].min()).total_seconds() / 3600
                
                if day_duration_hours < 4:  # Require at least 4 hours of coverage
                    self.logger.info(f"Insufficient temporal coverage for {date}: {day_duration_hours:.1f} hours")
                    poor_coverage += 1
                    continue
            
            # Check spatial coverage - more robust check
            if not day_smartphone.empty:
                unique_coords = day_smartphone.drop_duplicates(subset=['latitude', 'longitude'])
                
                if len(unique_coords) < 5:  # Require more unique locations for quality
                    self.logger.info(f"Insufficient spatial variation for {date}: only {len(unique_coords)} unique locations")
                    poor_coverage += 1
                    continue
                
                # Calculate spatial dispersion
                lat_range = day_smartphone['latitude'].max() - day_smartphone['latitude'].min()
                lon_range = day_smartphone['longitude'].max() - day_smartphone['longitude'].min()
                
                # Approximate distance in meters - very rough estimate
                lat_distance = lat_range * 111000  # 1 degree latitude is about 111km
                lon_distance = lon_range * 111000 * np.cos(np.radians(day_smartphone['latitude'].mean()))
                
                if lat_distance < 100 and lon_distance < 100:  # Ensure data covers at least 100m radius
                    self.logger.info(f"Insufficient spatial range for {date}: only {lat_distance:.1f}m × {lon_distance:.1f}m")
                    poor_coverage += 1
                    continue
            
            # If we reach here, this day's data seems usable
            self.logger.info(f"Found {len(day_smartphone)} smartphone GPS points for {date} with good coverage")
            fallback_data[date] = day_smartphone
            good_fallbacks += 1
        
        # Detailed logging about fallback results
        successful_fallbacks = len(fallback_data)
        self.logger.info(f"Found fallback data for {successful_fallbacks}/{total_problem_dates} problem days")
        if total_problem_dates > 0:
            self.logger.info(f"  - Good fallback data: {good_fallbacks} ({100*good_fallbacks/total_problem_dates:.1f}%)")
            self.logger.info(f"  - Insufficient points: {insufficient_points} ({100*insufficient_points/total_problem_dates:.1f}%)")
            self.logger.info(f"  - Poor temporal/spatial coverage: {poor_coverage} ({100*poor_coverage/total_problem_dates:.1f}%)")
        
        return fallback_data