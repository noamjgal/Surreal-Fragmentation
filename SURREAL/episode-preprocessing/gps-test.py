#!/usr/bin/env python3
"""
GPS Data Inspector Tool

This tool inspects GPS data at various stages of processing to identify
structural issues and processing failures to facilitate debugging.
"""
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import logging
import traceback
from datetime import datetime, timedelta
import json
import re
import pytz
from typing import Dict, List, Tuple, Optional, Union
import geopandas as gpd
from shapely.geometry import Point
from collections import defaultdict

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'data_inspector.log'),
        logging.StreamHandler()
    ]
)

# Set paths - update these to match your environment
RAW_DATA_DIR = Path("/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/raw")
QSTARZ_DIR = Path("/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/qstarz")
GPS_PREP_DIR = Path("/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/gps_preprocessed")
EPISODE_OUTPUT_DIR = Path("/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/episodes")

# Structure to track issues
class DataIssueTracker:
    def __init__(self):
        self.issues = {
            "raw_files": defaultdict(list),
            "preprocessed_files": defaultdict(list),
            "episode_files": defaultdict(list),
            "datetime_issues": defaultdict(list),
            "data_quality_issues": defaultdict(list),
            "file_access_issues": defaultdict(list),
            "processing_failures": defaultdict(list),
            "summary": {}
        }
        
    def add_issue(self, category: str, participant_id: str, description: str, details: Optional[Dict] = None):
        """Add an issue to the tracker"""
        issue_entry = {
            "description": description,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if details:
            issue_entry["details"] = details
            
        self.issues[category][participant_id].append(issue_entry)
        
    def save_report(self, output_path: Path):
        """Save the issue report as JSON"""
        # Generate summary statistics
        self.issues["summary"] = {
            "total_participants_inspected": len(set().union(*[set(issues.keys()) for issues in self.issues.values()])),
            "participants_with_raw_issues": len(self.issues["raw_files"]),
            "participants_with_preprocessing_issues": len(self.issues["preprocessed_files"]),
            "participants_with_episode_issues": len(self.issues["episode_files"]),
            "participants_with_datetime_issues": len(self.issues["datetime_issues"]),
            "participants_with_data_quality_issues": len(self.issues["data_quality_issues"]),
            "participants_with_file_access_issues": len(self.issues["file_access_issues"]),
            "participants_with_processing_failures": len(self.issues["processing_failures"]),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Helper function to convert NumPy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(i) for i in obj)
            else:
                return obj
        
        # Convert defaultdicts to regular dicts and handle NumPy types
        serializable_issues = convert_numpy_types({
            category: dict(issues) for category, issues in self.issues.items()
        })
        
        with open(output_path, 'w') as f:
            json.dump(serializable_issues, f, indent=2)
            
        # Also save a text summary for quick review
        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write(f"GPS Data Inspection Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total participants inspected: {self.issues['summary']['total_participants_inspected']}\n")
            f.write(f"Participants with raw data issues: {len(self.issues['raw_files'])}\n")
            f.write(f"Participants with preprocessing issues: {len(self.issues['preprocessed_files'])}\n")
            f.write(f"Participants with episode generation issues: {len(self.issues['episode_files'])}\n")
            f.write(f"Participants with datetime issues: {len(self.issues['datetime_issues'])}\n")
            f.write(f"Participants with data quality issues: {len(self.issues['data_quality_issues'])}\n\n")
            
            f.write("MOST COMMON ISSUES:\n")
            issue_counts = defaultdict(int)
            for category, participants in self.issues.items():
                if category not in ["summary"]:
                    for participant, issues in participants.items():
                        for issue in issues:
                            issue_counts[issue["description"]] += 1
                            
            for description, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- {description}: {count} occurrences\n")
                
            f.write("\nPARTICIPANTS WITH MOST ISSUES:\n")
            participant_issue_counts = defaultdict(int)
            for category, participants in self.issues.items():
                if category not in ["summary"]:
                    for participant, issues in participants.items():
                        participant_issue_counts[participant] += len(issues)
                        
            for participant, count in sorted(participant_issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- Participant {participant}: {count} issues\n")
                
            f.write("\n" + "="*80 + "\n")
                
        logging.info(f"Issue report saved to {output_path}")
        
class DataInspector:
    def __init__(self):
        self.issue_tracker = DataIssueTracker()
        
    def check_datetime_format(self, df: pd.DataFrame, col_name: str, participant_id: str, file_type: str) -> Tuple[bool, List[str]]:
        """Check if datetime format is consistent and valid"""
        issues = []
        is_valid = True
        
        if col_name not in df.columns:
            return False, [f"Column {col_name} not found in dataframe"]
            
        # Skip empty dataframes
        if df.empty:
            return True, []
            
        # Check if column is already datetime type
        if pd.api.types.is_datetime64_dtype(df[col_name]):
            # Check for NaT values
            nat_count = df[col_name].isna().sum()
            if nat_count > 0:
                issues.append(f"Found {nat_count} NaT values in {col_name}")
                is_valid = is_valid and (nat_count / len(df) < 0.1)  # Allow up to 10% NaT
                
            # Check for timezone information
            sample = df[col_name].iloc[0]
            if hasattr(sample, 'tz') and sample.tz is None:
                issues.append(f"Datetime column {col_name} lacks timezone information")
                
            return is_valid, issues
            
        # Column is not datetime type, try to parse it
        try:
            # Try to infer format
            pd.to_datetime(df[col_name], errors='raise')
            return True, []
        except Exception as e:
            # Check for specific parsing issues
            try:
                # Sample problematic values for diagnosis
                sample_values = df[col_name].iloc[:10].tolist()
                issues.append(f"Datetime parsing failed: {str(e)}")
                issues.append(f"Sample values: {sample_values}")
                
                # Check for leading zero issues in time component (common in smartphone data)
                if any(":" in str(val) for val in sample_values):
                    time_pattern_issues = [
                        val for val in sample_values 
                        if isinstance(val, str) and ":" in val and re.search(r':\d\b', val)
                    ]
                    if time_pattern_issues:
                        issues.append(f"Possible missing leading zeros in time: {time_pattern_issues}")
                        
                # Try custom parsing for specific formats we've observed
                custom_formats = [
                    # Try different date/time separators and formats
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y/%m/%d %H:%M",
                    "%d-%m-%Y %H:%M:%S",
                    "%d/%m/%Y %H:%M:%S"
                ]
                
                for fmt in custom_formats:
                    try:
                        pd.to_datetime(df[col_name], format=fmt)
                        issues.append(f"Datetime column could be parsed with format: {fmt}")
                        return False, issues
                    except:
                        pass
                        
                return False, issues
            except:
                return False, [f"Failed to analyze datetime issues in {col_name}: {str(e)}"]
                
    def check_coordinate_validity(self, df: pd.DataFrame, lat_col: str, lon_col: str) -> Tuple[bool, Dict]:
        """Check if coordinates are valid and within reasonable bounds"""
        results = {
            "valid": True,
            "missing_lat": 0,
            "missing_lon": 0,
            "zero_coords": 0,
            "out_of_bounds": 0,
            "valid_coords": 0
        }
        
        # Check if columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            columns_present = [col for col in [lat_col, lon_col] if col in df.columns]
            results["valid"] = False
            results["error"] = f"Missing coordinate columns. Present: {columns_present}"
            return False, results
            
        # Check for missing values
        results["missing_lat"] = df[lat_col].isna().sum()
        results["missing_lon"] = df[lon_col].isna().sum()
        
        # Check for zero coordinates (often default values)
        results["zero_coords"] = ((df[lat_col] == 0) & (df[lon_col] == 0)).sum()
        
        # Check for coordinates outside reasonable bounds
        # For Israel, approximately: Lat 29-34, Lon 34-36
        results["out_of_bounds"] = (
            (df[lat_col] < 29) | (df[lat_col] > 34) | 
            (df[lon_col] < 34) | (df[lon_col] > 36)
        ).sum()
        
        # Count valid coordinates
        results["valid_coords"] = len(df) - max(
            results["missing_lat"], 
            results["missing_lon"], 
            results["zero_coords"], 
            results["out_of_bounds"]
        )
        
        # Determine overall validity (at least 80% valid coordinates)
        if len(df) > 0:
            results["valid"] = results["valid_coords"] / len(df) >= 0.8
            
        return results["valid"], results
    
    def analyze_qstarz_raw_file(self, file_path: Path, participant_id: str):
        """Analyze a raw Qstarz GPS file for issues"""
        logging.info(f"Analyzing raw Qstarz file for participant {participant_id}: {file_path}")
        
        try:
            # Attempt to read the file
            df = pd.read_csv(file_path)
            
            # Check basic statistics
            stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist()
            }
            
            # Look for datetime columns
            datetime_cols = [col for col in df.columns if 'DATE' in col or 'TIME' in col]
            
            datetime_valid = True
            datetime_issues = []
            
            for col in datetime_cols:
                col_valid, col_issues = self.check_datetime_format(df, col, participant_id, "qstarz_raw")
                datetime_valid = datetime_valid and col_valid
                datetime_issues.extend(col_issues)
                
            # Check latitude/longitude columns
            lat_col = next((col for col in df.columns if 'LATITUDE' in col), None)
            lon_col = next((col for col in df.columns if 'LONGITUDE' in col), None)
            
            if lat_col and lon_col:
                coord_valid, coord_results = self.check_coordinate_validity(df, lat_col, lon_col)
                
                if not coord_valid:
                    self.issue_tracker.add_issue(
                        "data_quality_issues", 
                        participant_id,
                        "Raw Qstarz file has coordinate validity issues",
                        {"file": str(file_path), "coordinate_results": coord_results}
                    )
            else:
                self.issue_tracker.add_issue(
                    "data_quality_issues", 
                    participant_id,
                    "Raw Qstarz file missing coordinate columns",
                    {"file": str(file_path), "available_columns": df.columns.tolist()}
                )
                
            # Check for time gaps and sampling rate
            if datetime_valid and len(datetime_cols) > 0:
                # Use the first identified datetime column
                df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]])
                df = df.sort_values(datetime_cols[0])
                
                # Calculate time differences
                df['time_diff'] = df[datetime_cols[0]].diff().dt.total_seconds()
                
                # Analyze sampling rate
                median_gap = df['time_diff'].median()
                max_gap = df['time_diff'].max()
                gaps_over_1min = (df['time_diff'] > 60).sum()
                gaps_over_10min = (df['time_diff'] > 600).sum()
                
                time_stats = {
                    "median_gap_seconds": median_gap,
                    "max_gap_seconds": max_gap,
                    "gaps_over_1min": gaps_over_1min,
                    "gaps_over_10min": gaps_over_10min,
                    "expected_points_per_day": int(86400 / max(1, median_gap)),
                    "total_duration_hours": (df[datetime_cols[0]].max() - df[datetime_cols[0]].min()).total_seconds() / 3600
                }
                
                # Check for unusually large gaps
                if gaps_over_10min > 5:
                    self.issue_tracker.add_issue(
                        "data_quality_issues", 
                        participant_id,
                        f"Raw Qstarz file has {gaps_over_10min} gaps larger than 10 minutes",
                        {"file": str(file_path), "time_stats": time_stats}
                    )
            
            # Record datetime issues if any
            if datetime_issues:
                self.issue_tracker.add_issue(
                    "datetime_issues", 
                    participant_id,
                    "Raw Qstarz file has datetime format issues",
                    {"file": str(file_path), "issues": datetime_issues}
                )
                
            return stats
            
        except Exception as e:
            self.issue_tracker.add_issue(
                "raw_files", 
                participant_id,
                f"Failed to analyze raw Qstarz file: {str(e)}",
                {"file": str(file_path), "error": traceback.format_exc()}
            )
            return None
            
    def analyze_smartphone_raw_file(self, file_path: Path, participant_id: str):
        """Analyze a raw smartphone GPS file for issues"""
        logging.info(f"Analyzing raw smartphone GPS file for participant {participant_id}: {file_path}")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'utf-16']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                self.issue_tracker.add_issue(
                    "raw_files", 
                    participant_id,
                    "Failed to read smartphone GPS file with any encoding",
                    {"file": str(file_path)}
                )
                return None
                
            # Check basic statistics
            stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist()
            }
            
            # Look for datetime components
            date_col = next((col for col in df.columns if col.lower() == 'date'), None)
            time_col = next((col for col in df.columns if col.lower() == 'time'), None)
            
            # Check if both date and time columns exist
            if date_col and time_col:
                # Check date format
                date_valid, date_issues = self.check_datetime_format(df, date_col, participant_id, "smartphone_raw")
                
                # Check time format - focus on missing leading zeros issue
                time_values = df[time_col].astype(str).tolist()
                
                # Detect missing leading zeros in time format (e.g., "20:24:3" instead of "20:24:03")
                missing_zeros = [t for t in time_values if re.search(r':\d\b', t)]
                
                if missing_zeros:
                    sample = missing_zeros[:5]
                    self.issue_tracker.add_issue(
                        "datetime_issues", 
                        participant_id,
                        "Smartphone GPS time values missing leading zeros",
                        {"file": str(file_path), "sample_values": sample}
                    )
                    
                # Try to combine date and time with a custom parser
                combined_valid = False
                try:
                    # First fix the time format to ensure leading zeros
                    df['fixed_time'] = df[time_col].astype(str).apply(
                        lambda t: re.sub(r':(\d)\b', r':0\1', t)
                    )
                    
                    # Combine date and time
                    df['timestamp'] = pd.to_datetime(
                        df[date_col].astype(str) + ' ' + df['fixed_time'], 
                        errors='coerce'
                    )
                    
                    # Check if parsing was successful
                    combined_valid = df['timestamp'].notna().mean() > 0.9  # >90% valid
                    
                    if not combined_valid:
                        self.issue_tracker.add_issue(
                            "datetime_issues", 
                            participant_id,
                            "Failed to combine date and time into valid timestamps",
                            {
                                "file": str(file_path), 
                                "success_rate": f"{df['timestamp'].notna().mean()*100:.1f}%",
                                "sample_failures": df[df['timestamp'].isna()][[date_col, time_col]].head(5).to_dict('records')
                            }
                        )
                except Exception as e:
                    self.issue_tracker.add_issue(
                        "datetime_issues", 
                        participant_id,
                        f"Error combining date and time: {str(e)}",
                        {"file": str(file_path)}
                    )
            else:
                self.issue_tracker.add_issue(
                    "raw_files", 
                    participant_id,
                    "Smartphone GPS file missing date or time columns",
                    {"file": str(file_path), "available_columns": df.columns.tolist()}
                )
                
            # Check coordinates
            lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude']), None)
            lon_col = next((col for col in df.columns if col.lower() in ['lon', 'long', 'longitude']), None)
            
            if lat_col and lon_col:
                coord_valid, coord_results = self.check_coordinate_validity(df, lat_col, lon_col)
                
                if not coord_valid:
                    self.issue_tracker.add_issue(
                        "data_quality_issues", 
                        participant_id,
                        "Smartphone GPS file has coordinate validity issues",
                        {"file": str(file_path), "coordinate_results": coord_results}
                    )
            else:
                self.issue_tracker.add_issue(
                    "data_quality_issues", 
                    participant_id,
                    "Smartphone GPS file missing coordinate columns",
                    {"file": str(file_path), "available_columns": df.columns.tolist()}
                )
                
            return stats
            
        except Exception as e:
            self.issue_tracker.add_issue(
                "raw_files", 
                participant_id,
                f"Failed to analyze smartphone GPS file: {str(e)}",
                {"file": str(file_path), "error": traceback.format_exc()}
            )
            return None
    
    def analyze_preprocessed_gps_file(self, file_path: Path, participant_id: str):
        """Analyze a preprocessed GPS file for issues"""
        logging.info(f"Analyzing preprocessed GPS file for participant {participant_id}: {file_path}")
        
        try:
            # Read the preprocessed file
            df = pd.read_csv(file_path)
            
            # Check basic statistics
            stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "data_source": df['data_source'].iloc[0] if 'data_source' in df.columns and not df.empty else "unknown"
            }
            
            # Check datetime column
            if 'tracked_at' in df.columns:
                datetime_valid, datetime_issues = self.check_datetime_format(df, 'tracked_at', participant_id, "preprocessed")
                
                if datetime_issues:
                    self.issue_tracker.add_issue(
                        "datetime_issues", 
                        participant_id,
                        "Preprocessed GPS file has datetime format issues",
                        {"file": str(file_path), "issues": datetime_issues}
                    )
                    
                # Check for timezone awareness
                if datetime_valid:
                    # Parse the datetime string to check for timezone info
                    try:
                        sample_dt = df['tracked_at'].iloc[0]
                        if isinstance(sample_dt, str) and '+00:00' not in sample_dt and 'Z' not in sample_dt:
                            self.issue_tracker.add_issue(
                                "datetime_issues", 
                                participant_id,
                                "Preprocessed GPS timestamps may lack timezone information",
                                {"file": str(file_path), "sample": sample_dt}
                            )
                    except:
                        pass
                        
                # Check for time gaps
                try:
                    df['tracked_at'] = pd.to_datetime(df['tracked_at'])
                    df = df.sort_values('tracked_at')
                    df['time_diff'] = df['tracked_at'].diff().dt.total_seconds()
                    
                    # Calculate gap statistics
                    median_gap = df['time_diff'].median()
                    max_gap = df['time_diff'].max()
                    gaps_over_1min = (df['time_diff'] > 60).sum()
                    gaps_over_10min = (df['time_diff'] > 600).sum()
                    
                    time_stats = {
                        "median_gap_seconds": median_gap,
                        "max_gap_seconds": max_gap,
                        "gaps_over_1min": gaps_over_1min,
                        "gaps_over_10min": gaps_over_10min,
                        "total_duration_hours": (df['tracked_at'].max() - df['tracked_at'].min()).total_seconds() / 3600
                    }
                    
                    # Check for unusually large gaps
                    if gaps_over_10min > 5:
                        self.issue_tracker.add_issue(
                            "data_quality_issues", 
                            participant_id,
                            f"Preprocessed GPS file has {gaps_over_10min} gaps larger than 10 minutes",
                            {"file": str(file_path), "time_stats": time_stats}
                        )
                        
                    # Check if total duration makes sense (at least a full day)
                    if time_stats["total_duration_hours"] < 24:
                        self.issue_tracker.add_issue(
                            "data_quality_issues", 
                            participant_id,
                            f"Preprocessed GPS file has short duration: {time_stats['total_duration_hours']:.1f} hours",
                            {"file": str(file_path), "time_stats": time_stats}
                        )
                except Exception as e:
                    self.issue_tracker.add_issue(
                        "preprocessed_files", 
                        participant_id,
                        f"Error analyzing time gaps in preprocessed file: {str(e)}",
                        {"file": str(file_path)}
                    )
            else:
                self.issue_tracker.add_issue(
                    "preprocessed_files", 
                    participant_id,
                    "Preprocessed GPS file missing tracked_at column",
                    {"file": str(file_path), "available_columns": df.columns.tolist()}
                )
                
            # Check coordinates
            coord_valid, coord_results = self.check_coordinate_validity(df, 'latitude', 'longitude')
            
            if not coord_valid:
                self.issue_tracker.add_issue(
                    "data_quality_issues", 
                    participant_id,
                    "Preprocessed GPS file has coordinate validity issues",
                    {"file": str(file_path), "coordinate_results": coord_results}
                )
                
            return stats
            
        except Exception as e:
            self.issue_tracker.add_issue(
                "preprocessed_files", 
                participant_id,
                f"Failed to analyze preprocessed GPS file: {str(e)}",
                {"file": str(file_path), "error": traceback.format_exc()}
            )
            return None
            
    def analyze_episode_data(self, episode_dir: Path, participant_id: str):
        """Analyze episode detection output for issues"""
        logging.info(f"Analyzing episode data for participant {participant_id}")
        
        try:
            # Check if directory exists
            if not episode_dir.exists():
                self.issue_tracker.add_issue(
                    "episode_files", 
                    participant_id,
                    "Episode output directory does not exist",
                    {"directory": str(episode_dir)}
                )
                return None
                
            # Look for summary file
            summary_file = episode_dir / 'episode_summary.csv'
            if not summary_file.exists():
                self.issue_tracker.add_issue(
                    "episode_files", 
                    participant_id,
                    "Episode summary file not found",
                    {"expected_path": str(summary_file)}
                )
                return None
                
            # Analyze summary file
            summary_df = pd.read_csv(summary_file)
            
            stats = {
                "total_days": len(summary_df),
                "valid_days": summary_df['valid_day'].sum() if 'valid_day' in summary_df.columns else 0,
                "invalid_days": len(summary_df) - summary_df['valid_day'].sum() if 'valid_day' in summary_df.columns else 0,
                "days_by_method": summary_df.get('processing_method', pd.Series()).value_counts().to_dict()
            }
            
            # Check for days with no valid episodes
            zero_episodes_days = summary_df[
                (summary_df['mobility_episodes'] == 0) & 
                (summary_df['valid_day'] == True)
            ] if 'mobility_episodes' in summary_df.columns and 'valid_day' in summary_df.columns else pd.DataFrame()
            
            if not zero_episodes_days.empty:
                self.issue_tracker.add_issue(
                    "episode_files", 
                    participant_id,
                    f"Found {len(zero_episodes_days)} days marked valid but with zero mobility episodes",
                    {"dates": zero_episodes_days['date'].tolist() if 'date' in zero_episodes_days.columns else []}
                )
                
            # Analyze quality assessment file if it exists
            quality_file = episode_dir / 'day_quality_assessment.csv'
            if quality_file.exists():
                quality_df = pd.read_csv(quality_file)
                
                # Look for common failure reasons
                if 'valid' in quality_df.columns and 'stage' in quality_df.columns:
                    failure_stages = quality_df[quality_df['valid'] == False]['stage'].value_counts().to_dict()
                    
                    # Extract failure reasons if available
                    failure_reasons = {}
                    reason_columns = [col for col in quality_df.columns if 'reason' in col.lower()]
                    if reason_columns:
                        for reason_col in reason_columns:
                            reasons = quality_df[quality_df['valid'] == False][reason_col].dropna().value_counts().to_dict()
                            failure_reasons[reason_col] = reasons
                            
                    stats["failure_stages"] = failure_stages
                    stats["failure_reasons"] = failure_reasons
                    
                    # Look for specific patterns that indicate problems
                    if 'staypoints' in failure_stages and failure_stages.get('staypoints', 0) > 3:
                        self.issue_tracker.add_issue(
                            "processing_failures", 
                            participant_id,
                            f"Multiple days failing at staypoint generation: {failure_stages.get('staypoints')}",
                            {"failure_details": failure_reasons}
                        )
                        
            # For invalid days, check the specific daily files
            if 'valid_day' in summary_df.columns and 'date' in summary_df.columns:
                invalid_dates = summary_df[summary_df['valid_day'] == False]['date'].tolist()
                
                for date in invalid_dates[:5]:  # Limit to first 5 to avoid too much processing
                    # Check for mobility files
                    mobility_file = next(episode_dir.glob(f"{date}_mobility_episodes.csv"), None)
                    
                    if mobility_file and mobility_file.exists():
                        try:
                            mobility_df = pd.read_csv(mobility_file)
                            if not mobility_df.empty:
                                self.issue_tracker.add_issue(
                                    "episode_files", 
                                    participant_id,
                                    f"Day {date} marked invalid but has {len(mobility_df)} mobility episodes",
                                    {"file": str(mobility_file)}
                                )
                        except Exception as e:
                            self.issue_tracker.add_issue(
                                "episode_files", 
                                participant_id,
                                f"Error analyzing mobility file for date {date}: {str(e)}",
                                {"file": str(mobility_file) if mobility_file else "None"}
                            )
                            
            return stats
            
        except Exception as e:
            self.issue_tracker.add_issue(
                "episode_files", 
                participant_id,
                f"Failed to analyze episode data: {str(e)}",
                {"directory": str(episode_dir), "error": traceback.format_exc()}
            )
            return None
            
    def process_participant(self, participant_id: str):
        """Process all data for a single participant"""
        logging.info(f"Processing participant {participant_id}")
        
        participant_report = {
            "raw_qstarz": None,
            "raw_smartphone": None,
            "preprocessed": None,
            "episodes": None
        }
        
        # Check raw Qstarz data
        qstarz_file = next(QSTARZ_DIR.glob(f"{participant_id}*_Qstarz_processed.csv"), None)
        if qstarz_file:
            participant_report["raw_qstarz"] = self.analyze_qstarz_raw_file(qstarz_file, participant_id)
        else:
            logging.info(f"No raw Qstarz file found for participant {participant_id}")
            
        # Check raw smartphone data
        smartphone_folder = next(
            (f for f in RAW_DATA_DIR.glob(f"Participants/Pilot_*") 
             if f.name.endswith(f"_{participant_id.lstrip('0')}") or f.name.endswith(f"_{participant_id}")),
            None
        )
        if smartphone_folder:
            app_folder = smartphone_folder / '9 - Smartphone Tracking App'
            if app_folder.exists():
                smartphone_file = next(app_folder.glob(f"{participant_id.lstrip('0')}-gps.csv"), None)
                if smartphone_file:
                    participant_report["raw_smartphone"] = self.analyze_smartphone_raw_file(smartphone_file, participant_id)
                else:
                    logging.info(f"No smartphone GPS file found for participant {participant_id}")
            else:
                logging.info(f"No smartphone app folder found for participant {participant_id}")
        else:
            logging.info(f"No smartphone folder found for participant {participant_id}")
            
        # Check preprocessed data
        prep_file = GPS_PREP_DIR / f"{participant_id}_gps_prep.csv"
        if prep_file.exists():
            participant_report["preprocessed"] = self.analyze_preprocessed_gps_file(prep_file, participant_id)
        else:
            logging.info(f"No preprocessed GPS file found for participant {participant_id}")
            self.issue_tracker.add_issue(
                "file_access_issues", 
                participant_id,
                "Preprocessed GPS file not found",
                {"expected_path": str(prep_file)}
            )
            
        # Check episode data
        episode_dir = EPISODE_OUTPUT_DIR / participant_id
        if episode_dir.exists():
            participant_report["episodes"] = self.analyze_episode_data(episode_dir, participant_id)
        else:
            logging.info(f"No episode directory found for participant {participant_id}")
            self.issue_tracker.add_issue(
                "file_access_issues", 
                participant_id,
                "Episode directory not found",
                {"expected_path": str(episode_dir)}
            )
            
        # Check for processing consistency
        if participant_report["preprocessed"] and participant_report["episodes"] is None:
            self.issue_tracker.add_issue(
                "processing_failures", 
                participant_id,
                "Data was preprocessed but episode detection failed completely",
                {}
            )
            
        if (participant_report["raw_qstarz"] or participant_report["raw_smartphone"]) and participant_report["preprocessed"] is None:
            self.issue_tracker.add_issue(
                "processing_failures", 
                participant_id,
                "Raw data exists but preprocessing failed",
                {}
            )
            
        return participant_report
        
    def inspect_timezone_handling(self, participant_id: str):
        """Specifically analyze timezone handling across processing stages"""
        logging.info(f"Inspecting timezone handling for participant {participant_id}")
        
        timezone_report = {
            "raw_timezone": None,
            "preprocessed_timezone": None,
            "episodes_timezone": None,
            "inconsistencies": []
        }
        
        # Check raw Qstarz data
        qstarz_file = next(QSTARZ_DIR.glob(f"{participant_id}*_Qstarz_processed.csv"), None)
        if qstarz_file:
            try:
                df = pd.read_csv(qstarz_file)
                datetime_col = next((col for col in df.columns if 'UTC DATE TIME' in col), None)
                
                if datetime_col:
                    try:
                        sample_dt = pd.to_datetime(df[datetime_col].iloc[0])
                        timezone_report["raw_timezone"] = str(sample_dt.tz) if hasattr(sample_dt, 'tz') else "naive"
                    except:
                        timezone_report["raw_timezone"] = "parsing_failed"
            except:
                timezone_report["raw_timezone"] = "file_error"
                
        # Check preprocessed data
        prep_file = GPS_PREP_DIR / f"{participant_id}_gps_prep.csv"
        if prep_file.exists():
            try:
                df = pd.read_csv(prep_file)
                if 'tracked_at' in df.columns and not df.empty:
                    try:
                        # Try to determine if the string representation has timezone info
                        sample_dt_str = df['tracked_at'].iloc[0]
                        if isinstance(sample_dt_str, str):
                            if '+00:00' in sample_dt_str or 'Z' in sample_dt_str:
                                timezone_report["preprocessed_timezone"] = "UTC"
                            else:
                                timezone_report["preprocessed_timezone"] = "naive_string"
                                timezone_report["inconsistencies"].append(
                                    "Preprocessed timestamps appear to be timezone-naive strings"
                                )
                        
                        # Try parsing it
                        sample_dt = pd.to_datetime(sample_dt_str)
                        parsed_tz = str(sample_dt.tz) if hasattr(sample_dt, 'tz') else "naive"
                        
                        if timezone_report["preprocessed_timezone"] != parsed_tz and parsed_tz == "naive":
                            timezone_report["inconsistencies"].append(
                                f"Preprocessed timestamps have timezone in string ({timezone_report['preprocessed_timezone']}) but parse as naive"
                            )
                            
                        timezone_report["preprocessed_timezone"] = parsed_tz
                    except:
                        timezone_report["preprocessed_timezone"] = "parsing_failed"
            except:
                timezone_report["preprocessed_timezone"] = "file_error"
                
        # Check episode data - look at mobility episodes
        episode_dir = EPISODE_OUTPUT_DIR / participant_id
        if episode_dir.exists():
            # Find any mobility episodes file
            mobility_file = next(episode_dir.glob(f"*_mobility_episodes.csv"), None)
            
            if mobility_file:
                try:
                    df = pd.read_csv(mobility_file)
                    if 'started_at' in df.columns and not df.empty:
                        try:
                            sample_dt_str = df['started_at'].iloc[0]
                            if isinstance(sample_dt_str, str):
                                if '+00:00' in sample_dt_str or 'Z' in sample_dt_str:
                                    timezone_report["episodes_timezone"] = "UTC"
                                else:
                                    timezone_report["episodes_timezone"] = "naive_string"
                            
                            # Try parsing it
                            sample_dt = pd.to_datetime(sample_dt_str)
                            timezone_report["episodes_timezone"] = str(sample_dt.tz) if hasattr(sample_dt, 'tz') else "naive"
                        except:
                            timezone_report["episodes_timezone"] = "parsing_failed"
                except:
                    timezone_report["episodes_timezone"] = "file_error"
                    
        # Check for inconsistencies across processing stages
        if (timezone_report["raw_timezone"] and timezone_report["preprocessed_timezone"] and
            timezone_report["raw_timezone"] != timezone_report["preprocessed_timezone"]):
            timezone_report["inconsistencies"].append(
                f"Timezone mismatch between raw ({timezone_report['raw_timezone']}) and preprocessed ({timezone_report['preprocessed_timezone']})"
            )
            
        if (timezone_report["preprocessed_timezone"] and timezone_report["episodes_timezone"] and
            timezone_report["preprocessed_timezone"] != timezone_report["episodes_timezone"]):
            timezone_report["inconsistencies"].append(
                f"Timezone mismatch between preprocessed ({timezone_report['preprocessed_timezone']}) and episodes ({timezone_report['episodes_timezone']})"
            )
            
        if timezone_report["inconsistencies"]:
            self.issue_tracker.add_issue(
                "datetime_issues", 
                participant_id,
                "Timezone handling inconsistencies detected",
                {"timezone_report": timezone_report}
            )
            
        return timezone_report
        
    def run_inspection(self, participant_ids=None):
        """Run the full inspection process"""
        # Get list of participant IDs if not provided
        if participant_ids is None:
            # Find all participants with any data
            qstarz_ids = {f.stem.split('_')[0] for f in QSTARZ_DIR.glob('*_Qstarz_processed.csv')}
            preprocessed_ids = {f.stem.split('_')[0] for f in GPS_PREP_DIR.glob('*_gps_prep.csv')}
            episode_ids = {f.name for f in EPISODE_OUTPUT_DIR.glob('*') if f.is_dir()}
            
            participant_ids = sorted(qstarz_ids | preprocessed_ids | episode_ids)
            
        logging.info(f"Running inspection for {len(participant_ids)} participants")
        
        results = {}
        
        for participant_id in participant_ids:
            try:
                # Process regular data inspection
                results[participant_id] = self.process_participant(participant_id)
                
                # Also check timezone handling
                results[participant_id]["timezone_analysis"] = self.inspect_timezone_handling(participant_id)
                
            except Exception as e:
                logging.error(f"Error processing participant {participant_id}: {str(e)}")
                traceback.print_exc()
                
        # Save the issue report
        report_path = Path("data_inspection_report.json")
        self.issue_tracker.save_report(report_path)
        
        # Calculate total issues safely
        total_issues = 0
        for category_name, category in self.issue_tracker.issues.items():
            if category_name != "summary" and isinstance(category, dict):
                for participant_id, issues in category.items():
                    if isinstance(issues, list):
                        total_issues += len(issues)
        
        # Return summary of findings
        summary = {
            "participants_processed": len(results),
            "participants_with_issues": len(set().union(*[set(issues.keys()) for issues in self.issue_tracker.issues.values() if isinstance(issues, dict)])),
            "total_issues": total_issues,
            "report_path": str(report_path)
        }
        
        return summary

def main():
    """Main function to run the inspector"""
    inspector = DataInspector()
    summary = inspector.run_inspection()
    
    print("\n" + "="*80)
    print("GPS DATA INSPECTION SUMMARY")
    print("="*80)
    print(f"Participants processed: {summary['participants_processed']}")
    print(f"Participants with issues: {summary['participants_with_issues']}")
    print(f"Total issues found: {summary['total_issues']}")
    print(f"Detailed report saved to: {summary['report_path']}")
    print("="*80)
    
    # Also print the most critical issues
    issue_tracker = inspector.issue_tracker
    
    # Count issues by category
    category_counts = {}
    for category, participants in issue_tracker.issues.items():
        if category != "summary" and isinstance(participants, dict):
            total = 0
            for participant_id, issues in participants.items():
                if isinstance(issues, list):
                    total += len(issues)
            category_counts[category] = total
    
    print("\nISSUES BY CATEGORY:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if category != "summary":
            print(f"- {category}: {count} issues")
    
    return 0

if __name__ == "__main__":
    main()