import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCombiner:
    """
    Combines fragmentation metrics with EMA responses and generates analyses.
    """
    def __init__(self, 
                 fragmentation_dir: Path, 
                 ema_dir: Path,
                 output_dir: Path,
                 end_of_day_only: bool = True,
                 debug_mode: bool = False):
        """
        Initialize the metrics combiner.
        
        Args:
            fragmentation_dir: Directory containing fragmentation metrics
            ema_dir: Directory containing EMA data
            output_dir: Directory for saving outputs
            end_of_day_only: Whether to use only end-of-day EMA responses
            debug_mode: Whether to enable debug logging
        """
        self.fragmentation_dir = fragmentation_dir
        self.ema_dir = ema_dir
        self.output_dir = output_dir
        self.end_of_day_only = end_of_day_only
        self.debug_mode = debug_mode
        
        # Configure logging level
        if debug_mode:
            logger.setLevel(logging.DEBUG)
            
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fragmentation_data(self) -> pd.DataFrame:
        """
        Load fragmentation metrics from the fragmentation directory.
        
        Returns:
            DataFrame containing fragmentation metrics
        """
        logger.info("Loading fragmentation data...")
        
        # Try different possible filenames for fragmentation data
        possible_files = [
            'fragmentation_all_metrics.csv',
            'fragmentation_metrics.csv',
            'daily_fragmentation.csv'
        ]
        
        for filename in possible_files:
            file_path = self.fragmentation_dir / filename
            if file_path.exists():
                logger.info(f"Found fragmentation data at: {file_path}")
                frag_data = pd.read_csv(file_path)
                
                # Check if we have the expected columns
                required_cols = ['participant_id', 'date']
                frag_cols = [col for col in frag_data.columns if 'fragmentation' in col]
                
                if all(col in frag_data.columns for col in required_cols) and frag_cols:
                    logger.info(f"Loaded fragmentation data with {len(frag_data)} rows and {len(frag_data.columns)} columns")
                    
                    # Convert date to datetime if it's not already
                    if 'date' in frag_data.columns and not pd.api.types.is_datetime64_any_dtype(frag_data['date']):
                        frag_data['date'] = pd.to_datetime(frag_data['date'])
                    
                    return frag_data
        
        logger.error("No valid fragmentation data found. Please check the path.")
        return pd.DataFrame()
    
    def load_ema_data(self) -> pd.DataFrame:
        """
        Load normalized EMA data, focusing on end-of-day responses if specified.
        
        Returns:
            DataFrame containing EMA data
        """
        logger.info("Loading EMA data...")
        
        # First try the summary files
        summary_files = [
            'overall_summary_by_scale.csv',
            'overall_summary_by_variable.csv'
        ]
        
        for filename in summary_files:
            file_path = self.ema_dir / filename
            if file_path.exists():
                logger.info(f"Found EMA summary data at: {file_path}")
                ema_data = pd.read_csv(file_path)
                
                # Check if we have the expected columns
                if 'Participant_ID' in ema_data.columns and ('Scale' in ema_data.columns or 'Variable' in ema_data.columns):
                    logger.info(f"Loaded EMA summary data with {len(ema_data)} rows")
                    
                    # Convert date columns to datetime if they exist
                    for date_col in ['First_Response', 'Last_Response']:
                        if date_col in ema_data.columns and not pd.api.types.is_datetime64_any_dtype(ema_data[date_col]):
                            ema_data[date_col] = pd.to_datetime(ema_data[date_col])
                    
                    # If this is variable-level data, we'll process it differently
                    if 'Variable' in ema_data.columns:
                        return self._process_variable_ema_data(ema_data)
                    return ema_data
        
        # If summary files not found, look for individual normalized files
        all_ema_data = []
        normalized_files = list(self.ema_dir.glob("normalized_*.csv"))
        
        if normalized_files:
            logger.info(f"Found {len(normalized_files)} normalized EMA files")
            
            for file_path in normalized_files:
                try:
                    ema_file = pd.read_csv(file_path)
                    
                    # Filter for end-of-day EMA if specified
                    if self.end_of_day_only:
                        # Look for indicators of end-of-day EMA
                        # This will depend on your specific data, adapt as needed
                        eod_indicators = ['end_of_day', 'evening', 'night', 'EOD']
                        
                        # Try different potential column names that might indicate EMA timing
                        timing_cols = [col for col in ema_file.columns if any(
                            timing_term in col.lower() for timing_term in 
                            ['time', 'timing', 'period', 'session', 'type']
                        )]
                        
                        if timing_cols:
                            for col in timing_cols:
                                mask = ema_file[col].astype(str).str.lower().apply(
                                    lambda x: any(indicator.lower() in x for indicator in eod_indicators)
                                )
                                if mask.sum() > 0:
                                    ema_file = ema_file[mask].copy()
                                    logger.debug(f"Filtered to {len(ema_file)} end-of-day EMAs in {file_path.name}")
                                    break
                    
                    # Add to our collection
                    if not ema_file.empty:
                        all_ema_data.append(ema_file)
                
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
            
            if all_ema_data:
                combined_ema = pd.concat(all_ema_data, ignore_index=True)
                logger.info(f"Combined {len(combined_ema)} EMA rows from individual files")
                return combined_ema
        
        logger.error("No valid EMA data found. Please check the path.")
        return pd.DataFrame()
    
    def _process_variable_ema_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process variable-level EMA data to make it suitable for merging.
        
        Args:
            data: DataFrame with variable-level EMA data
            
        Returns:
            Processed DataFrame ready for merging
        """
        # Pivot the data to have variables as columns
        # This makes it easier to merge with fragmentation data
        
        # First, determine which value column to use
        value_cols = [col for col in data.columns if any(
            term in col for term in ['Mean', 'SD', 'Score']
        )]
        
        # Prioritize Z-standardized scores if available
        value_col = next((col for col in value_cols if 'Zstd' in col and 'Mean' in col), None)
        if not value_col:
            value_col = next((col for col in value_cols if 'Mean' in col), None)
        
        if not value_col:
            logger.warning("Could not find appropriate value column for EMA data")
            return data
        
        # Create date column from First_Response if needed
        if 'date' not in data.columns and 'First_Response' in data.columns:
            data['date'] = pd.to_datetime(data['First_Response']).dt.date
        
        # Pivot the data
        pivot_cols = ['Participant_ID', 'date']
        if all(col in data.columns for col in pivot_cols):
            try:
                # Group by participant, date, and variable
                pivot_data = data.pivot_table(
                    index=pivot_cols,
                    columns='Variable' if 'Variable' in data.columns else 'Scale',
                    values=value_col,
                    aggfunc='mean'
                ).reset_index()
                
                logger.info(f"Pivoted EMA data to have {len(pivot_data.columns)} columns")
                return pivot_data
            except Exception as e:
                logger.error(f"Error pivoting EMA data: {str(e)}")
                return data
        
        return data
    
    def _normalize_participant_id(self, participant_id):
        """
        Normalize participant IDs to handle different formats.
        Extracts the numeric portion from prefixed IDs like 'Surreal_123' or 'SURREAL123'.
        
        Args:
            participant_id: Participant ID in any format
            
        Returns:
            Normalized ID as string
        """
        if participant_id is None:
            return ""
        
        # Convert to string if not already
        id_str = str(participant_id).strip()
        
        # Extract numeric portion from prefixed IDs
        if 'surreal' in id_str.lower():
            # Extract all digits
            digits = ''.join(c for c in id_str if c.isdigit())
            if digits:
                # Remove leading zeros
                return str(int(digits))
            else:
                return id_str
        else:
            # For already numeric IDs, ensure consistent format
            try:
                # If it's purely numeric, remove leading zeros
                if id_str.isdigit():
                    return str(int(id_str))
                else:
                    return id_str
            except:
                return id_str

    def merge_data(self, frag_data: pd.DataFrame, ema_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge fragmentation and EMA data based on participant ID and date.
        
        Args:
            frag_data: DataFrame with fragmentation metrics
            ema_data: DataFrame with EMA responses
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging fragmentation and EMA data...")
        
        if frag_data.empty or ema_data.empty:
            logger.error("Cannot merge: One or both datasets are empty")
            return pd.DataFrame()
        
        # Handle different column naming conventions
        frag_id_col = 'participant_id'
        ema_id_col = 'Participant_ID' if 'Participant_ID' in ema_data.columns else 'participant_id'
        
        # Check if we have both ID columns in both datasets
        if frag_id_col not in frag_data.columns:
            logger.error(f"Column '{frag_id_col}' not found in fragmentation data")
            return pd.DataFrame()
        
        if ema_id_col not in ema_data.columns:
            logger.error(f"Column '{ema_id_col}' not found in EMA data")
            return pd.DataFrame()
        
        # Make copies to avoid modifying originals
        frag_data = frag_data.copy()
        ema_data = ema_data.copy()
        
        # Create normalized participant ID columns for merging
        logger.info("Normalizing participant IDs for better matching...")
        frag_data['normalized_id'] = frag_data[frag_id_col].apply(self._normalize_participant_id)
        ema_data['normalized_id'] = ema_data[ema_id_col].apply(self._normalize_participant_id)
        
        # Log original and normalized IDs for both datasets
        logger.info("Sample participant ID normalization for fragmentation data:")
        sample_frag_ids = list(zip(frag_data[frag_id_col].head(5), frag_data['normalized_id'].head(5)))
        for orig, norm in sample_frag_ids:
            logger.info(f"  Original: {orig} -> Normalized: {norm}")
        
        logger.info("Sample participant ID normalization for EMA data:")
        sample_ema_ids = list(zip(ema_data[ema_id_col].head(5), ema_data['normalized_id'].head(5)))
        for orig, norm in sample_ema_ids:
            logger.info(f"  Original: {orig} -> Normalized: {norm}")
        
        # Check for participant ID overlap
        frag_participants = set(frag_data['normalized_id'].unique())
        ema_participants = set(ema_data['normalized_id'].unique())
        common_participants = frag_participants.intersection(ema_participants)
        
        logger.info(f"Fragmentation data has {len(frag_participants)} unique participants")
        logger.info(f"EMA data has {len(ema_participants)} unique participants")
        logger.info(f"There are {len(common_participants)} participants in common after normalization")
        
        if len(common_participants) == 0:
            logger.error("No common participants found between the datasets even after normalization")
            # Print all unique IDs to help debugging
            logger.info(f"All normalized fragmentation IDs: {sorted(list(frag_participants))}")
            logger.info(f"All normalized EMA IDs: {sorted(list(ema_participants))}")
            return pd.DataFrame()
        
        # Check if EMA data is daily-level or summary-level
        # Look for date-related columns that might indicate daily data
        ema_date_cols = [col for col in ema_data.columns if any(
            term in col.lower() for term in ['date', 'day', 'response', 'time']
        )]
        
        # Default to summary-level EMA data
        has_daily_ema = False
        ema_date_col = None
        
        # Check if any date column has valid, non-placeholder dates
        for col in ema_date_cols:
            if pd.api.types.is_datetime64_any_dtype(ema_data[col]) or 'date' in col.lower():
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(ema_data[col]):
                    try:
                        ema_data[col] = pd.to_datetime(ema_data[col], errors='coerce')
                    except:
                        continue
                
                # Check if this column has multiple dates per participant
                date_counts_per_participant = ema_data.groupby('normalized_id')[col].nunique()
                max_dates_per_participant = date_counts_per_participant.max()
                
                # Check if dates are not placeholders (1970-01-01)
                if max_dates_per_participant > 1:
                    # Convert dates to date objects for comparison
                    dates = pd.to_datetime(ema_data[col]).dt.date
                    # Check if all dates are not epoch/placeholder (1970-01-01)
                    placeholder_date = pd.Timestamp('1970-01-01').date()
                    non_placeholder_dates = dates[dates != placeholder_date]
                    
                    if len(non_placeholder_dates) > 0:
                        has_daily_ema = True
                        ema_date_col = col
                        logger.info(f"Found valid daily date column: {col}")
                        break
        
        # Log what we detected
        if has_daily_ema:
            logger.info(f"Detected daily-level EMA data using column: {ema_date_col}")
        else:
            logger.info("No valid daily date columns found - treating as summary-level EMA data")
        
        # Since we have common participants but no common dates, force participant_id-only merge
        logger.info("Forcing participant ID-only merge since we have common participants")
        
        try:
            merged_df = pd.merge(
                frag_data,
                ema_data,
                on='normalized_id',
                how='inner'
            )
            
            # Clean up - remove the temporary normalized ID column
            if 'normalized_id' in merged_df.columns:
                merged_df = merged_df.drop('normalized_id', axis=1)
            
            # If we have duplicate column names (except for the merged columns), rename them
            duplicate_cols = [col for col in merged_df.columns if merged_df.columns.tolist().count(col) > 1]
            if duplicate_cols:
                rename_dict = {}
                for col in duplicate_cols:
                    if col not in ['normalized_id']:  # Skip already merged columns
                        # Find all occurrences and rename them
                        occurrences = [i for i, c in enumerate(merged_df.columns) if c == col]
                        for i, pos in enumerate(occurrences[1:], 1):  # Skip first occurrence
                            rename_dict[merged_df.columns[pos]] = f"{col}_ema_{i}"
            
            # Apply renaming
            if rename_dict:
                merged_df = merged_df.rename(columns=rename_dict)
            
            # Check if we have any data after merging
            if merged_df.empty:
                logger.error("No data after merging - this should not happen if we had common participants")
                return pd.DataFrame()
            
            logger.info(f"Merged data has {len(merged_df)} rows and {len(merged_df.columns)} columns")
            logger.info(f"Retained {len(merged_df)} of {len(frag_data.loc[frag_data['normalized_id'].isin(common_participants)])} fragmentation data points")
            
            return merged_df
        except Exception as e:
            logger.error(f"Error during merge: {str(e)}")
            # If there was an error, try a more basic merge approach
            logger.info("Trying a basic merge approach...")
            try:
                # Simplify the datasets to just ID and a few key columns
                frag_simple = frag_data[['normalized_id', frag_id_col]].drop_duplicates()
                
                # Find fragmentation columns
                frag_metric_cols = [col for col in frag_data.columns if 'fragmentation' in col.lower()]
                
                if frag_metric_cols:
                    # For each fragmentation metric, compute the mean per participant
                    for col in frag_metric_cols:
                        frag_simple[col] = frag_data.groupby('normalized_id')[col].mean().reindex(frag_simple['normalized_id']).values
                
                # Merge simplified dataframes
                merged_simple = pd.merge(
                    frag_simple,
                    ema_data,
                    on='normalized_id',
                    how='inner'
                )
                
                # Remove normalized_id
                if 'normalized_id' in merged_simple.columns:
                    merged_simple = merged_simple.drop('normalized_id', axis=1)
                
                logger.info(f"Basic merge approach returned {len(merged_simple)} rows")
                return merged_simple
                
            except Exception as e2:
                logger.error(f"Basic merge also failed: {str(e2)}")
                return pd.DataFrame()
    
    def analyze_combined_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze the combined dataset with fragmentation and EMA metrics.
        
        Args:
            data: DataFrame with combined fragmentation and EMA data
            
        Returns:
            Dictionary with analysis results
        """
        if data.empty:
            logger.error("No data to analyze")
            return {}
        
        logger.info("Analyzing combined data...")
        
        results = {}
        
        # Identify fragmentation and EMA metrics columns
        frag_cols = [col for col in data.columns if 'fragmentation' in col.lower()]
        
        # Potential EMA metric columns (adjust based on actual data)
        ema_indicators = ['anxiety', 'stress', 'mood', 'STAI', 'CES', 'CALM', 'PEACE', 
                           'SATISFACTION', 'HAPPY', 'ENJOYMENT', 'NERVOUS', 'WORRY']
        
        ema_cols = [col for col in data.columns if any(
            indicator in col for indicator in ema_indicators
        )]
        
        # Basic statistics for each group of metrics
        for col_group, col_list in [('fragmentation', frag_cols), ('ema', ema_cols)]:
            if col_list:
                group_stats = data[col_list].describe()
                results[f'{col_group}_stats'] = group_stats.to_dict()
                logger.info(f"Calculated statistics for {len(col_list)} {col_group} metrics")
        
        # Calculate correlations between fragmentation and EMA metrics
        if frag_cols and ema_cols:
            correlations = data[frag_cols + ema_cols].corr()
            # Extract just the cross-correlations (frag vs ema)
            cross_corr = correlations.loc[frag_cols, ema_cols]
            results['cross_correlations'] = cross_corr.to_dict()
            
            # Identify strongest correlations
            flat_corr = []
            for frag_col in frag_cols:
                for ema_col in ema_cols:
                    corr_val = correlations.loc[frag_col, ema_col]
                    if not pd.isna(corr_val):
                        flat_corr.append((frag_col, ema_col, corr_val))
            
            # Sort by absolute correlation value
            flat_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            results['strongest_correlations'] = flat_corr[:5]  # Top 5
            
            logger.info(f"Calculated {len(flat_corr)} cross-correlations between metrics")
            
            # Log the strongest correlations
            if flat_corr:
                logger.info("Strongest correlations between fragmentation and EMA metrics:")
                for frag, ema, corr in flat_corr[:5]:
                    logger.info(f"  {frag} vs {ema}: {corr:.3f}")
        
        # Participant-level summaries
        if 'participant_id' in data.columns:
            participant_counts = data['participant_id'].value_counts()
            results['participant_counts'] = participant_counts.to_dict()
            
            # Group by participant and calculate means
            participant_means = data.groupby('participant_id')[frag_cols + ema_cols].mean()
            results['participant_means'] = participant_means.to_dict()
            
            logger.info(f"Generated summaries for {len(participant_counts)} participants")
        
        return results
    
    def generate_visualizations(self, data: pd.DataFrame):
        """
        Generate visualization plots for combined metrics.
        
        Args:
            data: DataFrame with combined fragmentation and EMA data
        """
        if data.empty:
            logger.error("No data for visualizations")
            return
        
        logger.info("Generating visualizations...")
        
        # Create plots directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set(font_scale=1.6)  # Increase font size for better readability
        
        # Identify fragmentation and EMA metrics columns
        frag_cols = [col for col in data.columns if 'fragmentation' in col.lower()]
        
        # Potential EMA metric columns (adjust based on actual data)
        ema_indicators = ['anxiety', 'stress', 'mood', 'STAI', 'CES', 'CALM', 'PEACE', 
                           'SATISFACTION', 'HAPPY', 'ENJOYMENT', 'NERVOUS', 'WORRY']
        
        ema_cols = [col for col in data.columns if any(
            indicator in col for indicator in ema_indicators
        )]
        
        if not frag_cols or not ema_cols:
            logger.warning("Missing either fragmentation or EMA metrics for visualizations")
            return
        
        # 1. Correlation heatmap
        plt.figure(figsize=(16, 12))
        mask = np.zeros((len(frag_cols), len(ema_cols)))
        corr_matrix = data[frag_cols + ema_cols].corr().loc[frag_cols, ema_cols]
        
        # Create heatmap
        ax = sns.heatmap(
            corr_matrix, 
            cmap='coolwarm', 
            annot=True, 
            fmt='.2f', 
            linewidths=0.5,
            annot_kws={"size": 16}
        )
        plt.title('Correlation between Fragmentation and EMA Metrics', fontsize=24)
        plt.xticks(rotation=45, ha='right', fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300)
        plt.close()
        
        # 2. Scatter plots for top correlations
        # Calculate correlations
        top_correlations = []
        for frag_col in frag_cols:
            for ema_col in ema_cols:
                corr = data[frag_col].corr(data[ema_col])
                if pd.notna(corr):
                    top_correlations.append((frag_col, ema_col, corr))
        
        # Sort by absolute correlation
        top_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Create scatter plots for top 3 correlations
        for i, (frag_col, ema_col, corr) in enumerate(top_correlations[:3]):
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot with regression line
            sns.regplot(
                x=frag_col, 
                y=ema_col, 
                data=data,
                scatter_kws={"s": 80, "alpha": 0.7},
                line_kws={"color": "red", "lw": 2}
            )
            
            plt.title(f'{frag_col} vs {ema_col}\nCorrelation: {corr:.3f}', fontsize=24)
            plt.xlabel(frag_col, fontsize=18)
            plt.ylabel(ema_col, fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            plt.savefig(plots_dir / f'scatter_{i+1}_{frag_col}_{ema_col}.png', dpi=300)
            plt.close()
        
        # 3. Box plots of fragmentation by participant
        if 'participant_id' in data.columns and len(data['participant_id'].unique()) <= 15:
            for frag_col in frag_cols:
                plt.figure(figsize=(14, 8))
                
                # Create box plot
                sns.boxplot(
                    x='participant_id', 
                    y=frag_col, 
                    data=data,
                    palette='Set3'
                )
                
                plt.title(f'{frag_col} by Participant', fontsize=24)
                plt.xlabel('Participant ID', fontsize=18)
                plt.ylabel(frag_col, fontsize=18)
                plt.xticks(rotation=45, fontsize=16)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(plots_dir / f'boxplot_{frag_col}_by_participant.png', dpi=300)
                plt.close()
        
        # 4. Time series of fragmentation and EMA metrics
        if 'date' in data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            
            # Select top fragmentation and EMA metrics
            top_frag = frag_cols[0] if frag_cols else None
            top_ema = ema_cols[0] if ema_cols else None
            
            if top_frag and top_ema:
                plt.figure(figsize=(15, 10))
                
                # Create two y-axes
                fig, ax1 = plt.subplots(figsize=(15, 8))
                ax2 = ax1.twinx()
                
                # Plot time series
                sns.lineplot(
                    x='date', y=top_frag, data=data, ax=ax1,
                    color='blue', marker='o', markersize=8
                )
                
                sns.lineplot(
                    x='date', y=top_ema, data=data, ax=ax2,
                    color='red', marker='s', markersize=8
                )
                
                # Set labels and title
                ax1.set_xlabel('Date', fontsize=18)
                ax1.set_ylabel(top_frag, color='blue', fontsize=18)
                ax2.set_ylabel(top_ema, color='red', fontsize=18)
                
                plt.title(f'Time Series of {top_frag} and {top_ema}', fontsize=24)
                
                # Set tick parameters
                ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
                ax2.tick_params(axis='y', labelcolor='red', labelsize=16)
                ax1.tick_params(axis='x', labelsize=16, rotation=45)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'time_series.png', dpi=300)
                plt.close()
        
        logger.info(f"Saved visualizations to {plots_dir}")
    
    def run_analysis(self):
        """
        Run the complete combined analysis workflow.
        """
        logger.info("Starting combined metrics analysis...")
        
        # 1. Load data
        frag_data = self.load_fragmentation_data()
        ema_data = self.load_ema_data()
        
        if frag_data.empty or ema_data.empty:
            logger.error("Analysis cannot proceed due to missing data")
            return None
        
        # 2. Merge data
        combined_data = self.merge_data(frag_data, ema_data)
        
        if combined_data.empty:
            logger.error("No data after merging")
            return None
        
        # 3. Save the combined dataset
        combined_file = self.output_dir / 'combined_fragmentation_ema.csv'
        combined_data.to_csv(combined_file, index=False)
        logger.info(f"Saved combined dataset to {combined_file}")
        
        # 4. Analyze the data
        analysis_results = self.analyze_combined_data(combined_data)
        
        if analysis_results:
            # Convert numpy types to Python native types for JSON serialization
            serializable_results = {}
            for k, v in analysis_results.items():
                if isinstance(v, dict):
                    serializable_results[k] = {
                        k2: float(v2) if isinstance(v2, (np.float32, np.float64, np.int64)) else v2
                        for k2, v2 in v.items()
                    }
                else:
                    serializable_results[k] = float(v) if isinstance(v, (np.float32, np.float64, np.int64)) else v
            
            # Save analysis results
            results_file = self.output_dir / 'analysis_results.json'
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            logger.info(f"Saved analysis results to {results_file}")
        
        # 5. Generate visualizations
        self.generate_visualizations(combined_data)
        
        logger.info("Analysis complete!")
        return combined_data

def main():
    """Main function to run the analysis."""
    # Define paths using Path objects
    fragmentation_dir = Path('/Volumes/Extreme SSD/SURREAL-DataBackup/HUJI_data-main/processed/fragmentation')
    ema_dir = Path('SURREAL/EMA-Processing/output/normalized')
    output_dir = Path('SURREAL/metrics/output')
    
    # Check if directories exist
    for name, directory in [('Fragmentation', fragmentation_dir), ('EMA', ema_dir)]:
        if not directory.exists():
            logger.error(f"{name} directory not found at: {directory}")
            logger.error("Please check the path and try again.")
            return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run the combiner
    combiner = MetricsCombiner(
        fragmentation_dir=fragmentation_dir,
        ema_dir=ema_dir,
        output_dir=output_dir,
        end_of_day_only=True,  # Only use end-of-day EMA responses
        debug_mode=False       # Set to True for more detailed logging
    )
    
    # Run the analysis
    combined_data = combiner.run_analysis()
    
    # Print summary
    if combined_data is not None and not combined_data.empty:
        logger.info("\n" + "="*50)
        logger.info("COMBINED METRICS ANALYSIS SUMMARY")
        logger.info("="*50)
        
        # Summary statistics
        logger.info(f"\nTotal data points: {len(combined_data)}")
        
        if 'participant_id' in combined_data.columns:
            logger.info(f"Total participants: {combined_data['participant_id'].nunique()}")
        
        # Fragmentation metrics
        frag_cols = [col for col in combined_data.columns if 'fragmentation' in col.lower()]
        logger.info(f"\nFragmentation Metrics: {len(frag_cols)}")
        for col in frag_cols:
            valid_count = combined_data[col].notna().sum()
            logger.info(f"  {col}: {valid_count} valid values ({valid_count/len(combined_data)*100:.1f}%)")
            
            if valid_count > 0:
                logger.info(f"    Mean: {combined_data[col].mean():.4f}")
                logger.info(f"    Std Dev: {combined_data[col].std():.4f}")
        
        # EMA metrics count
        ema_indicators = ['anxiety', 'stress', 'mood', 'STAI', 'CES', 'CALM', 'PEACE', 
                          'SATISFACTION', 'HAPPY', 'ENJOYMENT', 'NERVOUS', 'WORRY']
        ema_cols = [col for col in combined_data.columns if any(
            indicator in col for indicator in ema_indicators
        )]
        
        logger.info(f"\nEMA Metrics: {len(ema_cols)}")
        
        # Output file locations
        logger.info("\nOutput Files:")
        logger.info(f"  Combined data: {output_dir / 'combined_fragmentation_ema.csv'}")
        logger.info(f"  Analysis results: {output_dir / 'analysis_results.json'}")
        logger.info(f"  Visualizations: {output_dir / 'plots/'}")
        
        logger.info("\nAnalysis complete!")
    else:
        logger.error("Analysis failed - no results were generated.")

if __name__ == "__main__":
    main() 