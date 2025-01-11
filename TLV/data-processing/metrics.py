import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

class MetricsProcessor:
    """
    A comprehensive processor for calculating and preparing metrics for hypothesis testing.
    Handles EMA scoring, digital usage metrics, mobility metrics, and control variables.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the metrics processor"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'metrics_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_ema_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process EMA responses and calculate STAI-6 index score.
        Now preserves demographic columns for later analysis.
        """
        df = data.copy()
        
        # Log initial columns
        self.logger.info("Initial EMA columns: %s", df.columns.tolist())
        
        # Store demographic columns
        demographic_cols = ['Gender', 'School', 'Class']
        
        # Reverse scoring for positive items (5-point scale)
        reverse_items = ['PEACE', 'RELAXATION', 'SATISFACTION']
        for item in reverse_items:
            df[f'{item}_R'] = 6 - df[item]
            # Drop original positive items
            df = df.drop(columns=[item])
        
        # Calculate STAI-6 score (20-80 range)
        stai_items = ['TENSE', 'RELAXATION_R', 'WORRY', 
                    'PEACE_R', 'IRRITATION', 'SATISFACTION_R']
        
        # Calculate mean and transform to 20-80 range
        df['STAI6_score'] = df[stai_items].mean(axis=1) * 20
        
        # Keep necessary columns including demographics
        columns_to_keep = ['Participant_ID', 'StartDate'] + demographic_cols + stai_items + ['STAI6_score', 'HAPPY']
        df = df[columns_to_keep]
        
        # Log final columns
        self.logger.info("Final EMA columns after processing: %s", df.columns.tolist())
        
        return df

    def calculate_tertiles(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Calculate tertiles for four key metrics:
        1. Digital fragmentation index
        2. Total time on device
        3. Total mobility duration
        4. Mobility fragmentation index
        
        Parameters:
            data (pd.DataFrame): Input data containing the required metrics
            
        Returns:
            Dict[str, Tuple[float, float]]: Dictionary of tertile cutoffs for each metric
        """
        tertiles = {}
        tertile_metrics = [
            'digital_fragmentation_index',
            'total_time_on_device',
            'total_duration_mobility',
            'moving_fragmentation_index'
        ]
        
        self.logger.info("Calculating tertiles for metrics: %s", tertile_metrics)
        
        for metric in tertile_metrics:
            try:
                lower, upper = np.percentile(data[metric].dropna(), [33.33, 66.67])
                tertiles[metric] = (lower, upper)
                self.logger.info(f"Tertiles for {metric}: Lower={lower:.2f}, Upper={upper:.2f}")
            except Exception as e:
                self.logger.error(f"Error calculating tertiles for {metric}: {str(e)}")
        
        return tertiles
        tertiles = {}
        for col in columns:
            try:
                lower, upper = np.percentile(data[col].dropna(), [33.33, 66.67])
                tertiles[col] = (lower, upper)
                self.logger.info(f"Tertiles for {col}: Lower={lower:.2f}, Upper={upper:.2f}")
            except Exception as e:
                self.logger.error(f"Error calculating tertiles for {col}: {str(e)}")
        return tertiles

    def assign_tertile_groups(self, data: pd.DataFrame, tertiles: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Assign tertile groups to data based on calculated cutoffs
        """
        df = data.copy()
        for col, (lower, upper) in tertiles.items():
            group_col = f"{col}_group"
            df[group_col] = pd.cut(
                df[col],
                bins=[-np.inf, lower, upper, np.inf],
                labels=['low', 'medium', 'high']
            )
        return df

    def calculate_control_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and normalize control variables
        """
        df = data.copy()
        
        # Calculate daily averages
        control_vars = [
            'total_time_on_device',
            'digital_fragmentation_index',
            'total_duration_mobility',
            'moving_fragmentation_index'
        ]
        
        # Calculate z-scores for control variables
        for var in control_vars:
            if var in df.columns:
                z_score_col = f"{var}_zscore"
                df[z_score_col] = (df[var] - df[var].mean()) / df[var].std()
        
        # Add time-based controls
        df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        return df

    def prepare_for_hypothesis_testing(self, 
                                     fragmentation_data: pd.DataFrame, 
                                     ema_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final dataset for hypothesis testing by combining and processing all metrics.
        
        Creates four separate tertile groupings:
        1. Digital Fragmentation Tertiles
        2. Digital Usage Time Tertiles
        3. Mobility Duration Tertiles
        4. Mobility Fragmentation Tertiles
        """
        self.logger.info("Starting preparation for hypothesis testing")
        
        try:
            # Process EMA scores
            processed_ema = self.process_ema_scores(ema_data)
            
            # Convert date formats
            processed_ema['date'] = pd.to_datetime(processed_ema['StartDate']).dt.strftime('%Y-%m-%d')
            fragmentation_data['date'] = pd.to_datetime(fragmentation_data['date']).dt.strftime('%Y-%m-%d')
            
            # Ensure participant IDs are strings in both datasets
            processed_ema['participant_id'] = processed_ema['Participant_ID'].astype(str)
            fragmentation_data['participant_id'] = fragmentation_data['participant_id'].astype(str)
            
            # Debug information
            self.logger.info("\nDate ranges:")
            self.logger.info(f"Fragmentation data: {fragmentation_data['date'].min()} to {fragmentation_data['date'].max()}")
            self.logger.info(f"EMA data: {processed_ema['date'].min()} to {processed_ema['date'].max()}")
            
            self.logger.info("\nParticipant IDs:")
            self.logger.info(f"Fragmentation data unique IDs: {sorted(fragmentation_data['participant_id'].unique())}")
            self.logger.info(f"EMA data unique IDs: {sorted(processed_ema['participant_id'].unique())}")
            
            # Merge with fragmentation data
            merged_data = pd.merge(
                fragmentation_data,
                processed_ema,
                on=['participant_id', 'date'],
                how='inner'
            )
            
            self.logger.info(f"\nMerged data shape: {merged_data.shape}")
            if merged_data.empty:
                self.logger.error("No matching records found after merge!")
                # Sample records from both datasets for debugging
                self.logger.info("\nSample fragmentation records:")
                self.logger.info(fragmentation_data[['participant_id', 'date']].head())
                self.logger.info("\nSample EMA records:")
                self.logger.info(processed_ema[['participant_id', 'date']].head())
                return None
            
            # Calculate tertiles
            tertiles = self.calculate_tertiles(merged_data)
            
            # Assign tertile groups
            for metric, (lower, upper) in tertiles.items():
                group_col = f"{metric}_group"
                merged_data[group_col] = pd.cut(
                    merged_data[metric],
                    bins=[-np.inf, lower, upper, np.inf],
                    labels=['low', 'medium', 'high']
                )
            
            # Add time-based controls
            merged_data['weekday'] = pd.to_datetime(merged_data['date']).dt.dayofweek
            merged_data['is_weekend'] = merged_data['weekday'].isin([5, 6]).astype(int)
            
            # Save processed data
            output_path = self.output_dir / 'prepared_metrics.csv'
            merged_data.to_csv(output_path, index=False)
            self.logger.info(f"Saved prepared metrics to {output_path}")
            
            # Generate summary statistics
            self._generate_summary_statistics(merged_data)
            
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error in prepare_for_hypothesis_testing: {str(e)}")
            raise

    def _generate_summary_statistics(self, data: pd.DataFrame):
        """Generate and save summary statistics for the prepared metrics"""
        try:
            summary_stats = {}
            
            # Calculate basic statistics for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            summary_stats['numeric'] = data[numeric_cols].describe()
            
            # Calculate group distributions
            group_cols = [col for col in data.columns if col.endswith('_group')]
            for col in group_cols:
                summary_stats[f'{col}_distribution'] = data[col].value_counts()
                
            # Add demographic summaries - handle each column separately to avoid type conflicts
            for demo_col in ['Gender', 'School', 'Class']:
                if demo_col in data.columns:
                    # Convert to string type to ensure consistent handling
                    counts = data[demo_col].astype(str).value_counts()
                    sheet_name = f'{demo_col}_distribution'
                    summary_stats[sheet_name] = pd.DataFrame({
                        demo_col: counts.index,
                        'count': counts.values
                    })
            
            # Add cross-tabulations of demographics with tertile groups
            for group_col in group_cols:
                for demo_col in ['Gender', 'School', 'Class']:
                    if demo_col in data.columns:
                        # Convert demographic column to string for consistent handling
                        demo_data = data[demo_col].astype(str)
                        tab_name = f'{demo_col}_{group_col}_crosstab'
                        summary_stats[tab_name] = pd.crosstab(
                            demo_data,
                            data[group_col]
                        )
            
            # Save summary statistics
            with pd.ExcelWriter(self.output_dir / 'summary_statistics.xlsx') as writer:
                for name, stats in summary_stats.items():
                    stats.to_excel(writer, sheet_name=name[:31])  # Excel sheet names limited to 31 chars
                    
        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {str(e)}", exc_info=True)
            # Continue execution even if summary statistics fail
            pass

def main():
    # Define paths
    output_dir = Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/metrics')
    fragmentation_path = Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/fragmentation/fragmentation_summary.csv')
    ema_path = Path('/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/End_of_the_day_questionnaire.xlsx')
    
    # Initialize processor
    processor = MetricsProcessor(output_dir)
    
    # Load data and log column names
    fragmentation_data = pd.read_csv(fragmentation_path)
    processor.logger.info("Fragmentation data columns: %s", fragmentation_data.columns.tolist())
    
    ema_data = pd.read_excel(ema_path)
    processor.logger.info("EMA data columns: %s", ema_data.columns.tolist())
    
    # Initialize processor
    processor = MetricsProcessor(output_dir)
    
    # Load data
    fragmentation_data = pd.read_csv(fragmentation_path)
    ema_data = pd.read_excel(ema_path)
    
    # Process and prepare metrics
    prepared_data = processor.prepare_for_hypothesis_testing(
        fragmentation_data,
        ema_data
    )
    
    print("Metrics processing completed successfully")

if __name__ == "__main__":
    main()