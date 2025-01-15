import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
from datetime import datetime

class FragmentationComparison:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define key variables
        self.predictor_cols = [
            'digital_fragmentation_index',
            'moving_fragmentation_index',
            'digital_frag_during_mobility',
            'digital_total_duration',
            'moving_total_duration'
        ]
        
        self.emotion_cols = [
            'TENSE',
            'RELAXATION_R',
            'WORRY',
            'PEACE_R',
            'IRRITATION',
            'SATISFACTION_R',
            'STAI6_score',
            'HAPPY'
        ]
        
        self._setup_logging()
        
    def _setup_logging(self):
        log_path = self.output_dir / 'fragmentation_analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load and preprocess data"""
        self.logger.info(f"Loading data from {self.input_path}")
        df = pd.read_csv(self.input_path)
        df = df[df['data_quality'] == 'good'].copy()
        self.data = df
        return df

    def get_descriptive_stats(self, split_type='quartile'):
        """Calculate descriptive statistics for high vs low groups"""
        stats_results = []
        
        for pred in self.predictor_cols:
            # Calculate participant means
            participant_means = self.data.groupby('user')[pred].mean()
            
            # Get split point based on type
            if split_type == 'quartile':
                split_point = participant_means.quantile(0.75)
                split_name = '75th percentile'
            else:  # median
                split_point = participant_means.median()
                split_name = 'median'
            
            # Identify high/low fragmentation participants
            high_frag_users = participant_means[participant_means > split_point].index
            high_frag_data = self.data[self.data['user'].isin(high_frag_users)]
            low_frag_data = self.data[~self.data['user'].isin(high_frag_users)]
            
            stats_results.append({
                'predictor': pred,
                'split_type': split_type,
                f'split_point_{split_name}': split_point,
                'overall_mean': self.data[pred].mean(),
                'overall_std': self.data[pred].std(),
                'high_frag_mean': high_frag_data[pred].mean(),
                'high_frag_std': high_frag_data[pred].std(),
                'low_frag_mean': low_frag_data[pred].mean(),
                'low_frag_std': low_frag_data[pred].std(),
                'n_high_frag_participants': len(high_frag_users),
                'n_low_frag_participants': len(participant_means.index) - len(high_frag_users),
                'n_high_frag_observations': len(high_frag_data),
                'n_low_frag_observations': len(low_frag_data)
            })
        
        return pd.DataFrame(stats_results)

    def run_emotion_comparisons(self, split_type='quartile'):
        """Compare emotional outcomes between high/low fragmentation groups"""
        results = []
        
        for pred in self.predictor_cols:
            # Calculate participant means
            participant_means = self.data.groupby('user')[pred].mean()
            
            # Get split point based on type
            if split_type == 'quartile':
                split_point = participant_means.quantile(0.75)
            else:  # median
                split_point = participant_means.median()
            
            # Identify groups
            high_frag_users = participant_means[participant_means > split_point].index
            
            for emotion in self.emotion_cols:
                # Get participant-level emotion means
                emotion_means = self.data.groupby('user')[emotion].mean()
                
                # Compare high vs low fragmentation groups
                high_frag_emotions = emotion_means[high_frag_users]
                low_frag_emotions = emotion_means[~emotion_means.index.isin(high_frag_users)]
                
                # Run t-test
                t_stat, p_val = stats.ttest_ind(
                    high_frag_emotions.dropna(),
                    low_frag_emotions.dropna()
                )
                
                # Calculate effect size
                pooled_std = np.sqrt((high_frag_emotions.std()**2 + low_frag_emotions.std()**2) / 2)
                cohens_d = (high_frag_emotions.mean() - low_frag_emotions.mean()) / pooled_std if pooled_std != 0 else np.nan
                
                results.append({
                    'predictor': pred,
                    'emotion': emotion,
                    'split_type': split_type,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'high_frag_mean': high_frag_emotions.mean(),
                    'low_frag_mean': low_frag_emotions.mean(),
                    'high_frag_std': high_frag_emotions.std(),
                    'low_frag_std': low_frag_emotions.std(),
                    'cohens_d': cohens_d,
                    'n_high': len(high_frag_emotions),
                    'n_low': len(low_frag_emotions)
                })
        
        return pd.DataFrame(results)

    def run_analysis(self):
        """Run the complete analysis pipeline for both split types"""
        # Load data
        self.load_data()
        
        # Get descriptive statistics for both splits
        quartile_stats = self.get_descriptive_stats(split_type='quartile')
        median_stats = self.get_descriptive_stats(split_type='median')
        desc_stats = pd.concat([quartile_stats, median_stats])
        
        # Run emotion comparisons for both splits
        quartile_results = self.run_emotion_comparisons(split_type='quartile')
        median_results = self.run_emotion_comparisons(split_type='median')
        emotion_results = pd.concat([quartile_results, median_results])
        
        # Sort results by p-value within each split type
        emotion_results = emotion_results.sort_values(['split_type', 'p_value'])
        
        # Round numeric columns
        numeric_cols = emotion_results.select_dtypes(include=[np.number]).columns
        emotion_results[numeric_cols] = emotion_results[numeric_cols].round(3)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with pd.ExcelWriter(self.output_dir / f'fragmentation_analysis_{timestamp}.xlsx') as writer:
            desc_stats.to_excel(writer, sheet_name='Descriptive Stats', index=False)
            
            # Save emotion comparisons separately for each split type
            quartile_results.to_excel(writer, sheet_name='Quartile Comparisons', index=False)
            median_results.to_excel(writer, sheet_name='Median Comparisons', index=False)
            
            # Save combined results
            emotion_results.to_excel(writer, sheet_name='All Comparisons', index=False)
        
        # Log summary statistics
        for split_type in ['quartile', 'median']:
            results = emotion_results[emotion_results['split_type'] == split_type]
            sig_results = results[results['p_value'] < 0.05]
            
            self.logger.info(f"\n{split_type.capitalize()} Split Analysis:")
            self.logger.info(f"Found {len(sig_results)} significant relationships (p < 0.05):")
            self.logger.info(sig_results[['predictor', 'emotion', 'p_value', 'cohens_d']])

def main():
    
    input_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/metrics/combined_metrics.csv'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'
    
    analyzer = FragmentationComparison(input_path, output_dir)
    analyzer.run_analysis()
    
    print("Analysis completed successfully")

if __name__ == "__main__":
    main()
    