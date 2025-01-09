# digital_usage_processor.py
import pandas as pd
import numpy as np

# digital_usage_processor.py
import pandas as pd
import numpy as np
import logging
import os

class DigitalUsageProcessor:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    def calculate_usage_metrics(self, data):
        """Calculate digital usage metrics and create user groups"""
        # Check required columns
        required_cols = ['participant_id', 'date', 'digital_fragmentation_index']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Calculate daily metrics for all participants
        daily_metrics = data.groupby(['participant_id', 'date']).agg({
            'digital_fragmentation_index': 'first',
        }).reset_index()

        # Calculate user-level metrics
        user_metrics = daily_metrics.groupby('participant_id').agg({
            'digital_fragmentation_index': ['count', 'mean', 'std'],
        })
        
        # Flatten column names
        user_metrics.columns = ['days_with_data', 'avg_fragmentation', 'std_fragmentation']
        user_metrics = user_metrics.reset_index()

        # Create numeric groups (1: low, 2: medium, 3: high)
        def assign_group(row):
            if pd.isna(row['avg_fragmentation']):
                return np.nan
            elif row['days_with_data'] >= 3:
                valid_users = user_metrics[user_metrics['days_with_data'] >= 3]
                terciles = np.percentile(valid_users['avg_fragmentation'].dropna(), [33.33, 66.67])
                
                if row['avg_fragmentation'] <= terciles[0]:
                    return 1  # low
                elif row['avg_fragmentation'] <= terciles[1]:
                    return 2  # medium
                else:
                    return 3  # high
            else:
                all_users_terciles = np.percentile(user_metrics['avg_fragmentation'].dropna(), [33.33, 66.67])
                if row['avg_fragmentation'] <= all_users_terciles[0]:
                    return 1  # low
                elif row['avg_fragmentation'] <= all_users_terciles[1]:
                    return 2  # medium
                else:
                    return 3  # high
        
        user_metrics['digital_usage_group'] = user_metrics.apply(assign_group, axis=1)
        
        # Save detailed group information if output directory is provided
        if self.output_dir:
            group_stats = user_metrics.groupby('digital_usage_group').agg({
                'participant_id': 'count',
                'days_with_data': ['mean', 'min', 'max'],
                'avg_fragmentation': ['mean', 'std']
            }).round(2)
            
            group_file = os.path.join(self.output_dir, 'digital_usage_groups.csv')
            group_stats.to_csv(group_file)
            user_metrics.to_csv(os.path.join(self.output_dir, 'user_metrics.csv'))
            
            self.logger.info("\nDigital Usage Group Distribution:")
            self.logger.info(user_metrics['digital_usage_group'].value_counts().sort_index())
            self.logger.info("\nGroup Statistics saved to: " + group_file)
        
        return user_metrics

    def add_usage_metrics(self, data, user_metrics):
        """Add usage metrics and groups to main dataset"""
        enhanced_data = data.merge(
            user_metrics[['participant_id', 'digital_usage_group', 'days_with_data', 
                         'avg_fragmentation', 'std_fragmentation']],
            on='participant_id',
            how='left'
        )
        
        # Create normalized versions of metrics
        metrics_to_normalize = ['days_with_data', 'avg_fragmentation', 'std_fragmentation']
        for metric in metrics_to_normalize:
            enhanced_data[f'{metric}_z'] = (
                enhanced_data[metric] - enhanced_data[metric].mean()
            ) / enhanced_data[metric].std()
        
        return enhanced_data