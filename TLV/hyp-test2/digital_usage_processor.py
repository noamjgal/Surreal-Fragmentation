# digital_usage_processor.py
import pandas as pd
import numpy as np

class DigitalUsageProcessor:
    @staticmethod
    def calculate_usage_metrics(data):
        """Calculate digital usage metrics and create user groups"""
        # Ensure we have all the necessary columns
        required_cols = ['participant_id', 'date', 'digital_fragmentation_index', 'total_duration_mobility']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Calculate daily digital metrics
        daily_metrics = data.groupby(['participant_id', 'date']).agg({
            'digital_fragmentation_index': 'first',
            'total_duration_mobility': 'first'  # Used as a proxy for day coverage
        }).reset_index()

        # Calculate user-level metrics
        user_metrics = daily_metrics.groupby('participant_id').agg({
            'digital_fragmentation_index': ['count', 'mean', 'std'],  # Include count of days and variability
            'total_duration_mobility': 'mean'
        })
        
        # Flatten column names
        user_metrics.columns = ['days_with_data', 'avg_fragmentation', 'std_fragmentation', 'avg_day_coverage']
        user_metrics = user_metrics.reset_index()

        # Create groups using average fragmentation
        # Filter out users with too few days of data
        min_days = 3  # Adjust this threshold as needed
        valid_users = user_metrics[user_metrics['days_with_data'] >= min_days]
        
        if len(valid_users) > 0:
            terciles = np.percentile(valid_users['avg_fragmentation'].dropna(), [33.33, 66.67])
            
            def assign_group(row):
                if row['days_with_data'] < min_days:
                    return 'insufficient_data'
                elif pd.isna(row['avg_fragmentation']):
                    return 'missing_data'
                elif row['avg_fragmentation'] <= terciles[0]:
                    return 'low'
                elif row['avg_fragmentation'] <= terciles[1]:
                    return 'medium'
                else:
                    return 'high'
            
            user_metrics['digital_usage_group'] = user_metrics.apply(assign_group, axis=1)
            
            # Print distribution of groups
            print("\nDigital Usage Group Distribution:")
            print(user_metrics['digital_usage_group'].value_counts())
            
            # Print group statistics
            print("\nGroup Statistics:")
            stats = user_metrics.groupby('digital_usage_group').agg({
                'avg_fragmentation': ['mean', 'std', 'count'],
                'days_with_data': ['mean', 'min', 'max']
            })
            print(stats)
            
            return user_metrics
        else:
            raise ValueError("No users have sufficient data for grouping")

    @staticmethod
    def add_usage_metrics(data, user_metrics):
        """Add usage metrics and groups to main dataset"""
        # Merge metrics back to main dataset
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