import pandas as pd
import numpy as np
import datetime

# preprocessor.py
class DataPreprocessor:
    @staticmethod
    def preprocess(data):
        data = data.copy()
        data = DataPreprocessor._handle_missing_values(data)
        data = DataPreprocessor._calculate_stai6(data)
        data = DataPreprocessor._add_derived_features(data)
        data = DataPreprocessor._create_binary_variables(data)
        data = DataPreprocessor._handle_infinite_values(data)
        return data
    
    @staticmethod
    def _handle_missing_values(data):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        for col in categorical_cols:
            if col not in datetime_cols:
                try:
                    valid_values = data[col][~data[col].apply(lambda x: isinstance(x, (pd.Timestamp, datetime.datetime)))]
                    most_common = valid_values.mode()[0] if not valid_values.empty else None
                    data[col] = data[col].fillna(most_common)
                except (TypeError, ValueError):
                    data[col] = data[col].ffill().bfill()
        
        for col in datetime_cols:
            data[col] = data[col].ffill().bfill()
        
        return data
    
    @staticmethod
    def _calculate_stai6(data):
        for item in ['RELAXATION', 'PEACE', 'SATISFACTION']:
            data[f'{item}_reversed'] = 5 - data[item]
        
        stai6_items = ['TENSE', 'RELAXATION_reversed', 'WORRY', 
                      'PEACE_reversed', 'IRRITATION', 'SATISFACTION_reversed']
        data['STAI6_score'] = data[stai6_items].mean(axis=1) * 20/6
        
        data = data.drop(columns=[f'{item}_reversed' for item in ['RELAXATION', 'PEACE', 'SATISFACTION']])
        return data
    
    @staticmethod
    def _add_derived_features(data):
        data['weekday'] = pd.to_datetime(data['date']).dt.weekday
        data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
        
        mobility_vars = ['total_duration_mobility', 'avg_duration_mobility', 'count_mobility']
        for var in mobility_vars:
            data[f'{var}_z'] = (data[var] - data[var].mean()) / data[var].std()
        
        return data
    
    @staticmethod
    def _create_binary_variables(data):
        if 'Gender' in data.columns:
            gender_map = {
                'נקבה': 0,  # Female
                'זכר': 1    # Male
            }
            data['Gender_binary'] = data['Gender'].map(gender_map)
        return data
    
    @staticmethod
    def _handle_infinite_values(data):
        data = data.replace([np.inf, -np.inf], np.nan)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        return data