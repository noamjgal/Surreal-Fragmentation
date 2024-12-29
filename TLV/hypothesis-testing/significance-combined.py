# config.py
class Config:
    def __init__(self):
        self.input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon'
        self.output_dir = self.input_dir + '/output'
        self.survey_file = self.input_dir + '/Survey/End_of_the_day_questionnaire.xlsx'
        self.participant_info_file = self.input_dir + '/participant_info.csv'
        self.frag_indices = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
        self.emotional_outcomes = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']
        self.population_factors = ['Gender', 'Class', 'School']

# data_loader.py
import pandas as pd
import numpy as np
import os
import datetime

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        # Load the fragmentation results
        frag_summary = pd.read_csv(os.path.join(self.config.input_dir, 'fragmentation/fragmentation_summary.csv'))
        # Load the survey responses
        survey_responses = pd.read_excel(self.config.survey_file)
        
        # Load participant info
        participant_info = pd.read_csv(self.config.participant_info_file)
        
        return self._merge_datasets(frag_summary, survey_responses, participant_info)
    
    def _merge_datasets(self, frag_summary, survey_responses, participant_info):
        # Standardize date formats
        frag_summary['date'] = pd.to_datetime(frag_summary['date']).dt.date
        survey_responses['date'] = pd.to_datetime(survey_responses['StartDate']).dt.date
        
        # Ensure consistent ID formats
        for df, id_col in [(frag_summary, 'participant_id'), 
                          (survey_responses, 'Participant_ID'),
                          (participant_info, 'user')]:
            df[id_col] = df[id_col].astype(str)
        
        # Merge datasets
        merged = pd.merge(
            frag_summary, 
            survey_responses,
            left_on=['participant_id', 'date'],
            right_on=['Participant_ID', 'date'],
            how='inner'
        )
        
        merged = pd.merge(
            merged,
            participant_info,
            left_on='participant_id',
            right_on='user',
            how='left'
        )
        
        return merged

# preprocessor.py
class DataPreprocessor:
    @staticmethod
    def preprocess(data):
        data = data.copy()
        
        # Handle missing values
        data = DataPreprocessor._handle_missing_values(data)
        
        # Calculate STAI6 score
        data = DataPreprocessor._calculate_stai6(data)
        
        # Add derived features
        data = DataPreprocessor._add_derived_features(data)
        
        # Create binary gender variable
        data = DataPreprocessor._create_binary_variables(data)
        
        # Handle infinite values
        data = DataPreprocessor._handle_infinite_values(data)
        
        return data
    
    @staticmethod
    def _handle_missing_values(data):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        # Handle numeric columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        # Handle categorical columns (excluding datetime)
        for col in categorical_cols:
            if col not in datetime_cols:
                try:
                    # Get most common value, excluding datetime objects
                    valid_values = data[col][~data[col].apply(lambda x: isinstance(x, (pd.Timestamp, datetime.datetime)))]
                    most_common = valid_values.mode()[0] if not valid_values.empty else None
                    data[col] = data[col].fillna(most_common)
                except (TypeError, ValueError):
                    # If mode fails, use forward/backward fill
                    data[col] = data[col].ffill().bfill()
        
        # Handle datetime columns
        for col in datetime_cols:
            data[col] = data[col].ffill().bfill()
        
        return data
    
    @staticmethod
    def _calculate_stai6(data):
        # Create reversed scores
        for item in ['RELAXATION', 'PEACE', 'SATISFACTION']:
            data[f'{item}_reversed'] = 5 - data[item]
        
        # Calculate STAI6
        stai6_items = ['TENSE', 'RELAXATION_reversed', 'WORRY', 
                      'PEACE_reversed', 'IRRITATION', 'SATISFACTION_reversed']
        data['STAI6_score'] = data[stai6_items].mean(axis=1) * 20/6
        
        # Cleanup
        data = data.drop(columns=[f'{item}_reversed' for item in ['RELAXATION', 'PEACE', 'SATISFACTION']])
        
        return data
    
    @staticmethod
    def _add_derived_features(data):
        data['weekday'] = pd.to_datetime(data['date']).dt.weekday
        data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
        
        # Standardize mobility metrics
        mobility_vars = ['total_duration_mobility', 'avg_duration_mobility', 'count_mobility']
        for var in mobility_vars:
            data[f'{var}_z'] = (data[var] - data[var].mean()) / data[var].std()
        
        return data
    
    @staticmethod
    def _create_binary_variables(data):
        if 'Gender' in data.columns:
            # Hebrew to binary gender mapping
            gender_map = {
                'נקבה': 0,  # Female
                'זכר': 1    # Male
            }
            data['Gender_binary'] = data['Gender'].map(gender_map)
        
        return data
    
    @staticmethod
    def _handle_infinite_values(data):
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, fill NaN with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        return data

# statistical_tests.py
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

class StatisticalAnalyzer:
    def __init__(self, config):
        self.config = config
        # Initialize control variables but filter based on actual data columns
        self.all_potential_controls = [
            'Gender_binary', 'Class', 'School_location',
            'total_duration_mobility_z', 'avg_duration_mobility_z', 'count_mobility_z',
            'is_weekend'
        ]
        self.control_variables = []  # Will be set when analyze_data is called
    
    def correlation_analysis(self, data, var1, var2, control_vars=None):
        """Perform partial correlation if control variables are provided."""
        if control_vars is None:
            corr, p_value = stats.pearsonr(data[var1], data[var2])
            return {
                'type': 'bivariate',
                'correlation': corr,
                'p_value': p_value,
                'n': len(data)
            }
        else:
            # Partial correlation
            control_matrix = data[control_vars]
            partial_corr = pd.DataFrame(
                np.linalg.pinv(data[[var1, var2] + control_vars].corr().values),
                columns=data[[var1, var2] + control_vars].columns,
                index=data[[var1, var2] + control_vars].columns
            )
            r_partial = -partial_corr.iloc[0,1] / np.sqrt(partial_corr.iloc[0,0] * partial_corr.iloc[1,1])
            
            # Calculate p-value for partial correlation
            t_stat = r_partial * np.sqrt((len(data)-len(control_vars)-2)/(1-r_partial**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data)-len(control_vars)-2))
            
            return {
                'type': 'partial',
                'correlation': r_partial,
                'p_value': p_value,
                'control_vars': control_vars,
                'n': len(data)
            }
    
    def set_control_variables(self, data):
        """Set control variables based on available columns in the data."""
        self.control_variables = [var for var in self.all_potential_controls 
                                if var in data.columns]
        
    def regression_analysis(self, data, predictor, outcome, potential_controls=None):
        """Perform regression with automated control variable selection."""
        if not self.control_variables:  # If control variables haven't been set
            self.set_control_variables(data)
            
        if potential_controls is None:
            potential_controls = self.control_variables.copy()  # Create a copy to avoid modifying original
        
        # Base model without controls
        X_base = sm.add_constant(data[predictor])
        base_model = sm.OLS(data[outcome], X_base).fit()
        base_results = {
            'type': 'base',
            'predictor': predictor,
            'outcome': outcome,
            'coefficient': base_model.params[predictor],
            'std_error': base_model.bse[predictor],
            'p_value': base_model.pvalues[predictor],
            'r_squared': base_model.rsquared,
            'adj_r_squared': base_model.rsquared_adj,
            'aic': base_model.aic,
            'bic': base_model.bic,
            'n': len(data),
            'control_vars': []
        }
        
        # Forward stepwise selection of controls
        current_controls = []
        best_model = base_model
        improvement_threshold = 0.01  # Minimum required improvement in R-squared
        
        while potential_controls:
            best_improvement = 0
            best_control = None
            
            for control in potential_controls:
                X_test = sm.add_constant(data[[predictor] + current_controls + [control]])
                test_model = sm.OLS(data[outcome], X_test).fit()
                improvement = test_model.rsquared_adj - best_model.rsquared_adj
                
                if improvement > best_improvement and improvement > improvement_threshold:
                    best_improvement = improvement
                    best_control = control
            
            if best_control:
                current_controls.append(best_control)
                potential_controls.remove(best_control)
                X_best = sm.add_constant(data[[predictor] + current_controls])
                best_model = sm.OLS(data[outcome], X_best).fit()
            else:
                break
        
        if current_controls:
            controlled_results = {
                'type': 'controlled',
                'predictor': predictor,
                'outcome': outcome,
                'coefficient': best_model.params[predictor],
                'std_error': best_model.bse[predictor],
                'p_value': best_model.pvalues[predictor],
                'r_squared': best_model.rsquared,
                'adj_r_squared': best_model.rsquared_adj,
                'aic': best_model.aic,
                'bic': best_model.bic,
                'n': len(data),
                'control_vars': current_controls,
                'control_coefficients': {var: best_model.params[var] for var in current_controls},
                'control_p_values': {var: best_model.pvalues[var] for var in current_controls}
            }
            return [base_results, controlled_results]
        
        return [base_results]
    
    def group_comparison(self, data, group_var, outcome_var, threshold=None, control_vars=None):
        """Perform group comparison with optional controls using ANCOVA."""
        # Remove debug prints:
        # print("Available columns:", data.columns.tolist())
        # print(f"Looking for group_var: {group_var}")
        
        if group_var == 'Gender':
            # Create binary gender groups directly from Hebrew values
            group1 = data[data[group_var] == 'נקבה'][outcome_var]  # Female
            group2 = data[data[group_var] == 'זכר'][outcome_var]   # Male
        else:
            if threshold is not None:
                data = data.copy()
                data[group_var] = data[group_var] > threshold
                
                # Basic t-test
                group1 = data[data[group_var]][outcome_var]
                group2 = data[~data[group_var]][outcome_var]
            else:
                # For other categorical variables
                unique_values = sorted(data[group_var].unique())
                group1 = data[data[group_var] == unique_values[0]][outcome_var]
                group2 = data[data[group_var] == unique_values[1]][outcome_var]
        
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        effect_size = (group1.mean() - group2.mean()) / np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        
        base_results = {
            'type': 'unadjusted',
            'test': 't_test',
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            'group1_std': group1.std(),
            'group2_std': group2.std(),
            'group1_n': len(group1),
            'group2_n': len(group2)
        }
        
        if control_vars:
            # For ANCOVA with Gender, create a binary variable first
            data_copy = data.copy()
            if group_var == 'Gender':
                data_copy['Gender_binary'] = data_copy[group_var].map({'נקבה': 0, 'זכר': 1})
                formula = f"{outcome_var} ~ C(Gender_binary) + " + " + ".join(control_vars)
            else:
                formula = f"{outcome_var} ~ C({group_var}) + " + " + ".join(control_vars)
            
            model = sm.OLS.from_formula(formula, data=data_copy).fit()
            
            # Get the correct parameter name based on the variable type
            if group_var == 'Gender':
                param_name = "C(Gender_binary)[T.1]"
            else:
                param_name = f"C({group_var})[T.True]" if threshold is not None else f"C({group_var})[T.{unique_values[1]}]"
            
            controlled_results = {
                'type': 'adjusted',
                'test': 'ancova',
                'coefficient': model.params[param_name],
                'std_error': model.bse[param_name],
                'p_value': model.pvalues[param_name],
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'control_vars': control_vars,
                'control_coefficients': {var: model.params[var] for var in control_vars},
                'control_p_values': {var: model.pvalues[var] for var in control_vars},
                'n': len(data)
            }
            return [base_results, controlled_results]
        
        return [base_results]

# results_manager.py
class ResultsManager:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
    
    def save_results(self, results_dict, filename):
        output_path = os.path.join(self.config.output_dir, filename)
        
        if isinstance(results_dict, pd.DataFrame):
            results_dict.to_csv(output_path, index=False)
        else:
            pd.DataFrame(results_dict).to_csv(output_path, index=False)
            
        print(f"Results saved to: {output_path}")
    
    def apply_multiple_testing_correction(self, results_df, group_cols, p_value_col='p_value'):
        for group in results_df.groupby(group_cols):
            mask = (results_df[group_cols] == group[0]).all(axis=1)
            results_df.loc[mask, 'FDR_p_value'] = multipletests(
                results_df.loc[mask, p_value_col].fillna(1),
                method='fdr_bh'
            )[1]
        return results_df

# analysis_runner.py
class AnalysisRunner:
    def __init__(self, config, analyzer, results_manager):
        self.config = config
        self.analyzer = analyzer
        self.results_manager = results_manager
    
    def run_mobility_analysis(self, data):
        results = []
        for frag_index in self.config.frag_indices:
            for mobility_metric in ['total_duration_mobility', 'avg_duration_mobility', 'count_mobility']:
                # Correlation analysis
                corr_results = self.analyzer.correlation_analysis(data, frag_index, mobility_metric)
                
                # Regression analysis with control optimization
                reg_results = self.analyzer.regression_analysis(data, frag_index, mobility_metric)
                
                # Group comparison using median split
                group_results = self.analyzer.group_comparison(
                    data, frag_index, mobility_metric,
                    threshold=data[frag_index].median()
                )
                
                results.extend([{
                    'analysis_type': 'mobility',
                    'frag_index': frag_index,
                    'mobility_metric': mobility_metric,
                    **result
                } for result in [corr_results] + reg_results + group_results])
        
        return pd.DataFrame(results)
    
    def run_emotional_analysis(self, data):
        results = []
        for frag_index in self.config.frag_indices:
            for outcome in self.config.emotional_outcomes:
                # Correlation analysis
                corr_results = self.analyzer.correlation_analysis(data, frag_index, outcome)
                
                # Regression with control optimization
                reg_results = self.analyzer.regression_analysis(data, frag_index, outcome)
                
                # Group comparison using both median and quartile splits
                for threshold in [data[frag_index].median(), data[frag_index].quantile(0.75)]:
                    group_results = self.analyzer.group_comparison(
                        data, frag_index, outcome,
                        threshold=threshold
                    )
                
                    results.extend([{
                        'analysis_type': 'emotional',
                        'frag_index': frag_index,
                        'outcome': outcome,
                        'threshold_type': 'median' if threshold == data[frag_index].median() else 'quartile',
                        **result
                    } for result in [corr_results] + reg_results + group_results])
        
        return pd.DataFrame(results)
    
    def run_population_analysis(self, data):
        results = []
        for factor in self.config.population_factors:
            for frag_index in self.config.frag_indices:
                # Group comparison with controls
                group_results = self.analyzer.group_comparison(
                    data, factor, frag_index,
                    control_vars=[var for var in self.analyzer.control_variables if var != factor]
                )
                
                results.extend([{
                    'analysis_type': 'population',
                    'factor': factor,
                    'frag_index': frag_index,
                    **result
                } for result in group_results])
        
        return pd.DataFrame(results)

# main.py
def main():
    # Initialize components
    config = Config()
    loader = DataLoader(config)
    preprocessor = DataPreprocessor()
    analyzer = StatisticalAnalyzer(config)
    results_manager = ResultsManager(config)
    
    # Initialize analysis runner
    runner = AnalysisRunner(config, analyzer, results_manager)
    
    # Load and preprocess data
    raw_data = loader.load_data()
    processed_data = preprocessor.preprocess(raw_data)
    
    # Set control variables based on available data columns
    analyzer.set_control_variables(processed_data)
    
    # Run analyses
    analysis_functions = {
        'mobility': runner.run_mobility_analysis,
        'emotional': runner.run_emotional_analysis,
        'population': runner.run_population_analysis
    }
    
    for analysis_type, analysis_func in analysis_functions.items():
        # Run analysis
        results_df = analysis_func(processed_data)
        
        # Apply multiple testing correction within each result type
        if 'p_value' in results_df.columns:
            results_df = results_manager.apply_multiple_testing_correction(
                results_df,
                ['analysis_type', 'type']
            )
        
        # Split results by test type
        for test_type in results_df['type'].unique():
            test_results = results_df[results_df['type'] == test_type]
            results_manager.save_results(
                test_results,
                f'{analysis_type}_{test_type}_analysis.csv'
            )
        
        # Save summary of significant findings
        significant_results = results_df[
            (results_df['p_value'] < 0.05) | 
            (results_df.get('FDR_p_value', 1) < 0.05)
        ]
        results_manager.save_results(
            significant_results,
            f'{analysis_type}_significant_findings.csv'
        )

if __name__ == "__main__":
    main()