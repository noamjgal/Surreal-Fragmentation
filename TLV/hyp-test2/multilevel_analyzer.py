# multilevel_analyzer.py
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
import numpy as np
from scipy import stats
import logging

class MultilevelAnalyzer:
    def __init__(self, config):
        self.config = config
        self.min_group_size = 2
        self.min_obs_per_participant = 1
        
        # Enhanced logging setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
            # Add file handler for persistent logging
            file_handler = logging.FileHandler('multilevel_analysis.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def determine_analysis_type(self, data):
        """Determine appropriate analysis type based on data structure"""
        participant_counts = data.groupby('participant_id').size()
        
        analysis_types = {
            'multilevel_eligible': participant_counts[participant_counts >= 2].index,
            'between_subject': participant_counts[participant_counts == 1].index,
            'total_participants': len(participant_counts),
            'avg_observations': participant_counts.mean()
        }
        
        return analysis_types

    def run_analysis(self, data, predictor, outcome, group_var='digital_usage_group'):
        """Enhanced analysis that adapts to data structure"""
        try:
            self.logger.info(f"\nAnalyzing {predictor} → {outcome}")
            data = data.copy()
            
            # Initial diagnostics
            self.print_initial_diagnostics(data, predictor, outcome, group_var)
            
            # Determine analysis approaches
            analysis_structure = self.determine_analysis_type(data)
            
            # Prepare results container
            results = {
                'multilevel': None,
                'between_subject': None,
                'correlational': None,
                'data_structure': analysis_structure
            }
            
            # 1. Run Multilevel Analysis if possible
            if len(analysis_structure['multilevel_eligible']) >= 2:
                mlm_data = data[data['participant_id'].isin(analysis_structure['multilevel_eligible'])]
                results['multilevel'] = self.run_multilevel_model(mlm_data, predictor, outcome, group_var)
            
            # 2. Run Between-Subject Analysis
            between_data = data.groupby('participant_id').agg({
                predictor: 'mean',
                outcome: 'mean',
                group_var: 'first'
            }).reset_index()
            results['between_subject'] = self.run_between_subject_analysis(
                between_data, predictor, outcome, group_var
            )
            
            # 3. Run Basic Correlational Analysis
            results['correlational'] = self.run_correlational_analysis(
                data, predictor, outcome
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            return None

    def run_multilevel_model(self, data, predictor, outcome, group_var):
        """Run multilevel model with robust fitting"""
        try:
            # Standardize variables
            data[predictor + '_std'] = (data[predictor] - data[predictor].mean()) / data[predictor].std()
            data[outcome + '_proc'] = self.preprocess_outcome(data[outcome])
            
            # Setup model
            y = data[outcome + '_proc']
            X = sm.add_constant(data[predictor + '_std'])
            groups = pd.Categorical(data['participant_id']).codes
            
            # Try different fitting approaches
            for method in ['lbfgs', 'cg']:
                try:
                    md = MixedLM(y, X, groups=groups)
                    model = md.fit(reml=True, method=method, maxiter=2000)
                    if model.converged:
                        return model
                except Exception as e:
                    self.logger.warning(f"Method {method} failed: {str(e)}")
            
            return self.run_robust_fallback(data, predictor, outcome)
            
        except Exception as e:
            self.logger.error(f"Multilevel model failed: {str(e)}")
            return None

    def run_between_subject_analysis(self, data, predictor, outcome, group_var):
        """Run between-subject analysis"""
        try:
            # Run regression
            X = sm.add_constant(data[predictor])
            model = sm.OLS(data[outcome], X).fit()
            
            # Run group comparisons if enough data
            group_effects = None
            if len(data[group_var].unique()) >= 2:
                group_effects = self.analyze_group_differences(
                    data, predictor, outcome, group_var
                )
            
            return {
                'regression': model,
                'group_effects': group_effects
            }
            
        except Exception as e:
            self.logger.error(f"Between-subject analysis failed: {str(e)}")
            return None

    def run_correlational_analysis(self, data, predictor, outcome):
        """Run basic correlational analysis"""
        try:
            # Pearson correlation
            pearson_r, p_value = stats.pearsonr(data[predictor], data[outcome])
            
            # Spearman correlation (robust to non-normality)
            spearman_r, spearman_p = stats.spearmanr(data[predictor], data[outcome])
            
            return {
                'pearson': {'r': pearson_r, 'p': p_value},
                'spearman': {'r': spearman_r, 'p': spearman_p},
                'n': len(data)
            }
        except Exception as e:
            self.logger.error(f"Correlational analysis failed: {str(e)}")
            return None

    def print_initial_diagnostics(self, data, predictor, outcome, group_var):
        """Print comprehensive initial diagnostics"""
        self.logger.info("\nOutcome Variable Diagnostics:")
        outcome_data = data[outcome]
        self.logger.info(f"Type: {outcome_data.dtype}")
        self.logger.info(f"Range: {outcome_data.min()} - {outcome_data.max()}")
        self.logger.info("\nDistribution Summary:")
        self.logger.info(outcome_data.describe())
        
        if outcome_data.dtype in ['int64', 'int32']:
            value_counts = outcome_data.value_counts().sort_index()
            self.logger.info("\nValue Counts:")
            self.logger.info(value_counts)
        
        self.logger.info("\nPredictor Variable Diagnostics:")
        pred_data = data[predictor]
        self.logger.info(f"Type: {pred_data.dtype}")
        self.logger.info(f"Range: {pred_data.min()} - {pred_data.max()}")
        self.logger.info("\nDistribution Summary:")
        self.logger.info(pred_data.describe())
        
        self.logger.info("\nGrouping Variable Diagnostics:")
        group_counts = data.groupby(group_var)['participant_id'].nunique()
        self.logger.info("\nParticipants per group:")
        self.logger.info(group_counts)
        
        self.logger.info("\nObservations per participant:")
        obs_counts = data.groupby('participant_id').size()
        self.logger.info(obs_counts.describe())
        
    def preprocess_outcome(self, outcome_data):
        """Preprocess outcome variable based on its characteristics"""
        if outcome_data.dtype in ['int64', 'int32'] and len(outcome_data.unique()) <= 7:
            # Ordinal/Likert data - convert to float but keep scale
            return outcome_data.astype(float)
        else:
            # Continuous data - standardize
            return (outcome_data - outcome_data.mean()) / outcome_data.std()

    def analyze_group_differences(self, data, predictor, outcome, group_var):
        """Analyze differences between groups"""
        try:
            # ANOVA
            groups = [group for _, group in data.groupby(group_var)[outcome]]
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Effect size (eta-squared)
            def eta_squared(f_stat, groups):
                n = sum(len(g) for g in groups)
                k = len(groups)
                return (f_stat * (k-1)) / (f_stat * (k-1) + (n-k))
            
            return {
                'test': 'anova',
                'f_statistic': f_stat,
                'p_value': p_value,
                'effect_size': eta_squared(f_stat, groups),
                'groups': data[group_var].unique().tolist(),
                'n_per_group': data.groupby(group_var).size().to_dict()
            }
        except Exception as e:
            self.logger.error(f"Group difference analysis failed: {str(e)}")
            return None

    def run_robust_fallback(self, data, predictor, outcome):
        """Fallback analysis when MLM fails"""
        try:
            # Aggregate to participant level
            agg_data = data.groupby('participant_id').agg({
                predictor: ['mean', 'std'],
                outcome: ['mean', 'std']
            }).reset_index()
            
            # Run robust regression
            X = sm.add_constant(agg_data[(predictor, 'mean')])
            model = sm.RLM(agg_data[(outcome, 'mean')], X).fit()
            
            return {
                'method': 'robust_regression',
                'coefficient': model.params[1],
                'std_error': model.bse[1],
                'p_value': 2 * (1 - stats.t.cdf(abs(model.params[1] / model.bse[1]), len(agg_data) - 2)),
                'n_participants': len(agg_data)
            }
        except Exception as e:
            self.logger.error(f"Robust fallback failed: {str(e)}")
            return None