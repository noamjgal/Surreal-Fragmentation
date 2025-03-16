#!/usr/bin/env python3
"""
Simplified Multilevel Analysis of Fragmentation Effects

This script performs multilevel regression analysis to examine the relationship 
between fragmentation metrics and emotional outcomes while properly accounting
for within-participant variance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging
from datetime import datetime
import warnings

class SimplifiedMultilevelAnalysis:
    def __init__(self, output_dir=None, debug=False, data_path=None):
        """Initialize the multilevel analysis.
        
        Args:
            output_dir (str): Directory to save outputs
            debug (bool): Enable debug logging
            data_path (str): Path to pooled data file (override default)
        """
        # Set paths for standardized data
        self.pooled_data_path = Path(data_path) if data_path else Path("pooled/processed/pooled_stai_data_population.csv")
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = Path(__file__).parent
            self.output_dir = script_dir / "results" / "multilevel_analysis"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        self.model_results = {}
        
        # Define fragmentation predictors
        self.fragmentation_predictors = [
            'digital_fragmentation', 
            'mobility_fragmentation', 
            'overlap_fragmentation'
        ]
        
        # Define outcome variables
        self.outcome_variables = [
            'anxiety_score_std',   # Standardized anxiety score
            'mood_score_std'       # Standardized mood/depression score
        ]
        
        # Define subset variables for interaction testing
        self.subset_variables = {
            'dataset': 'dataset_source',
            'gender': 'gender_standardized',
            'age': 'age_group',
            'location': 'location_type'
        }
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'multilevel_analysis_{timestamp}.log'
        
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing simplified multilevel analysis of fragmentation effects")
        self.logger.info(f"Pooled data path: {self.pooled_data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load pooled data and prepare for multilevel modeling.
        
        Returns:
            DataFrame: Pooled dataset with within and between-person variables
        """
        self.logger.info(f"Loading pooled data from {self.pooled_data_path}")
        try:
            df = pd.read_csv(self.pooled_data_path)
            self.logger.info(f"Pooled data loaded successfully with shape: {df.shape}")
            
            # Replace hyphens in column names with underscores for easier referencing
            df.columns = [col.replace('-', '_') for col in df.columns]
            
            # Update variables to match the new column names if needed
            self.outcome_variables = [var.replace('-', '_') for var in self.outcome_variables]
            self.fragmentation_predictors = [var.replace('-', '_') for var in self.fragmentation_predictors]
            
            # Check for required columns
            required_vars = self.fragmentation_predictors + self.outcome_variables + ['participant_id']
            missing_vars = [var for var in required_vars if var not in df.columns]
            if missing_vars:
                self.logger.error(f"Missing critical variables: {missing_vars}")
                self.logger.error("Available columns: " + ", ".join(df.columns.tolist()))
                return None
            
            # Create decomposition of predictors (within and between components)
            self.logger.info("Decomposing fragmentation metrics into within and between-person components")
            for predictor in self.fragmentation_predictors:
                # Calculate person mean (between-person component)
                df[f'{predictor}_between'] = df.groupby('participant_id')[predictor].transform('mean')
                
                # Calculate person-centered values (within-person component)
                df[f'{predictor}_within'] = df[predictor] - df[f'{predictor}_between']
                
                # Log decomposition stats
                within_var = df[f'{predictor}_within'].var()
                between_var = df[f'{predictor}_between'].var()
                total_var = df[predictor].var()
                within_pct = 100 * within_var / total_var if total_var > 0 else 0
                between_pct = 100 * between_var / total_var if total_var > 0 else 0
                
                self.logger.info(f"Decomposition stats for {predictor}:")
                self.logger.info(f"  Within-person variance: {within_var:.4f} ({within_pct:.1f}%)")
                self.logger.info(f"  Between-person variance: {between_var:.4f} ({between_pct:.1f}%)")
            
            # Add sufficient participant observations check
            obs_per_participant = df.groupby('participant_id').size()
            participants_with_enough_obs = (obs_per_participant >= 3).sum()
            self.logger.info(f"Participants with 3+ observations: {participants_with_enough_obs} out of {len(obs_per_participant)}")
            
            # Basic demographics
            self.logger.info(f"Dataset statistics:")
            self.logger.info(f"  Total observations: {len(df)}")
            self.logger.info(f"  Total participants: {df['participant_id'].nunique()}")
            self.logger.info(f"  Observations per participant: min={obs_per_participant.min()}, "
                           f"max={obs_per_participant.max()}, mean={obs_per_participant.mean():.1f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

    def run_basic_multilevel_model(self, df, outcome_var, predictor):
        """Run a simple multilevel model with random intercepts.
        
        Args:
            df (DataFrame): Dataset with within/between decomposition
            outcome_var (str): Outcome variable name
            predictor (str): Predictor base name (without _within/_between suffixes)
            
        Returns:
            dict: Model results
        """
        self.logger.info(f"Running basic multilevel model for {outcome_var} ~ {predictor}")
        
        # Define predictor components
        within_pred = f"{predictor}_within"
        between_pred = f"{predictor}_between"
        
        # Build dataset for analysis (handle missing values)
        model_vars = [outcome_var, within_pred, between_pred, 'participant_id']
        model_data = df[model_vars].dropna()
        
        if len(model_data) < 20:
            self.logger.warning(f"Not enough valid data for {outcome_var} ~ {predictor} (n={len(model_data)})")
            return None
            
        try:
            # Build and fit the model - try a simpler specification first
            model_formula = f"{outcome_var} ~ {within_pred} + {between_pred}"
            
            self.logger.info(f"Fitting random intercepts model: {model_formula}")
            model = smf.mixedlm(
                formula=model_formula,
                data=model_data,
                groups=model_data["participant_id"],
                re_formula="~1"  # Random intercepts only
            )
            
            # Try to fit with restricted maximum likelihood
            try:
                result = model.fit(reml=True)
                method = "REML"
            except:
                self.logger.warning(f"REML fitting failed, trying ML estimation")
                result = model.fit(reml=False) 
                method = "ML"
                
            # Extract model results
            self.logger.info(f"Model successfully fitted using {method}")
            
            # Extract key parameters
            params = result.params
            pvalues = result.pvalues
            
            # Get within-person effect
            within_coef = params.get(within_pred, np.nan)
            within_pval = pvalues.get(within_pred, np.nan)
            within_sig = '***' if within_pval < 0.001 else '**' if within_pval < 0.01 else '*' if within_pval < 0.05 else ''
            
            # Get between-person effect
            between_coef = params.get(between_pred, np.nan)
            between_pval = pvalues.get(between_pred, np.nan)
            between_sig = '***' if between_pval < 0.001 else '**' if between_pval < 0.01 else '*' if between_pval < 0.05 else ''
            
            # Safely extract random effects variance components
            # Handle different possible return types from statsmodels
            random_effects = result.cov_re
            if isinstance(random_effects, np.ndarray):
                var_intercept = float(random_effects[0])  # For 1D array or first element
            elif isinstance(random_effects, pd.DataFrame):
                var_intercept = float(random_effects.iloc[0, 0])  # Get first element if it's a DataFrame
            elif hasattr(random_effects, 'iloc'):
                var_intercept = float(random_effects.iloc[0])  # For Series
            elif hasattr(random_effects, 'item'):  
                var_intercept = float(random_effects.item())  # For scalar
            else:
                # If none of the above, try direct conversion or use a fallback value
                try:
                    var_intercept = float(random_effects)
                except:
                    var_intercept = np.nan
                    self.logger.warning(f"Could not extract random intercept variance, setting to NaN")
            
            var_residual = float(result.scale)
            
            # Calculate ICC
            if var_intercept > 0 and var_residual > 0:
                icc = var_intercept / (var_intercept + var_residual)
            else:
                icc = np.nan
            
            # Create results dictionary
            model_results = {
                'outcome': outcome_var,
                'predictor': predictor,
                'model_type': 'Random Intercepts',
                'n_obs': len(model_data),
                'n_participants': model_data['participant_id'].nunique(),
                'within_coef': within_coef,
                'within_pval': within_pval,
                'within_sig': within_sig,
                'between_coef': between_coef,
                'between_pval': between_pval,
                'between_sig': between_sig,
                'var_intercept': var_intercept,
                'var_residual': var_residual,
                'icc': icc,
                'aic': result.aic,
                'bic': result.bic,
                'log_likelihood': result.llf,
                'method': method
            }
            
            # Log key results
            self.logger.info(f"Model results for {outcome_var} ~ {predictor}:")
            self.logger.info(f"  Within-person effect: {within_coef:.4f}, p={within_pval:.4f} {within_sig}")
            self.logger.info(f"  Between-person effect: {between_coef:.4f}, p={between_pval:.4f} {between_sig}")
            self.logger.info(f"  ICC: {icc:.4f}, AIC: {result.aic:.1f}")
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error in multilevel model {outcome_var} ~ {predictor}: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None
    
    def run_moderation_analysis(self, df, outcome_var, predictor, moderator_var):
        """Run a separate-groups analysis to test moderation effects.
        
        Args:
            df (DataFrame): Dataset with within/between decomposition
            outcome_var (str): Outcome variable name
            predictor (str): Predictor base name (without _within/_between suffixes)
            moderator_var (str): Moderating variable for subgroup analysis
            
        Returns:
            dict: Subgroup model results
        """
        self.logger.info(f"Running moderation analysis: {outcome_var} ~ {predictor} by {moderator_var}")
        
        # Check if moderator variable exists
        if moderator_var not in df.columns:
            self.logger.warning(f"Moderator variable {moderator_var} not found in data")
            return None
            
        # Get unique values of the moderator
        moderator_values = df[moderator_var].dropna().unique()
        
        if len(moderator_values) < 2:
            self.logger.warning(f"Moderator {moderator_var} has less than 2 categories, skipping moderation")
            return None
            
        # Define predictor components
        within_pred = f"{predictor}_within"
        between_pred = f"{predictor}_between"
        
        # Results container for each subgroup
        subgroup_results = {}
        
        # Run separate models for each subgroup
        for value in moderator_values:
            # Filter data for this subgroup
            subgroup_data = df[df[moderator_var] == value]
            
            if len(subgroup_data) < 20 or subgroup_data['participant_id'].nunique() < 5:
                self.logger.warning(f"Subgroup {moderator_var}={value} has insufficient data")
                continue
                
            # Build dataset for analysis (handle missing values)
            model_vars = [outcome_var, within_pred, between_pred, 'participant_id']
            model_data = subgroup_data[model_vars].dropna()
            
            if len(model_data) < 20:
                self.logger.warning(f"Not enough valid data for {outcome_var} ~ {predictor} in subgroup {moderator_var}={value}")
                continue
                
            try:
                # Build and fit the model
                model_formula = f"{outcome_var} ~ {within_pred} + {between_pred}"
                
                self.logger.info(f"Fitting subgroup model: {model_formula} for {moderator_var}={value}")
                model = smf.mixedlm(
                    formula=model_formula,
                    data=model_data,
                    groups=model_data["participant_id"],
                    re_formula="~1"  # Random intercepts only
                )
                
                # Try to fit with restricted maximum likelihood
                try:
                    result = model.fit(reml=True)
                    method = "REML"
                except:
                    self.logger.warning(f"REML fitting failed for subgroup, trying ML estimation")
                    result = model.fit(reml=False) 
                    method = "ML"
                    
                # Extract model results
                self.logger.info(f"Subgroup model successfully fitted using {method}")
                
                # Extract key parameters
                params = result.params
                pvalues = result.pvalues
                
                # Get within-person effect
                within_coef = params.get(within_pred, np.nan)
                within_pval = pvalues.get(within_pred, np.nan)
                within_sig = '***' if within_pval < 0.001 else '**' if within_pval < 0.01 else '*' if within_pval < 0.05 else ''
                
                # Get between-person effect
                between_coef = params.get(between_pred, np.nan)
                between_pval = pvalues.get(between_pred, np.nan)
                between_sig = '***' if between_pval < 0.001 else '**' if between_pval < 0.01 else '*' if between_pval < 0.05 else ''
                
                # Create results dictionary for this subgroup
                group_results = {
                    'outcome': outcome_var,
                    'predictor': predictor,
                    'moderator': moderator_var,
                    'subgroup': value,
                    'model_type': 'Subgroup Model',
                    'n_obs': len(model_data),
                    'n_participants': model_data['participant_id'].nunique(),
                    'within_coef': within_coef,
                    'within_pval': within_pval,
                    'within_sig': within_sig,
                    'between_coef': between_coef,
                    'between_pval': between_pval,
                    'between_sig': between_sig,
                    'aic': result.aic,
                    'bic': result.bic,
                    'method': method
                }
                
                # Log key results
                self.logger.info(f"Subgroup results for {moderator_var}={value}:")
                self.logger.info(f"  Within-person effect: {within_coef:.4f}, p={within_pval:.4f} {within_sig}")
                self.logger.info(f"  Between-person effect: {between_coef:.4f}, p={between_pval:.4f} {between_sig}")
                
                # Store results
                subgroup_results[str(value)] = group_results
                
            except Exception as e:
                self.logger.error(f"Error in subgroup model {outcome_var} ~ {predictor} for {moderator_var}={value}: {str(e)}")
                continue
        
        # Compare coefficients across subgroups if we have multiple valid models
        if len(subgroup_results) >= 2:
            # Calculate z-tests for differences between subgroups
            subgroup_comparisons = []
            subgroup_values = list(subgroup_results.keys())
            
            for i in range(len(subgroup_values)):
                for j in range(i+1, len(subgroup_values)):
                    value1 = subgroup_values[i]
                    value2 = subgroup_values[j]
                    
                    result1 = subgroup_results[value1]
                    result2 = subgroup_results[value2]
                    
                    # Calculate z-score for difference in within-person effects
                    # Formula: z = (b1 - b2) / sqrt(SE1^2 + SE2^2)
                    # We approximate SE from p-values and coefficients
                    within_diff = result1['within_coef'] - result2['within_coef']
                    
                    # Only compare if we have valid coefficients and p-values
                    if (not np.isnan(result1['within_coef']) and not np.isnan(result2['within_coef']) and
                        not np.isnan(result1['within_pval']) and not np.isnan(result2['within_pval'])):
                        
                        # Approximate standard errors
                        # For a two-tailed t-test with large df, p = 2*Phi(-|t|), so |t| = -Phi^{-1}(p/2)
                        # Since t = b/SE, SE = b/t
                        from scipy import stats
                        
                        def approx_se(b, p):
                            # Return a conservative estimate of SE based on p-value
                            if p >= 1:
                                return abs(b)  # Maximum uncertainty
                            if p <= 0:
                                return 0.0001 * abs(b)  # Minimum uncertainty
                            
                            # Get t-value from p-value (two-tailed test with large df)
                            t = abs(stats.norm.ppf(p/2))
                            if t == 0:
                                return abs(b)
                            return abs(b/t)
                        
                        se1 = approx_se(result1['within_coef'], result1['within_pval'])
                        se2 = approx_se(result2['within_coef'], result2['within_pval'])
                        
                        # Calculate z-score
                        pooled_se = np.sqrt(se1**2 + se2**2)
                        if pooled_se > 0:
                            z_score = within_diff / pooled_se
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                            
                            sig_diff = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                            
                            comparison = {
                                'group1': value1,
                                'group2': value2,
                                'within_diff': within_diff,
                                'z_score': z_score,
                                'p_value': p_value,
                                'sig_diff': sig_diff
                            }
                            
                            subgroup_comparisons.append(comparison)
                            
                            self.logger.info(f"Comparison of {value1} vs {value2}:")
                            self.logger.info(f"  Within-effect difference: {within_diff:.4f}, z={z_score:.2f}, p={p_value:.4f} {sig_diff}")
            
            if subgroup_comparisons:
                subgroup_results['comparisons'] = subgroup_comparisons
        
        return subgroup_results if subgroup_results else None
    
    def run_multilevel_analysis(self, df):
        """Run full multilevel analysis for all outcomes and predictors.
        
        Args:
            df (DataFrame): Prepared dataset
            
        Returns:
            bool: Success status
        """
        if df is None or df.empty:
            self.logger.error("No data available for analysis")
            return False
        
        # Results containers
        basic_models = {}
        moderation_analyses = {}
        successful_models = 0
        
        # 1. Run basic multilevel models
        self.logger.info("Running basic multilevel models")
        for outcome_var in self.outcome_variables:
            outcome_models = {}
            for predictor in self.fragmentation_predictors:
                model_result = self.run_basic_multilevel_model(df, outcome_var, predictor)
                if model_result:
                    outcome_models[predictor] = model_result
                    successful_models += 1
            
            if outcome_models:
                basic_models[outcome_var] = outcome_models
        
        # 2. Run moderation analyses
        self.logger.info("Running moderation analyses")
        for outcome_var in self.outcome_variables:
            outcome_moderations = {}
            for predictor in self.fragmentation_predictors:
                predictor_moderations = {}
                for moderator_name, moderator_var in self.subset_variables.items():
                    moderation_result = self.run_moderation_analysis(df, outcome_var, predictor, moderator_var)
                    if moderation_result:
                        predictor_moderations[moderator_name] = moderation_result
                        successful_models += 1
                
                if predictor_moderations:
                    outcome_moderations[predictor] = predictor_moderations
            
            if outcome_moderations:
                moderation_analyses[outcome_var] = outcome_moderations
        
        # Store all results
        self.model_results['basic_models'] = basic_models
        self.model_results['moderation_analyses'] = moderation_analyses
        
        self.logger.info(f"Completed multilevel analysis with {successful_models} successful models")
        
        # Consider any successful subset model as a success, even if basic models failed
        return successful_models > 0
    
    def save_results(self):
        """Save multilevel model results to Excel file"""
        if not self.model_results or (not self.model_results.get('basic_models') and not self.model_results.get('moderation_analyses')):
            self.logger.warning("No results to save")
            return None
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Prepare summary dataframes
            basic_summary = []
            moderation_summary = []
            comparison_summary = []
            
            # Process basic models
            for outcome_var, outcome_models in self.model_results.get('basic_models', {}).items():
                for predictor, model_result in outcome_models.items():
                    basic_summary.append({
                        'Outcome': outcome_var,
                        'Predictor': predictor,
                        'Within-Effect': model_result.get('within_coef', np.nan),
                        'Within-P': model_result.get('within_pval', np.nan),
                        'Within-Sig': model_result.get('within_sig', ''),
                        'Between-Effect': model_result.get('between_coef', np.nan),
                        'Between-P': model_result.get('between_pval', np.nan),
                        'Between-Sig': model_result.get('between_sig', ''),
                        'N': model_result.get('n_obs', 0),
                        'Participants': model_result.get('n_participants', 0),
                        'ICC': model_result.get('icc', np.nan),
                        'AIC': model_result.get('aic', np.nan),
                        'BIC': model_result.get('bic', np.nan)
                    })
            
            # Process moderation analyses
            for outcome_var, outcome_moderations in self.model_results.get('moderation_analyses', {}).items():
                for predictor, predictor_moderations in outcome_moderations.items():
                    for moderator_name, moderation_result in predictor_moderations.items():
                        for subgroup, subgroup_result in moderation_result.items():
                            if subgroup != 'comparisons':
                                moderation_summary.append({
                                    'Outcome': outcome_var,
                                    'Predictor': predictor,
                                    'Moderator': moderator_name,
                                    'Subgroup': subgroup,
                                    'Within-Effect': subgroup_result.get('within_coef', np.nan),
                                    'Within-P': subgroup_result.get('within_pval', np.nan),
                                    'Within-Sig': subgroup_result.get('within_sig', ''),
                                    'Between-Effect': subgroup_result.get('between_coef', np.nan),
                                    'Between-P': subgroup_result.get('between_pval', np.nan),
                                    'Between-Sig': subgroup_result.get('between_sig', ''),
                                    'N': subgroup_result.get('n_obs', 0),
                                    'Participants': subgroup_result.get('n_participants', 0)
                                })
                        
                        # Process comparisons
                        if 'comparisons' in moderation_result:
                            for comparison in moderation_result['comparisons']:
                                comparison_summary.append({
                                    'Outcome': outcome_var,
                                    'Predictor': predictor,
                                    'Moderator': moderator_name,
                                    'Group1': comparison.get('group1', ''),
                                    'Group2': comparison.get('group2', ''),
                                    'Effect Difference': comparison.get('within_diff', np.nan),
                                    'Z-score': comparison.get('z_score', np.nan),
                                    'P-value': comparison.get('p_value', np.nan),
                                    'Significance': comparison.get('sig_diff', '')
                                })
            
            # Convert to dataframes
            basic_df = pd.DataFrame(basic_summary) if basic_summary else pd.DataFrame()
            moderation_df = pd.DataFrame(moderation_summary) if moderation_summary else pd.DataFrame()
            comparison_df = pd.DataFrame(comparison_summary) if comparison_summary else pd.DataFrame()
            
            # Round numeric columns
            for df in [basic_df, moderation_df, comparison_df]:
                if not df.empty:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].round(4)
            
            # Save to Excel
            output_path = self.output_dir / f'multilevel_analysis_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(output_path) as writer:
                # Save main summary sheets
                if not basic_df.empty:
                    basic_df.to_excel(writer, sheet_name='Basic Models', index=False)
                
                if not moderation_df.empty:
                    moderation_df.to_excel(writer, sheet_name='Subgroup Effects', index=False)
                
                if not comparison_df.empty:
                    comparison_df.to_excel(writer, sheet_name='Group Comparisons', index=False)
                
                # Create outcome-specific sheets
                for outcome_var in self.outcome_variables:
                    # Basic results for this outcome
                    outcome_basic = basic_df[basic_df['Outcome'] == outcome_var] if not basic_df.empty else pd.DataFrame()
                    if not outcome_basic.empty:
                        sheet_name = f"{outcome_var.replace('_std', '')}_Basic"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                        outcome_basic.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Moderation results for this outcome
                    outcome_mod = moderation_df[moderation_df['Outcome'] == outcome_var] if not moderation_df.empty else pd.DataFrame()
                    if not outcome_mod.empty:
                        sheet_name = f"{outcome_var.replace('_std', '')}_Subgroups"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                        outcome_mod.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Comparison results for this outcome
                    outcome_comp = comparison_df[comparison_df['Outcome'] == outcome_var] if not comparison_df.empty else pd.DataFrame()
                    if not outcome_comp.empty:
                        sheet_name = f"{outcome_var.replace('_std', '')}_Comparisons"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                        outcome_comp.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create moderator-specific comparison sheets
                moderators = comparison_df['Moderator'].unique() if not comparison_df.empty else []
                for moderator in moderators:
                    mod_comp = comparison_df[comparison_df['Moderator'] == moderator]
                    if not mod_comp.empty:
                        sheet_name = f"{moderator.title()}_Comparisons"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                        mod_comp.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(f"Saved multilevel model results to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run the simplified multilevel analysis."""
    try:
        # Create analyzer with debug mode
        analyzer = SimplifiedMultilevelAnalysis(debug=True)
        
        # Load and prepare data
        df = analyzer.load_data()
        
        if df is None or df.empty:
            print("Error: Failed to load data")
            return 1
        
        # Run multilevel models
        if analyzer.run_multilevel_analysis(df):
            # Save results
            results_path = analyzer.save_results()
            
            if results_path:
                print(f"Multilevel analysis of fragmentation effects completed successfully!")
                print(f"Results saved to: {results_path}")
                return 0
            else:
                print("Error: Failed to save results")
                return 1
        else:
            print("Error: Failed to run models")
            return 1
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    # Ignore certain warnings
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
    
    exit(main())