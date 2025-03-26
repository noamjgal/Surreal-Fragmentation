#!/usr/bin/env python3
"""
Improved Multilevel Analysis of Fragmentation Effects

This script performs multilevel regression analysis to examine the relationship 
between fragmentation metrics and emotional outcomes while properly accounting
for within-participant variance, adding control variables, and testing 
cross-fragmentation models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging
from datetime import datetime
import warnings

# Add function to calculate R-squared for multilevel models
def calculate_r_squared(model, y_true):
    """Calculate R² for multilevel models.
    
    Calculates two types of R²:
    1. Marginal R² - variance explained by fixed effects
    2. Conditional R² - variance explained by fixed and random effects together
    
    Args:
        model: Fitted statsmodels MixedLM model
        y_true: True outcome values
        
    Returns:
        dict: Dictionary with marginal and conditional R² values
    """
    try:
        # Get fixed effects predictions using the model's predict method
        # This is more reliable than trying to access exog directly
        y_pred = model.predict()
        fixed_effects_var = np.var(y_pred)
        
        # Get residual variance
        var_residual = model.scale
        
        # Get random effects variance (intercept variance)
        # Handle different types of cov_re objects
        try:
            if hasattr(model.cov_re, 'iloc'):
                var_random = float(model.cov_re.iloc[0, 0])
            elif hasattr(model.cov_re, 'item'):
                var_random = float(model.cov_re.item())
            elif isinstance(model.cov_re, np.ndarray):
                var_random = float(model.cov_re.flat[0])
            else:
                var_random = float(model.cov_re)
        except (AttributeError, IndexError, ValueError):
            var_random = 0.0
            
        # Ensure variances are valid positive numbers
        fixed_effects_var = max(0.0, fixed_effects_var)
        var_residual = max(0.0, var_residual)
        var_random = max(0.0, var_random)
        
        # Total variance
        var_total = fixed_effects_var + var_random + var_residual
        
        # Calculate R-squared values with safety checks
        if var_total > 0:
            r2_marginal = fixed_effects_var / var_total
            r2_conditional = (fixed_effects_var + var_random) / var_total
        else:
            # If total variance is zero, fall back to traditional R² calculation
            residuals = y_true - y_pred
            total_ss = np.sum((y_true - np.mean(y_true))**2)
            residual_ss = np.sum(residuals**2)
            
            if total_ss > 0:
                r2_marginal = 1 - (residual_ss / total_ss)
                r2_conditional = r2_marginal  # In this fallback, we can't distinguish
            else:
                r2_marginal = 0.0
                r2_conditional = 0.0
        
        # Ensure R² values are in valid range [0,1]
        r2_marginal = max(0.0, min(1.0, r2_marginal))
        r2_conditional = max(0.0, min(1.0, r2_conditional))
        
        return {
            'r2_marginal': r2_marginal,
            'r2_conditional': r2_conditional
        }
    except Exception as e:
        logging.warning(f"Error calculating R-squared: {str(e)}")
        return {
            'r2_marginal': 0.0,
            'r2_conditional': 0.0
        }

class ImprovedMultilevelAnalysis:
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
            'overlap_fragmentation',
            'digital_home_fragmentation',  # NEW: Digital fragmentation during home periods
            'digital_home_mobility_delta'  # NEW: Delta between home and mobility fragmentation
        ]
        
        # Define outcome variables
        self.outcome_variables = [
            'anxiety_score_std',   # Standardized anxiety score
            'mood_score_std'       # Standardized mood/depression score
        ]
        
        # Define control variables to include in models
        self.control_variables = {
            'age': 'age_group',
            'gender': 'gender_standardized',
            'location': 'location_type',
            'digital_duration': 'digital_total_duration',
            'mobility_duration': 'mobility_total_duration',
            'overlap_duration': 'overlap_total_duration',
            'digital_home_duration': 'digital_home_total_duration',
            'active_transport_duration': 'active_transport_duration',
            'mechanized_transport_duration': 'mechanized_transport_duration',
            'home_duration': 'home_duration',
            'out_of_home_duration': 'out_of_home_duration'
        }
        
        # Define subset variables for interaction testing (removed dataset since it's identical to age)
        self.subset_variables = {
            'age': 'age_group',
            'gender': 'gender_standardized',
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
        self.logger.info(f"Initializing improved multilevel analysis of fragmentation effects")
        self.logger.info(f"Pooled data path: {self.pooled_data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load pooled data and prepare for multilevel modeling with corrected participant IDs.
        
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
            
            # Check for dataset source column (needed for creating unique participant IDs)
            dataset_source_col = 'dataset_source'
            if dataset_source_col not in df.columns:
                self.logger.error(f"Missing dataset source column: {dataset_source_col}")
                self.logger.error("Available columns: " + ", ".join(df.columns.tolist()))
                return None
            
            # Create unique participant IDs by combining dataset source and participant ID
            self.logger.info("Creating unique participant IDs across datasets")
            df['original_participant_id'] = df['participant_id']  # Keep original ID for reference
            
            # Check that we have both necessary columns
            if 'participant_id' in df.columns and dataset_source_col in df.columns:
                # Create unique participant ID
                df['participant_id'] = df[dataset_source_col] + '_' + df['participant_id'].astype(str)
                
                # Log the number of unique participants before and after correction
                original_ids_count = df['original_participant_id'].nunique()
                new_ids_count = df['participant_id'].nunique()
                self.logger.info(f"Original number of participant IDs: {original_ids_count}")
                self.logger.info(f"Corrected number of unique participant IDs: {new_ids_count}")
                
                # Check if this fixed the issue
                if new_ids_count > original_ids_count:
                    self.logger.info(f"Successfully disambiguated {new_ids_count - original_ids_count} duplicate participant IDs")
                elif new_ids_count == original_ids_count:
                    self.logger.info("No duplicate participant IDs were found")
                else:
                    self.logger.warning("Unexpected reduction in unique participant IDs after correction")
            else:
                self.logger.error("Could not create unique participant IDs due to missing columns")
                return None
                
            # Check for required columns
            required_vars = (self.fragmentation_predictors + 
                            self.outcome_variables + 
                            list(self.control_variables.values()) + 
                            ['participant_id'])
            
            missing_vars = [var for var in required_vars if var not in df.columns]
            if missing_vars:
                self.logger.error(f"Missing critical variables: {missing_vars}")
                self.logger.error("Available columns: " + ", ".join(df.columns.tolist()))
                return None
            
            # Create decomposition of predictors (within and between components)
            self.logger.info("Decomposing fragmentation metrics into within and between-person components")
            for predictor in self.fragmentation_predictors:
                if predictor not in df.columns:
                    self.logger.warning(f"Predictor {predictor} not found in data, will be skipped in analysis")
                    continue
                    
                # Calculate person mean (between-person component) using the new unique IDs
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
            
            # Report participant counts by dataset
            if dataset_source_col in df.columns:
                dataset_counts = df.groupby(dataset_source_col)['participant_id'].nunique()
                self.logger.info(f"Participant count by dataset:")
                for dataset, count in dataset_counts.items():
                    self.logger.info(f"  {dataset}: {count} participants")
            
            # Report categorical distributions for control variables
            for control_name, control_var in self.control_variables.items():
                if control_var in df.columns:
                    value_counts = df[control_var].value_counts()
                    self.logger.info(f"Distribution of {control_name} ({control_var}):")
                    for val, count in value_counts.items():
                        self.logger.info(f"  {val}: {count} ({100 * count / len(df):.1f}%)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

    def run_basic_multilevel_model(self, df, outcome_var, predictor, with_controls=False):
        """Run a simple multilevel model with random intercepts.
        
        Args:
            df (DataFrame): Dataset with within/between decomposition
            outcome_var (str): Outcome variable name
            predictor (str): Predictor base name (without _within/_between suffixes)
            with_controls (bool): Whether to include control variables
            
        Returns:
            dict: Model results
        """
        model_type = "With Controls" if with_controls else "Basic"
        self.logger.info(f"Running {model_type} multilevel model for {outcome_var} ~ {predictor}")
        
        # Define predictor components
        within_pred = f"{predictor}_within"
        between_pred = f"{predictor}_between"
        
        # Build dataset for analysis (handle missing values)
        model_vars = [outcome_var, within_pred, between_pred, 'participant_id']
        
        # Add control variables if requested
        control_terms = []
        if with_controls:
            for control_name, control_var in self.control_variables.items():
                if control_var in df.columns:  # Only add if the control variable exists
                    model_vars.append(control_var)
                    # Add control variable to formula
                    control_terms.append(control_var)
                    self.logger.info(f"Adding control variable: {control_var}")
                else:
                    self.logger.warning(f"Control variable {control_var} not found in data, skipping")
        
        model_data = df[model_vars].dropna()
        
        if len(model_data) < 20:
            self.logger.warning(f"Not enough valid data for {outcome_var} ~ {predictor} (n={len(model_data)})")
            return None
            
        try:
            # Build model formula with or without controls
            if with_controls:
                control_formula = " + " + " + ".join(control_terms)
                model_formula = f"{outcome_var} ~ {within_pred} + {between_pred}{control_formula}"
            else:
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
            conf_int = result.conf_int()
            
            # Get within-person effect
            within_coef = params.get(within_pred, np.nan)
            within_pval = pvalues.get(within_pred, np.nan)
            within_sig = '***' if within_pval < 0.001 else '**' if within_pval < 0.01 else '*' if within_pval < 0.05 else ''
            
            # Get within-person confidence interval
            if within_pred in conf_int.index:
                within_ci_low = conf_int.loc[within_pred, 0]
                within_ci_high = conf_int.loc[within_pred, 1]
            else:
                within_ci_low = within_ci_high = np.nan
            
            # Get between-person effect
            between_coef = params.get(between_pred, np.nan)
            between_pval = pvalues.get(between_pred, np.nan)
            between_sig = '***' if between_pval < 0.001 else '**' if between_pval < 0.01 else '*' if between_pval < 0.05 else ''
            
            # Get between-person confidence interval
            if between_pred in conf_int.index:
                between_ci_low = conf_int.loc[between_pred, 0]
                between_ci_high = conf_int.loc[between_pred, 1]
            else:
                between_ci_low = between_ci_high = np.nan
            
            # Extract control variable effects (if included)
            control_effects = {}
            if with_controls:
                for control_name, control_var in self.control_variables.items():
                    # Some controls might be categorical with multiple parameters
                    control_params = {k: v for k, v in params.items() if control_var in k}
                    control_pvals = {k: v for k, v in pvalues.items() if control_var in k}
                    control_sig = {k: '***' if v < 0.001 else '**' if v < 0.01 else '*' if v < 0.05 else '' 
                                  for k, v in control_pvals.items()}
                    
                    # Get confidence intervals for control variables
                    control_ci_low = {k: conf_int.loc[k, 0] if k in conf_int.index else np.nan 
                                     for k in control_params.keys()}
                    control_ci_high = {k: conf_int.loc[k, 1] if k in conf_int.index else np.nan 
                                      for k in control_params.keys()}
                    
                    if control_params:
                        control_effects[control_name] = {
                            'params': control_params,
                            'pvals': control_pvals,
                            'sig': control_sig,
                            'ci_low': control_ci_low,
                            'ci_high': control_ci_high
                        }
            
            # Safely extract random effects variance components
            try:
                random_effects = result.cov_re
                if hasattr(random_effects, 'iloc'):
                    var_intercept = float(random_effects.iloc[0, 0])
                elif hasattr(random_effects, 'item'):
                    var_intercept = float(random_effects.item())
                elif isinstance(random_effects, np.ndarray):
                    var_intercept = float(random_effects.flat[0])
                else:
                    var_intercept = float(random_effects)
            except (AttributeError, IndexError, ValueError):
                var_intercept = 0.0
                self.logger.warning(f"Could not extract random intercept variance, setting to 0")
            
            var_residual = float(result.scale)
            
            # Calculate ICC
            if var_intercept > 0 and var_residual > 0:
                icc = var_intercept / (var_intercept + var_residual)
            else:
                icc = 0.0
                
            # Calculate R-squared values
            r_squared = calculate_r_squared(result, model_data[outcome_var])
            r2_marginal = r_squared['r2_marginal']
            r2_conditional = r_squared['r2_conditional']
            
            # Calculate AIC and BIC manually if necessary
            aic = result.aic
            bic = result.bic
            
            # If AIC/BIC are NaN, calculate them manually
            if np.isnan(aic) or np.isnan(bic):
                n = len(model_data)  # Sample size
                k = len(params)  # Number of parameters
                llf = result.llf  # Log-likelihood
                
                if not np.isnan(llf):
                    # Calculate AIC and BIC
                    aic = 2 * k - 2 * llf
                    bic = k * np.log(n) - 2 * llf
                    self.logger.info(f"Calculated AIC/BIC manually: AIC={aic:.2f}, BIC={bic:.2f}")
                else:
                    # If log-likelihood is NaN, try to calculate from deviance
                    try:
                        deviance = result.deviance
                        if not np.isnan(deviance):
                            # For normal distribution, deviance = -2 * log-likelihood
                            llf = -deviance / 2
                            aic = 2 * k - 2 * llf
                            bic = k * np.log(n) - 2 * llf
                            self.logger.info(f"Calculated AIC/BIC from deviance: AIC={aic:.2f}, BIC={bic:.2f}")
                    except:
                        self.logger.warning("Could not calculate AIC/BIC - both log-likelihood and deviance are NaN")
            
            # Create results dictionary
            model_results = {
                'outcome': outcome_var,
                'predictor': predictor,
                'model_type': 'With Controls' if with_controls else 'Basic',
                'n_obs': len(model_data),
                'n_participants': model_data['participant_id'].nunique(),
                'within_coef': within_coef,
                'within_pval': within_pval,
                'within_sig': within_sig,
                'within_ci_low': within_ci_low,
                'within_ci_high': within_ci_high,
                'between_coef': between_coef,
                'between_pval': between_pval,
                'between_sig': between_sig,
                'between_ci_low': between_ci_low,
                'between_ci_high': between_ci_high,
                'var_intercept': var_intercept,
                'var_residual': var_residual,
                'icc': icc,
                'r2_marginal': r2_marginal,  # Add R² values
                'r2_conditional': r2_conditional,
                'aic': aic,
                'bic': bic,
                'log_likelihood': result.llf,
                'method': method,
                'formula': model_formula  # Add formula to results for reference
            }
            
            # Add control effects if included
            if with_controls:
                model_results['control_effects'] = control_effects
            
            # Log key results
            self.logger.info(f"Model results for {outcome_var} ~ {predictor} ({model_type}):")
            self.logger.info(f"  Within-person effect: {within_coef:.4f}, p={within_pval:.4f} {within_sig}")
            self.logger.info(f"  Between-person effect: {between_coef:.4f}, p={between_pval:.4f} {between_sig}")
            self.logger.info(f"  ICC: {icc:.4f}, R² marginal: {r2_marginal:.4f}, R² conditional: {r2_conditional:.4f}")
            self.logger.info(f"  AIC: {aic:.1f}")
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error in multilevel model {outcome_var} ~ {predictor}: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None
    
    def run_cross_fragmentation_model(self, df, outcome_var, primary_predictor, control_predictor, with_full_controls=True):
        """Run a multilevel model with one fragmentation controlling for another.
        
        Args:
            df (DataFrame): Dataset with within/between decomposition
            outcome_var (str): Outcome variable name
            primary_predictor (str): Main predictor variable
            control_predictor (str): Fragmentation variable to control for
            with_full_controls (bool): Whether to include demographic control variables
            
        Returns:
            dict: Model results
        """
        model_type = "Cross-Fragmentation" + (" with Full Controls" if with_full_controls else "")
        self.logger.info(f"Running {model_type} model: {outcome_var} ~ {primary_predictor} controlling for {control_predictor}")
        
        # Define predictor components for both primary and control
        primary_within = f"{primary_predictor}_within"
        primary_between = f"{primary_predictor}_between"
        control_within = f"{control_predictor}_within"
        control_between = f"{control_predictor}_between"
        
        # Build dataset for analysis (handle missing values)
        model_vars = [outcome_var, primary_within, primary_between, 
                      control_within, control_between, 'participant_id']
        
        # Add demographic control variables too if requested
        control_terms = [control_within, control_between]
        if with_full_controls:
            for control_var in self.control_variables.values():
                model_vars.append(control_var)
                control_terms.append(control_var)
        
        model_data = df[model_vars].dropna()
        
        if len(model_data) < 20:
            self.logger.warning(f"Not enough valid data for cross-fragmentation model (n={len(model_data)})")
            return None
            
        try:
            # Build formula with control fragmentation and demographic controls if requested
            control_formula = " + " + " + ".join(control_terms)
            model_formula = f"{outcome_var} ~ {primary_within} + {primary_between}{control_formula}"
            
            self.logger.info(f"Fitting cross-fragmentation model: {model_formula}")
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
            conf_int = result.conf_int()
            
            # Get within-person effect for primary predictor
            within_coef = params.get(primary_within, np.nan)
            within_pval = pvalues.get(primary_within, np.nan)
            within_sig = '***' if within_pval < 0.001 else '**' if within_pval < 0.01 else '*' if within_pval < 0.05 else ''
            
            # Get within-person confidence interval
            if primary_within in conf_int.index:
                within_ci_low = conf_int.loc[primary_within, 0]
                within_ci_high = conf_int.loc[primary_within, 1]
            else:
                within_ci_low = within_ci_high = np.nan
            
            # Get between-person effect for primary predictor
            between_coef = params.get(primary_between, np.nan)
            between_pval = pvalues.get(primary_between, np.nan)
            between_sig = '***' if between_pval < 0.001 else '**' if between_pval < 0.01 else '*' if between_pval < 0.05 else ''
            
            # Get between-person confidence interval
            if primary_between in conf_int.index:
                between_ci_low = conf_int.loc[primary_between, 0]
                between_ci_high = conf_int.loc[primary_between, 1]
            else:
                between_ci_low = between_ci_high = np.nan
                
            # Get control predictor effects
            control_within_coef = params.get(control_within, np.nan)
            control_within_pval = pvalues.get(control_within, np.nan)
            control_within_sig = '***' if control_within_pval < 0.001 else '**' if control_within_pval < 0.01 else '*' if control_within_pval < 0.05 else ''
            
            control_between_coef = params.get(control_between, np.nan)
            control_between_pval = pvalues.get(control_between, np.nan)
            control_between_sig = '***' if control_between_pval < 0.001 else '**' if control_between_pval < 0.01 else '*' if control_between_pval < 0.05 else ''
            
            # Safely extract random effects variance components
            try:
                random_effects = result.cov_re
                if hasattr(random_effects, 'iloc'):
                    var_intercept = float(random_effects.iloc[0, 0])
                elif hasattr(random_effects, 'item'):
                    var_intercept = float(random_effects.item())
                elif isinstance(random_effects, np.ndarray):
                    var_intercept = float(random_effects.flat[0])
                else:
                    var_intercept = float(random_effects)
            except (AttributeError, IndexError, ValueError):
                var_intercept = 0.0
                self.logger.warning(f"Could not extract random intercept variance, setting to 0")
            
            var_residual = float(result.scale)
            
            # Calculate ICC
            if var_intercept > 0 and var_residual > 0:
                icc = var_intercept / (var_intercept + var_residual)
            else:
                icc = 0.0
                
            # Calculate R-squared values
            r_squared = calculate_r_squared(result, model_data[outcome_var])
            r2_marginal = r_squared['r2_marginal']
            r2_conditional = r_squared['r2_conditional']
            
            # Calculate AIC and BIC manually if necessary
            aic = result.aic
            bic = result.bic
            
            # If AIC/BIC are NaN, calculate them manually
            if np.isnan(aic) or np.isnan(bic):
                n = len(model_data)  # Sample size
                k = len(params)  # Number of parameters
                llf = result.llf  # Log-likelihood
                
                if not np.isnan(llf):
                    # Calculate AIC and BIC
                    aic = 2 * k - 2 * llf
                    bic = k * np.log(n) - 2 * llf
                    self.logger.info(f"Calculated AIC/BIC manually: AIC={aic:.2f}, BIC={bic:.2f}")
                else:
                    # If log-likelihood is NaN, try to calculate from deviance
                    try:
                        deviance = result.deviance
                        if not np.isnan(deviance):
                            # For normal distribution, deviance = -2 * log-likelihood
                            llf = -deviance / 2
                            aic = 2 * k - 2 * llf
                            bic = k * np.log(n) - 2 * llf
                            self.logger.info(f"Calculated AIC/BIC from deviance: AIC={aic:.2f}, BIC={bic:.2f}")
                    except:
                        self.logger.warning("Could not calculate AIC/BIC - both log-likelihood and deviance are NaN")
                        aic = bic = np.nan
            
            # Create results dictionary
            model_results = {
                'outcome': outcome_var,
                'predictor': primary_predictor,
                'control_predictor': control_predictor,
                'model_type': 'Cross-Fragmentation' + (" With Full Controls" if with_full_controls else ""),
                'n_obs': len(model_data),
                'n_participants': model_data['participant_id'].nunique(),
                'within_coef': within_coef,
                'within_pval': within_pval,
                'within_sig': within_sig,
                'within_ci_low': within_ci_low,
                'within_ci_high': within_ci_high,
                'between_coef': between_coef,
                'between_pval': between_pval,
                'between_sig': between_sig,
                'between_ci_low': between_ci_low,
                'between_ci_high': between_ci_high,
                'control_within_coef': control_within_coef,
                'control_within_pval': control_within_pval,
                'control_within_sig': control_within_sig,
                'control_between_coef': control_between_coef,
                'control_between_pval': control_between_pval,
                'control_between_sig': control_between_sig,
                'var_intercept': var_intercept,
                'var_residual': var_residual,
                'icc': icc,
                'r2_marginal': r2_marginal,  # Properly calculated R² values
                'r2_conditional': r2_conditional,
                'aic': aic,  # Properly calculated AIC
                'bic': bic,  # Properly calculated BIC
                'log_likelihood': result.llf,
                'method': method,
                'formula': model_formula  # Add formula to results for reference
            }
            
            # Log key results
            model_type_str = "Cross-fragmentation" + (" with controls" if with_full_controls else "")
            self.logger.info(f"{model_type_str} model results for {outcome_var} ~ {primary_predictor} | {control_predictor}:")
            self.logger.info(f"  Primary within-effect: {within_coef:.4f}, p={within_pval:.4f} {within_sig}")
            self.logger.info(f"  Primary between-effect: {between_coef:.4f}, p={between_pval:.4f} {between_sig}")
            self.logger.info(f"  Control within-effect: {control_within_coef:.4f}, p={control_within_pval:.4f} {control_within_sig}")
            self.logger.info(f"  Control between-effect: {control_between_coef:.4f}, p={control_between_pval:.4f} {control_between_sig}")
            self.logger.info(f"  ICC: {icc:.4f}, R² marginal: {r2_marginal:.4f}, R² conditional: {r2_conditional:.4f}")
            self.logger.info(f"  AIC: {aic:.1f}, BIC: {bic:.1f}")
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-fragmentation model: {str(e)}")
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
        skipped_subgroups = []
        
        # Run separate models for each subgroup
        for value in moderator_values:
            # Filter data for this subgroup
            subgroup_data = df[df[moderator_var] == value]
            
            # Lower threshold to 10 observations and 3 participants (previously 20 and 5)
            if len(subgroup_data) < 10 or subgroup_data['participant_id'].nunique() < 3:
                self.logger.warning(f"Subgroup {moderator_var}={value} has insufficient data (n={len(subgroup_data)}, participants={subgroup_data['participant_id'].nunique()})")
                skipped_subgroups.append(str(value))
                continue
                
            # Build dataset for analysis (handle missing values)
            model_vars = [outcome_var, within_pred, between_pred, 'participant_id']
            model_data = subgroup_data[model_vars].dropna()
            
            if len(model_data) < 10:  # Also lower this threshold
                self.logger.warning(f"Not enough valid data for {outcome_var} ~ {predictor} in subgroup {moderator_var}={value} (n={len(model_data)})")
                skipped_subgroups.append(str(value))
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
                conf_int = result.conf_int()
                
                # Get within-person effect
                within_coef = params.get(within_pred, np.nan)
                within_pval = pvalues.get(within_pred, np.nan)
                within_sig = '***' if within_pval < 0.001 else '**' if within_pval < 0.01 else '*' if within_pval < 0.05 else ''
                
                # Get within-person confidence interval
                if within_pred in conf_int.index:
                    within_ci_low = conf_int.loc[within_pred, 0]
                    within_ci_high = conf_int.loc[within_pred, 1]
                else:
                    within_ci_low = within_ci_high = np.nan
                
                # Get between-person effect
                between_coef = params.get(between_pred, np.nan)
                between_pval = pvalues.get(between_pred, np.nan)
                between_sig = '***' if between_pval < 0.001 else '**' if between_pval < 0.01 else '*' if between_pval < 0.05 else ''
                
                # Get between-person confidence interval
                if between_pred in conf_int.index:
                    between_ci_low = conf_int.loc[between_pred, 0]
                    between_ci_high = conf_int.loc[between_pred, 1]
                else:
                    between_ci_low = between_ci_high = np.nan
                
                # Calculate R-squared values
                r_squared = calculate_r_squared(result, model_data[outcome_var])
                r2_marginal = r_squared['r2_marginal']
                r2_conditional = r_squared['r2_conditional']
                
                # Calculate AIC and BIC manually if necessary
                aic = result.aic
                bic = result.bic
                
                # If AIC/BIC are NaN, calculate them manually
                if np.isnan(aic) or np.isnan(bic):
                    n = len(model_data)  # Sample size
                    k = len(params)  # Number of parameters
                    llf = result.llf  # Log-likelihood
                    
                    if not np.isnan(llf):
                        # Calculate AIC and BIC
                        aic = 2 * k - 2 * llf
                        bic = k * np.log(n) - 2 * llf
                        self.logger.info(f"Calculated AIC/BIC manually: AIC={aic:.2f}, BIC={bic:.2f}")
                    else:
                        # If log-likelihood is NaN, try to calculate from deviance
                        try:
                            deviance = result.deviance
                            if not np.isnan(deviance):
                                # For normal distribution, deviance = -2 * log-likelihood
                                llf = -deviance / 2
                                aic = 2 * k - 2 * llf
                                bic = k * np.log(n) - 2 * llf
                                self.logger.info(f"Calculated AIC/BIC from deviance: AIC={aic:.2f}, BIC={bic:.2f}")
                        except:
                            self.logger.warning("Could not calculate AIC/BIC - both log-likelihood and deviance are NaN")
                
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
                    'within_ci_low': within_ci_low,
                    'within_ci_high': within_ci_high,
                    'between_coef': between_coef,
                    'between_pval': between_pval,
                    'between_sig': between_sig,
                    'between_ci_low': between_ci_low,
                    'between_ci_high': between_ci_high,
                    'r2_marginal': r2_marginal,  # Add R² values
                    'r2_conditional': r2_conditional,
                    'aic': aic,
                    'bic': bic,
                    'method': method,
                    'formula': model_formula
                }
                
                # Log key results
                self.logger.info(f"Subgroup results for {moderator_var}={value}:")
                self.logger.info(f"  Within-person effect: {within_coef:.4f}, p={within_pval:.4f} {within_sig}")
                self.logger.info(f"  R² marginal: {r2_marginal:.4f}, R² conditional: {r2_conditional:.4f}")
                
                # Store results
                subgroup_results[str(value)] = group_results
                
            except Exception as e:
                self.logger.error(f"Error in subgroup model {outcome_var} ~ {predictor} for {moderator_var}={value}: {str(e)}")
                skipped_subgroups.append(str(value))
                continue
        
        # Log summary of skipped subgroups
        if skipped_subgroups:
            self.logger.warning(f"Skipped subgroups for {outcome_var} ~ {predictor} by {moderator_var}: {', '.join(skipped_subgroups)}")
        
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
                        z_score = (result1['within_coef'] - result2['within_coef']) / pooled_se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        
                        sig_diff = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                        
                        comparison = {
                            'group1': value1,
                            'group2': value2,
                            'within_diff': result1['within_coef'] - result2['within_coef'],
                            'z_score': z_score,
                            'p_value': p_value,
                            'sig_diff': sig_diff
                        }
                        
                        subgroup_comparisons.append(comparison)
                        
                        self.logger.info(f"Comparison of {value1} vs {value2}:")
                        self.logger.info(f"  Within-effect difference: {result1['within_coef'] - result2['within_coef']:.4f}, z={z_score:.2f}, p={p_value:.4f} {sig_diff}")
            
            if subgroup_comparisons:
                subgroup_results['comparisons'] = subgroup_comparisons
        elif len(subgroup_results) == 1:
            self.logger.warning(f"Only one subgroup ({list(subgroup_results.keys())[0]}) had sufficient data for {outcome_var} ~ {predictor} by {moderator_var}")
        elif len(subgroup_results) == 0:
            self.logger.warning(f"No subgroups had sufficient data for {outcome_var} ~ {predictor} by {moderator_var}")
        
        return subgroup_results if subgroup_results else None
    
    def run_home_mobility_comparison(self, df, outcome_var):
        """Run a dedicated analysis comparing digital fragmentation at home vs. during mobility.
        
        Args:
            df (DataFrame): Dataset with within/between decomposition
            outcome_var (str): Outcome variable name
            
        Returns:
            dict: Model results
        """
        self.logger.info(f"Running home vs. mobility fragmentation comparison for {outcome_var}")
        
        # We need all three metrics for this analysis
        digital_home = 'digital_home_fragmentation'
        overlap = 'overlap_fragmentation'
        
        model_vars = [outcome_var, f'{digital_home}_within', f'{digital_home}_between',
                      f'{overlap}_within', f'{overlap}_between', 'participant_id']
        
        # Add control variables
        for control_var in self.control_variables.values():
            model_vars.append(control_var)
        
        model_data = df[model_vars].dropna()
        
        if len(model_data) < 20:
            self.logger.warning(f"Not enough valid data for home vs. mobility comparison (n={len(model_data)})")
            return None
        
        try:
            # Build model formula with location controls
            control_terms = [control_var for control_var in self.control_variables.values()]
            control_formula = " + " + " + ".join(control_terms) if control_terms else ""
            
            # The model includes both fragmentation types
            model_formula = (f"{outcome_var} ~ {digital_home}_within + {digital_home}_between + "
                            f"{overlap}_within + {overlap}_between{control_formula}")
            
            self.logger.info(f"Fitting home vs. mobility comparison model: {model_formula}")
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
            
            # Extract key parameters
            params = result.params
            pvalues = result.pvalues
            conf_int = result.conf_int()
            
            # Safely extract random effects variance components for ICC calculation
            try:
                random_effects = result.cov_re
                if hasattr(random_effects, 'iloc'):
                    var_intercept = float(random_effects.iloc[0, 0])
                elif hasattr(random_effects, 'item'):
                    var_intercept = float(random_effects.item())
                elif isinstance(random_effects, np.ndarray):
                    var_intercept = float(random_effects.flat[0])
                else:
                    var_intercept = float(random_effects)
            except (AttributeError, IndexError, ValueError):
                var_intercept = 0.0
                self.logger.warning(f"Could not extract random intercept variance, setting to 0")
            
            var_residual = float(result.scale)
            
            # Calculate ICC
            if var_intercept > 0 and var_residual > 0:
                icc = var_intercept / (var_intercept + var_residual)
            else:
                icc = 0.0
            
            # Calculate R-squared values
            r_squared = calculate_r_squared(result, model_data[outcome_var])
            r2_marginal = r_squared['r2_marginal']
            r2_conditional = r_squared['r2_conditional']
            
            # Create results dictionary with coefficients for both predictors
            model_results = {
                'outcome': outcome_var,
                'model_type': 'Home-Mobility Comparison',
                'n_obs': len(model_data),
                'n_participants': model_data['participant_id'].nunique(),
                'method': method,
                'r2_marginal': r2_marginal,  # Add R² values
                'r2_conditional': r2_conditional,
                'var_intercept': var_intercept,
                'var_residual': var_residual,
                'icc': icc,
                'aic': result.aic,
                'bic': result.bic
            }
            
            # Extract and format coefficients for both predictors
            for prefix, name in [(digital_home, 'Home'), (overlap, 'Mobility')]:
                within_var = f"{prefix}_within"
                between_var = f"{prefix}_between"
                
                within_coef = params.get(within_var, np.nan)
                within_pval = pvalues.get(within_var, np.nan)
                within_sig = '***' if within_pval < 0.001 else '**' if within_pval < 0.01 else '*' if within_pval < 0.05 else ''
                
                between_coef = params.get(between_var, np.nan)
                between_pval = pvalues.get(between_var, np.nan)
                between_sig = '***' if between_pval < 0.001 else '**' if between_pval < 0.01 else '*' if between_pval < 0.05 else ''
                
                # Get confidence intervals if available
                within_ci_low = conf_int.loc[within_var, 0] if within_var in conf_int.index else np.nan
                within_ci_high = conf_int.loc[within_var, 1] if within_var in conf_int.index else np.nan
                
                between_ci_low = conf_int.loc[between_var, 0] if between_var in conf_int.index else np.nan
                between_ci_high = conf_int.loc[between_var, 1] if between_var in conf_int.index else np.nan
                
                # Store values in results dictionary with location-specific keys
                model_results[f'{name}_within_coef'] = within_coef
                model_results[f'{name}_within_pval'] = within_pval
                model_results[f'{name}_within_sig'] = within_sig
                model_results[f'{name}_within_ci_low'] = within_ci_low
                model_results[f'{name}_within_ci_high'] = within_ci_high
                
                model_results[f'{name}_between_coef'] = between_coef
                model_results[f'{name}_between_pval'] = between_pval
                model_results[f'{name}_between_sig'] = between_sig
                model_results[f'{name}_between_ci_low'] = between_ci_low
                model_results[f'{name}_between_ci_high'] = between_ci_high
            
            # Log key results
            self.logger.info(f"Home-Mobility comparison for {outcome_var}:")
            self.logger.info(f"  Home within-effect: {model_results['Home_within_coef']:.4f}, p={model_results['Home_within_pval']:.4f} {model_results['Home_within_sig']}")
            self.logger.info(f"  Mobility within-effect: {model_results['Mobility_within_coef']:.4f}, p={model_results['Mobility_within_pval']:.4f} {model_results['Mobility_within_sig']}")
            self.logger.info(f"  R² marginal: {r2_marginal:.4f}, R² conditional: {r2_conditional:.4f}")
            self.logger.info(f"  ICC: {icc:.4f}")
            
            return model_results
            
        except Exception as e:
            self.logger.error(f"Error in home-mobility comparison: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None
    
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
        basic_models = {}          # Models without controls
        controlled_models = {}     # Models with demographic controls
        cross_frag_models = {}     # Digital vs. mobility fragmentation models (basic)
        cross_frag_controlled_models = {}  # Digital vs. mobility with full demographic controls
        moderation_analyses = {}   # Subgroup analyses
        home_mobility_comparisons = {}  # NEW: Home vs. mobility comparisons
        successful_models = 0
        
        # 1. Run basic multilevel models (without controls)
        self.logger.info("Running basic multilevel models (without controls)")
        for outcome_var in self.outcome_variables:
            outcome_models = {}
            for predictor in self.fragmentation_predictors:
                model_result = self.run_basic_multilevel_model(df, outcome_var, predictor, with_controls=False)
                if model_result:
                    outcome_models[predictor] = model_result
                    successful_models += 1
            
            if outcome_models:
                basic_models[outcome_var] = outcome_models
        
        # 2. Run multilevel models with demographic controls
        self.logger.info("Running multilevel models with demographic controls")
        for outcome_var in self.outcome_variables:
            outcome_models = {}
            for predictor in self.fragmentation_predictors:
                model_result = self.run_basic_multilevel_model(df, outcome_var, predictor, with_controls=True)
                if model_result:
                    outcome_models[predictor] = model_result
                    successful_models += 1
            
            if outcome_models:
                controlled_models[outcome_var] = outcome_models
        
        # 3a. Run basic cross-fragmentation models (without full demographic controls)
        self.logger.info("Running basic cross-fragmentation models")
        for outcome_var in self.outcome_variables:
            outcome_models = {}
            
            # Original cross-fragmentation models (without full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_fragmentation', 'mobility_fragmentation', with_full_controls=False)
            if model_result:
                outcome_models['digital_controlling_for_mobility'] = model_result
                successful_models += 1
            
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'mobility_fragmentation', 'digital_fragmentation', with_full_controls=False)
            if model_result:
                outcome_models['mobility_controlling_for_digital'] = model_result
                successful_models += 1
            
            # Digital home fragmentation controlling for mobility (without full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_fragmentation', 'mobility_fragmentation', with_full_controls=False)
            if model_result:
                outcome_models['digital_home_controlling_for_mobility'] = model_result
                successful_models += 1
            
            # Digital home fragmentation controlling for digital during mobility (without full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_fragmentation', 'overlap_fragmentation', with_full_controls=False)
            if model_result:
                outcome_models['digital_home_controlling_for_overlap'] = model_result
                successful_models += 1
            
            # Digital-mobility delta controlling for mobility (without full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_mobility_delta', 'mobility_fragmentation', with_full_controls=False)
            if model_result:
                outcome_models['home_mobility_delta_controlling_for_mobility'] = model_result
                successful_models += 1
            
            # Digital-mobility delta controlling for digital (without full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_mobility_delta', 'digital_fragmentation', with_full_controls=False)
            if model_result:
                outcome_models['home_mobility_delta_controlling_for_digital'] = model_result
                successful_models += 1
            
            if outcome_models:
                cross_frag_models[outcome_var] = outcome_models
        
        # 3b. Run cross-fragmentation models WITH full demographic controls
        self.logger.info("Running cross-fragmentation models with full demographic controls")
        for outcome_var in self.outcome_variables:
            outcome_models = {}
            
            # Original cross-fragmentation models (with full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_fragmentation', 'mobility_fragmentation', with_full_controls=True)
            if model_result:
                outcome_models['digital_controlling_for_mobility'] = model_result
                successful_models += 1
            
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'mobility_fragmentation', 'digital_fragmentation', with_full_controls=True)
            if model_result:
                outcome_models['mobility_controlling_for_digital'] = model_result
                successful_models += 1
            
            # Digital home fragmentation controlling for mobility (with full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_fragmentation', 'mobility_fragmentation', with_full_controls=True)
            if model_result:
                outcome_models['digital_home_controlling_for_mobility'] = model_result
                successful_models += 1
            
            # Digital home fragmentation controlling for digital during mobility (with full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_fragmentation', 'overlap_fragmentation', with_full_controls=True)
            if model_result:
                outcome_models['digital_home_controlling_for_overlap'] = model_result
                successful_models += 1
            
            # Digital-mobility delta controlling for mobility (with full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_mobility_delta', 'mobility_fragmentation', with_full_controls=True)
            if model_result:
                outcome_models['home_mobility_delta_controlling_for_mobility'] = model_result
                successful_models += 1
            
            # Digital-mobility delta controlling for digital (with full controls)
            model_result = self.run_cross_fragmentation_model(
                df, outcome_var, 'digital_home_mobility_delta', 'digital_fragmentation', with_full_controls=True)
            if model_result:
                outcome_models['home_mobility_delta_controlling_for_digital'] = model_result
                successful_models += 1
            
            if outcome_models:
                cross_frag_controlled_models[outcome_var] = outcome_models
        
        # 4. Run moderation analyses
        self.logger.info("Running moderation analyses")
        for outcome_var in self.outcome_variables:
            outcome_moderations = {}
            for predictor in self.fragmentation_predictors:
                predictor_moderations = {}
                for moderator_name, moderator_var in self.subset_variables.items():
                    # Skip the dataset moderator (since it's identical to age)
                    if moderator_name == 'dataset':
                        continue
                        
                    moderation_result = self.run_moderation_analysis(df, outcome_var, predictor, moderator_var)
                    if moderation_result:
                        predictor_moderations[moderator_name] = moderation_result
                        successful_models += 1
                
                if predictor_moderations:
                    outcome_moderations[predictor] = predictor_moderations
            
            if outcome_moderations:
                moderation_analyses[outcome_var] = outcome_moderations
        
        # 5. Run special home-mobility comparisons
        self.logger.info("Running special home-mobility comparisons")
        for outcome_var in self.outcome_variables:
            result = self.run_home_mobility_comparison(df, outcome_var)
            if result:
                home_mobility_comparisons[outcome_var] = result
                successful_models += 1
        
        # Store all results
        self.model_results['basic_models'] = basic_models
        self.model_results['controlled_models'] = controlled_models
        self.model_results['cross_frag_models'] = cross_frag_models
        self.model_results['cross_frag_controlled_models'] = cross_frag_controlled_models  # NEW: models with full controls
        self.model_results['moderation_analyses'] = moderation_analyses
        self.model_results['home_mobility_comparisons'] = home_mobility_comparisons
        
        self.logger.info(f"Completed multilevel analysis with {successful_models} successful models")
        
        return successful_models > 0
    
    def save_results(self):
        """Save multilevel model results to Excel file"""
        if not self.model_results:
            self.logger.warning("No results to save")
            return None
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Prepare summary dataframes
            basic_summary = []
            controlled_summary = []
            cross_frag_summary = []
            cross_frag_controlled_summary = []  # NEW: for cross-frag models with full controls
            moderation_summary = []
            comparison_summary = []
            
            # Process basic models (without controls)
            for outcome_var, outcome_models in self.model_results.get('basic_models', {}).items():
                for predictor, model_result in outcome_models.items():
                    # Create CI strings for cleaner presentation
                    within_ci = f"[{model_result.get('within_ci_low', np.nan):.2f}, {model_result.get('within_ci_high', np.nan):.2f}]"
                    between_ci = f"[{model_result.get('between_ci_low', np.nan):.2f}, {model_result.get('between_ci_high', np.nan):.2f}]"
                    
                    basic_summary.append({
                        'Outcome': outcome_var,
                        'Predictor': predictor,
                        'Model Type': 'Basic',
                        'Within-Effect': model_result.get('within_coef', np.nan),
                        'Within-P': model_result.get('within_pval', np.nan),
                        'Within-Sig': model_result.get('within_sig', ''),
                        'Within-CI': within_ci,
                        'Within-CI Low': model_result.get('within_ci_low', np.nan),
                        'Within-CI High': model_result.get('within_ci_high', np.nan),
                        'Between-Effect': model_result.get('between_coef', np.nan),
                        'Between-P': model_result.get('between_pval', np.nan),
                        'Between-Sig': model_result.get('between_sig', ''),
                        'Between-CI': between_ci,
                        'Between-CI Low': model_result.get('between_ci_low', np.nan),
                        'Between-CI High': model_result.get('between_ci_high', np.nan),
                        'N': model_result.get('n_obs', 0),
                        'Participants': model_result.get('n_participants', 0),
                        'ICC': model_result.get('icc', np.nan),
                        'R² Marginal': model_result.get('r2_marginal', np.nan),
                        'R² Conditional': model_result.get('r2_conditional', np.nan),
                        'AIC': model_result.get('aic', np.nan),
                        'BIC': model_result.get('bic', np.nan),
                        'Formula': model_result.get('formula', '')  # Add model formula to output
                    })
            
            # Process models with demographic controls
            control_var_effects = []  # New list for control variable effects
            
            for outcome_var, outcome_models in self.model_results.get('controlled_models', {}).items():
                for predictor, model_result in outcome_models.items():
                    # Create CI strings for cleaner presentation
                    within_ci = f"[{model_result.get('within_ci_low', np.nan):.2f}, {model_result.get('within_ci_high', np.nan):.2f}]"
                    between_ci = f"[{model_result.get('between_ci_low', np.nan):.2f}, {model_result.get('between_ci_high', np.nan):.2f}]"
                    
                    controlled_summary.append({
                        'Outcome': outcome_var,
                        'Predictor': predictor,
                        'Model Type': 'With Controls',
                        'Within-Effect': model_result.get('within_coef', np.nan),
                        'Within-P': model_result.get('within_pval', np.nan),
                        'Within-Sig': model_result.get('within_sig', ''),
                        'Within-CI': within_ci,
                        'Within-CI Low': model_result.get('within_ci_low', np.nan),
                        'Within-CI High': model_result.get('within_ci_high', np.nan),
                        'Between-Effect': model_result.get('between_coef', np.nan),
                        'Between-P': model_result.get('between_pval', np.nan),
                        'Between-Sig': model_result.get('between_sig', ''),
                        'Between-CI': between_ci,
                        'Between-CI Low': model_result.get('between_ci_low', np.nan),
                        'Between-CI High': model_result.get('between_ci_high', np.nan),
                        'N': model_result.get('n_obs', 0),
                        'Participants': model_result.get('n_participants', 0),
                        'ICC': model_result.get('icc', np.nan),
                        'R² Marginal': model_result.get('r2_marginal', np.nan),
                        'R² Conditional': model_result.get('r2_conditional', np.nan),
                        'AIC': model_result.get('aic', np.nan),
                        'BIC': model_result.get('bic', np.nan),
                        'Formula': model_result.get('formula', '')  # Add model formula to output
                    })
                    
                    # Extract control variable effects if present
                    if 'control_effects' in model_result:
                        for control_name, control_data in model_result['control_effects'].items():
                            for param_name, coefficient in control_data['params'].items():
                                # Extract the level name from parameter (e.g., "age_group[T.26-35]" -> "26-35")
                                if '[T.' in param_name:
                                    level = param_name.split('[T.')[1].rstrip(']')
                                elif param_name == control_name:
                                    level = 'Continuous'  # For continuous variables
                                else:
                                    level = param_name.replace(control_name, '').lstrip('_').lstrip('[').rstrip(']')
                                
                                # Get p-value and significance
                                p_value = control_data['pvals'].get(param_name, np.nan)
                                significance = control_data['sig'].get(param_name, '')
                                
                                # Get confidence intervals
                                ci_low = control_data['ci_low'].get(param_name, np.nan)
                                ci_high = control_data['ci_high'].get(param_name, np.nan)
                                
                                # Create formatted CI string
                                ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
                                
                                control_var_effects.append({
                                    'Outcome': outcome_var,
                                    'Predictor': predictor,
                                    'Control Variable': control_name,
                                    'Parameter': param_name,  # Add full parameter name
                                    'Level': level,
                                    'Coefficient': coefficient,
                                    'P-value': p_value,
                                    'Significance': significance,
                                    'CI': ci_str,
                                    'CI Low': ci_low,
                                    'CI High': ci_high,
                                    'Formula': model_result.get('formula', '')  # Add model formula to output
                                })
            
            # Process cross-fragmentation models (basic versions - without full controls)
            for outcome_var, outcome_models in self.model_results.get('cross_frag_models', {}).items():
                for model_name, model_result in outcome_models.items():
                    primary_pred = model_result.get('predictor', '')
                    control_pred = model_result.get('control_predictor', '')
                    
                    # Create CI strings for cleaner presentation
                    within_ci = f"[{model_result.get('within_ci_low', np.nan):.2f}, {model_result.get('within_ci_high', np.nan):.2f}]"
                    between_ci = f"[{model_result.get('between_ci_low', np.nan):.2f}, {model_result.get('between_ci_high', np.nan):.2f}]"
                    
                    cross_frag_summary.append({
                        'Outcome': outcome_var,
                        'Primary Predictor': primary_pred,
                        'Control Predictor': control_pred,
                        'Model Type': 'Cross-Fragmentation Basic',
                        'Within-Effect': model_result.get('within_coef', np.nan),
                        'Within-P': model_result.get('within_pval', np.nan),
                        'Within-Sig': model_result.get('within_sig', ''),
                        'Within-CI': within_ci,
                        'Within-CI Low': model_result.get('within_ci_low', np.nan),
                        'Within-CI High': model_result.get('within_ci_high', np.nan),
                        'Between-Effect': model_result.get('between_coef', np.nan),
                        'Between-P': model_result.get('between_pval', np.nan),
                        'Between-Sig': model_result.get('between_sig', ''),
                        'Between-CI': between_ci,
                        'Between-CI Low': model_result.get('between_ci_low', np.nan), 
                        'Between-CI High': model_result.get('between_ci_high', np.nan),
                        'Control Within-Effect': model_result.get('control_within_coef', np.nan),
                        'Control Within-P': model_result.get('control_within_pval', np.nan),
                        'Control Within-Sig': model_result.get('control_within_sig', ''),
                        'Control Between-Effect': model_result.get('control_between_coef', np.nan),
                        'Control Between-P': model_result.get('control_between_pval', np.nan),
                        'Control Between-Sig': model_result.get('control_between_sig', ''),
                        'N': model_result.get('n_obs', 0),
                        'Participants': model_result.get('n_participants', 0),
                        'ICC': model_result.get('icc', np.nan),
                        'R² Marginal': model_result.get('r2_marginal', np.nan),
                        'R² Conditional': model_result.get('r2_conditional', np.nan),
                        'AIC': model_result.get('aic', np.nan),
                        'BIC': model_result.get('bic', np.nan),
                        'Formula': model_result.get('formula', '')  # Add model formula to output
                    })
            
            # NEW: Process cross-fragmentation models with full controls
            for outcome_var, outcome_models in self.model_results.get('cross_frag_controlled_models', {}).items():
                for model_name, model_result in outcome_models.items():
                    primary_pred = model_result.get('predictor', '')
                    control_pred = model_result.get('control_predictor', '')
                    
                    # Create CI strings for cleaner presentation
                    within_ci = f"[{model_result.get('within_ci_low', np.nan):.2f}, {model_result.get('within_ci_high', np.nan):.2f}]"
                    between_ci = f"[{model_result.get('between_ci_low', np.nan):.2f}, {model_result.get('between_ci_high', np.nan):.2f}]"
                    
                    cross_frag_controlled_summary.append({
                        'Outcome': outcome_var,
                        'Primary Predictor': primary_pred,
                        'Control Predictor': control_pred,
                        'Model Type': 'Cross-Fragmentation With Full Controls',
                        'Within-Effect': model_result.get('within_coef', np.nan),
                        'Within-P': model_result.get('within_pval', np.nan),
                        'Within-Sig': model_result.get('within_sig', ''),
                        'Within-CI': within_ci,
                        'Within-CI Low': model_result.get('within_ci_low', np.nan),
                        'Within-CI High': model_result.get('within_ci_high', np.nan),
                        'Between-Effect': model_result.get('between_coef', np.nan),
                        'Between-P': model_result.get('between_pval', np.nan),
                        'Between-Sig': model_result.get('between_sig', ''),
                        'Between-CI': between_ci,
                        'Between-CI Low': model_result.get('between_ci_low', np.nan),
                        'Between-CI High': model_result.get('between_ci_high', np.nan),
                        'Control Within-Effect': model_result.get('control_within_coef', np.nan),
                        'Control Within-P': model_result.get('control_within_pval', np.nan),
                        'Control Within-Sig': model_result.get('control_within_sig', ''),
                        'Control Between-Effect': model_result.get('control_between_coef', np.nan),
                        'Control Between-P': model_result.get('control_between_pval', np.nan),
                        'Control Between-Sig': model_result.get('control_between_sig', ''),
                        'N': model_result.get('n_obs', 0),
                        'Participants': model_result.get('n_participants', 0),
                        'ICC': model_result.get('icc', np.nan),
                        'R² Marginal': model_result.get('r2_marginal', np.nan),
                        'R² Conditional': model_result.get('r2_conditional', np.nan),
                        'AIC': model_result.get('aic', np.nan),
                        'BIC': model_result.get('bic', np.nan),
                        'Formula': model_result.get('formula', '')  # Add model formula to output
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
                                    'Within-CI Low': subgroup_result.get('within_ci_low', np.nan),
                                    'Within-CI High': subgroup_result.get('within_ci_high', np.nan),
                                    'Between-Effect': subgroup_result.get('between_coef', np.nan),
                                    'Between-P': subgroup_result.get('between_pval', np.nan),
                                    'Between-Sig': subgroup_result.get('between_sig', ''),
                                    'Between-CI Low': subgroup_result.get('between_ci_low', np.nan),
                                    'Between-CI High': subgroup_result.get('between_ci_high', np.nan),
                                    'R² Marginal': subgroup_result.get('r2_marginal', np.nan),
                                    'R² Conditional': subgroup_result.get('r2_conditional', np.nan),
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
            
            # Process home-mobility comparison results
            home_mobility_summary = []
            for outcome_var, result in self.model_results.get('home_mobility_comparisons', {}).items():
                home_mobility_summary.append({
                    'Outcome': outcome_var,
                    'Model Type': 'Home-Mobility Comparison',
                    'Home Within-Effect': result.get('Home_within_coef', np.nan),
                    'Home Within-P': result.get('Home_within_pval', np.nan),
                    'Home Within-Sig': result.get('Home_within_sig', ''),
                    'Home Within-CI Low': result.get('Home_within_ci_low', np.nan),
                    'Home Within-CI High': result.get('Home_within_ci_high', np.nan),
                    'Mobility Within-Effect': result.get('Mobility_within_coef', np.nan),
                    'Mobility Within-P': result.get('Mobility_within_pval', np.nan),
                    'Mobility Within-Sig': result.get('Mobility_within_sig', ''),
                    'Mobility Within-CI Low': result.get('Mobility_within_ci_low', np.nan),
                    'Mobility Within-CI High': result.get('Mobility_within_ci_high', np.nan),
                    'R² Marginal': result.get('r2_marginal', np.nan),
                    'R² Conditional': result.get('r2_conditional', np.nan),
                    'N': result.get('n_obs', 0),
                    'Participants': result.get('n_participants', 0),
                    'AIC': result.get('aic', np.nan),
                    'BIC': result.get('bic', np.nan)
                })
            
            # Convert to dataframes
            basic_df = pd.DataFrame(basic_summary) if basic_summary else pd.DataFrame()
            controlled_df = pd.DataFrame(controlled_summary) if controlled_summary else pd.DataFrame()
            cross_frag_df = pd.DataFrame(cross_frag_summary) if cross_frag_summary else pd.DataFrame()
            cross_frag_controlled_df = pd.DataFrame(cross_frag_controlled_summary) if cross_frag_controlled_summary else pd.DataFrame()
            moderation_df = pd.DataFrame(moderation_summary) if moderation_summary else pd.DataFrame()
            comparison_df = pd.DataFrame(comparison_summary) if comparison_summary else pd.DataFrame()
            control_var_df = pd.DataFrame(control_var_effects) if control_var_effects else pd.DataFrame()
            home_mobility_df = pd.DataFrame(home_mobility_summary) if home_mobility_summary else pd.DataFrame()
            
            # Round numeric columns
            for df in [basic_df, controlled_df, cross_frag_df, cross_frag_controlled_df, moderation_df, comparison_df, control_var_df, home_mobility_df]:
                if not df.empty:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].round(4)
            
            # Save to Excel
            output_path = self.output_dir / f'multilevel_analysis_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(output_path) as writer:
                # Save main summary sheets
                if not basic_df.empty:
                    basic_df.to_excel(writer, sheet_name='Basic Models', index=False)
                
                if not controlled_df.empty:
                    controlled_df.to_excel(writer, sheet_name='Controlled Models', index=False)
                
                if not cross_frag_df.empty:
                    cross_frag_df.to_excel(writer, sheet_name='Cross-Frag Basic', index=False)
                
                # NEW: Save cross-fragmentation models with full controls
                if not cross_frag_controlled_df.empty:
                    cross_frag_controlled_df.to_excel(writer, sheet_name='Cross-Frag With Controls', index=False)
                
                if not moderation_df.empty:
                    moderation_df.to_excel(writer, sheet_name='Subgroup Effects', index=False)
                
                if not comparison_df.empty:
                    comparison_df.to_excel(writer, sheet_name='Group Comparisons', index=False)
                
                # Save control variable effects
                if not control_var_df.empty:
                    control_var_df.to_excel(writer, sheet_name='Control Variable Effects', index=False)
                
                # Create combined overview sheet
                combined_models = pd.concat([
                    basic_df.assign(model_category='Basic'),
                    controlled_df.assign(model_category='With Controls')
                ]) if not (basic_df.empty and controlled_df.empty) else pd.DataFrame()
                
                if not combined_models.empty:
                    combined_models.to_excel(writer, sheet_name='All Models Overview', index=False)
                
                # Create combined cross-fragmentation sheet
                if not (cross_frag_df.empty and cross_frag_controlled_df.empty):
                    all_cross_frag = pd.concat([
                        cross_frag_df,
                        cross_frag_controlled_df
                    ])
                    all_cross_frag.to_excel(writer, sheet_name='All Cross-Frag Models', index=False)
                
                # Create outcome-specific sheets
                for outcome_var in self.outcome_variables:
                    # Combine basic and controlled models for this outcome
                    outcome_basic = basic_df[basic_df['Outcome'] == outcome_var] if not basic_df.empty else pd.DataFrame()
                    outcome_controlled = controlled_df[controlled_df['Outcome'] == outcome_var] if not controlled_df.empty else pd.DataFrame()
                    
                    outcome_combined = pd.concat([
                        outcome_basic.assign(model_category='Basic'),
                        outcome_controlled.assign(model_category='With Controls')
                    ]) if not (outcome_basic.empty and outcome_controlled.empty) else pd.DataFrame()
                    
                    if not outcome_combined.empty:
                        sheet_name = f"{outcome_var.replace('_std', '')}_Models"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                        outcome_combined.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Cross-fragmentation results for this outcome (combine basic and controlled)
                    outcome_cross_basic = cross_frag_df[cross_frag_df['Outcome'] == outcome_var] if not cross_frag_df.empty else pd.DataFrame()
                    outcome_cross_controlled = cross_frag_controlled_df[cross_frag_controlled_df['Outcome'] == outcome_var] if not cross_frag_controlled_df.empty else pd.DataFrame()
                    
                    outcome_cross = pd.concat([outcome_cross_basic, outcome_cross_controlled]) if not (outcome_cross_basic.empty and outcome_cross_controlled.empty) else pd.DataFrame()
                    
                    if not outcome_cross.empty:
                        sheet_name = f"{outcome_var.replace('_std', '')}_CrossFrag"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                        outcome_cross.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Control variable effects for this outcome
                    outcome_controls = control_var_df[control_var_df['Outcome'] == outcome_var] if not control_var_df.empty else pd.DataFrame()
                    if not outcome_controls.empty:
                        sheet_name = f"{outcome_var.replace('_std', '')}_Controls"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                        outcome_controls.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Moderation results for this outcome
                    outcome_mod = moderation_df[moderation_df['Outcome'] == outcome_var] if not moderation_df.empty else pd.DataFrame()
                    if not outcome_mod.empty:
                        sheet_name = f"{outcome_var.replace('_std', '')}_Subgroups"
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[:30]
                            outcome_mod.to_excel(writer, sheet_name=sheet_name, index=False)
                            # Create combined summary table for presentation
                            presenter_df = pd.DataFrame()
                            if not controlled_df.empty:
                                # Select key columns for presentation
                                presenter_cols = ['Outcome', 'Predictor', 'Within-Effect', 'Within-P', 'Within-Sig', 
                                                'Within-CI', 'Between-Effect', 'Between-P', 'Between-Sig', 
                                                'Between-CI', 'R² Marginal', 'R² Conditional', 'N', 'Participants']
                                presenter_df = controlled_df[presenter_cols].copy()
                                
                                # Format numeric columns for presentation
                                for col in ['Within-Effect', 'Between-Effect', 'R² Marginal', 'R² Conditional']:
                                    presenter_df[col] = presenter_df[col].round(2)
                                for col in ['Within-P', 'Between-P']:
                                    presenter_df[col] = presenter_df[col].round(3)
                                
                                # Create formatted coefficient strings with significance markers
                                presenter_df['Within-Coef'] = presenter_df.apply(
                                    lambda x: f"{x['Within-Effect']:.2f}{x['Within-Sig']}", axis=1)
                                presenter_df['Between-Coef'] = presenter_df.apply(
                                    lambda x: f"{x['Between-Effect']:.2f}{x['Within-Sig']}", axis=1)
                                
                                presenter_df = presenter_df[['Outcome', 'Predictor', 'Within-Coef', 'Within-CI', 
                                                         'Between-Coef', 'Between-CI', 'R² Marginal', 'R² Conditional', 'N', 'Participants']]
                                presenter_df.to_excel(writer, sheet_name='Presentation Table', index=False)
                
                # Create control-variable specific sheets
                if not control_var_df.empty:
                    control_vars = control_var_df['Control Variable'].unique()
                    for control_var in control_vars:
                        control_specific = control_var_df[control_var_df['Control Variable'] == control_var]
                        if not control_specific.empty:
                            sheet_name = f"{control_var.title()}_Effects"
                            if len(sheet_name) > 30:  # Excel sheet name limit
                                sheet_name = sheet_name[:30]
                            control_specific.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Save home-mobility comparison results
                if not home_mobility_df.empty:
                    home_mobility_df.to_excel(writer, sheet_name='Home-Mobility Comparison', index=False)
            
            self.logger.info(f"Saved multilevel model results to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run the improved multilevel analysis."""
    try:
        # Create analyzer with debug mode
        analyzer = ImprovedMultilevelAnalysis(debug=True)
        
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