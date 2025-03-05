#!/usr/bin/env python3
"""
Multilevel Regression Analysis for Pooled SURREAL and TLV Dataset

This script performs multilevel regression analysis to examine the relationship
between fragmentation metrics and emotional outcomes across both the SURREAL and TLV
datasets, while controlling for demographic variables and dataset source.

Usage:
    python multilevel_analysis.py [--output_dir /path/to/results] [--debug]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import logging
import argparse
from datetime import datetime
import traceback
import warnings


class MultilevelAnalysis:
    def __init__(self, output_dir=None, debug=False):
        """Initialize the multilevel analysis.
        
        Args:
            output_dir (str): Directory to save outputs
            debug (bool): Enable debug logging
        """
        # Hard-coded path to pooled data
        self.input_path = Path('pooled/processed/pooled_stai_data.csv')
        
        # Set output directory
        script_dir = Path(__file__).parent
        self.output_dir = Path(output_dir) if output_dir else script_dir / "results" / "multilevel"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.debug = debug
        self.results = []
        
        # Setup logging
        self._setup_logging()
        
        # Define variable categories
        self.emotion_metrics = ['anxiety_score_std', 'mood_score_std']
        self.fragmentation_metrics = {
            'raw': ['digital_fragmentation', 'mobility_fragmentation', 'overlap_fragmentation']
        }
        self.demographic_vars = ['gender_standardized', 'location_type', 'age_group']
        
        # No duration metrics in this analysis, but keep the structure for compatibility
        self.duration_metrics = []
        self.episode_metrics = []
        
        # Column name map for display purposes
        self.col_name_map = {
            'anxiety_score_std': 'Standardized Anxiety',
            'mood_score_std': 'Standardized Mood',
            'digital_fragmentation': 'Digital Fragmentation',
            'mobility_fragmentation': 'Mobility Fragmentation',
            'overlap_fragmentation': 'Overlap Fragmentation'
        }

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
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
        self.logger.info(f"Initializing multilevel analysis")
        self.logger.info(f"Input data: {self.input_path}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_and_preprocess_data(self):
        """Load and preprocess the pooled dataset."""
        self.logger.info(f"Loading data from {self.input_path}")
        
        try:
            # Load dataset
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Check for participant ID and uniqueness
            participants = df['participant_id'].unique()
            n_participants = len(participants)
            self.logger.info(f"Number of participants: {n_participants}")
            
            # Dataset sources
            dataset_sources = df['dataset_source'].unique()
            self.logger.info(f"Dataset sources: {list(dataset_sources)}")
            
            # Number of observations from each dataset
            for source in dataset_sources:
                n_source = df[df['dataset_source'] == source].shape[0]
                self.logger.info(f"Observations from {source}: {n_source}")
            
            # Check for missing values in key variables
            for var in self.emotion_metrics + self.fragmentation_metrics['raw'] + self.demographic_vars:
                missing = df[var].isna().sum()
                missing_pct = 100 * missing / df.shape[0]
                self.logger.info(f"Missing values in {var}: {missing} ({missing_pct:.1f}%)")
            
            # Create dummy variables for categorical predictors
            for var in ['gender_standardized', 'location_type', 'age_group', 'dataset_source']:
                if var in df.columns:
                    # Ensure it's categorical
                    df[var] = df[var].astype('category')
                    
                    # Create dummies
                    dummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
                    
                    # Log the created dummy variables
                    self.logger.info(f"Created dummy variables: {list(dummies.columns)}")
                    
                    # Add to dataframe
                    df = pd.concat([df, dummies], axis=1)
            
            # Person-mean center the continuous predictors for within/between decomposition
            grouped = df.groupby('participant_id')
            
            for var in self.fragmentation_metrics['raw']:
                # Skip variables with too many missing values
                if df[var].isna().sum() > df.shape[0] * 0.5:
                    self.logger.warning(f"Skipping {var} for person-centering due to >50% missing values")
                    continue
                
                # Calculate person means
                person_means = grouped[var].transform('mean')
                
                # Calculate within-person deviations
                df[f'{var}_within'] = df[var] - person_means
                
                # Keep the person means for between-person effects
                df[f'{var}_between'] = person_means
                
                # Log
                within_var = df[f'{var}_within'].var()
                between_var = df[f'{var}_between'].var()
                
                self.logger.info(f"Person-centered {var}: within-person variance = {within_var:.4f}, between-person variance = {between_var:.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def run_multilevel_analysis(self):
        """Run the multilevel analysis for all emotion metrics."""
        self.logger.info("Starting multilevel analysis")
        
        try:
            # Check if data is available
            if not hasattr(self, 'data') or self.data is None:
                # If not already loaded, load and preprocess
                self.data = self.load_and_preprocess_data()
                
                if self.data is None or self.data.empty:
                    self.logger.error("No data available for analysis")
                    return None
            
            # Suppress specific warnings
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning, message="overflow encountered in exp")
            
            # Iteration through all emotion metrics and fragmentation metrics
            for outcome in self.emotion_metrics:
                self.logger.info(f"\nAnalyzing outcome: {outcome}")
                
                # Get all fragmentation predictors
                predictors = self.fragmentation_metrics['raw']
                
                for predictor in predictors:
                    # Skip if too many missing values for the predictor
                    if self.data[predictor].isna().sum() > self.data.shape[0] * 0.5:
                        self.logger.warning(f"Skipping {predictor} - too many missing values")
                        continue
                    
                    self.logger.info(f"  Predictor: {predictor}")
                    
                    # Run the stepwise multilevel models
                    self._run_stepwise_models(outcome, predictor)
                    
                    # Run dataset interaction model
                    self._run_interaction_model(outcome, predictor)
                    
            self.logger.info(f"Multilevel analysis completed with {len(self.results)} model results")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in multilevel analysis: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def _run_stepwise_models(self, outcome, predictor):
        """Run the stepwise multilevel models for a specific outcome-predictor pair."""
        try:
            # Filter data to non-missing values for this analysis
            data = self.data.dropna(subset=[outcome, predictor, f'{predictor}_within', f'{predictor}_between'])
            
            if data.shape[0] < 30:
                self.logger.warning(f"  Insufficient observations ({data.shape[0]}) for modeling. Skipping.")
                return None
            
            self.logger.info(f"  Running stepwise models with {data.shape[0]} observations")
            
            # STEP 1: Base model with no controls
            self.logger.info(f"  Step 1: Base model (no controls)")
            model_result = self._fit_base_model(outcome, predictor, data)
            if model_result:
                self.results.append(model_result)
                
            # STEP 2: Control for dataset source
            self.logger.info(f"  Step 2: Dataset control model")
            model_result = self._fit_dataset_control_model(outcome, predictor, data)
            if model_result:
                self.results.append(model_result)
            
            # STEP 3: Control for demographics
            self.logger.info(f"  Step 3: Demographics control model")
            model_result = self._fit_demographics_model(outcome, predictor, data)
            if model_result:
                self.results.append(model_result)
                
            return True
            
        except Exception as e:
            self.logger.error(f"  Error in stepwise models for {outcome} ~ {predictor}: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def _fit_base_model(self, outcome, predictor, data):
        """Fit the base multilevel model with no controls."""
        try:
            # Define the formula with within and between effects
            formula = f"{outcome} ~ 1 + {predictor}_within + {predictor}_between"
            
            # Add random intercept for participant
            formula += " + (1|participant_id)"
            
            # Log the formula
            self.logger.info(f"  Formula: {formula}")
            
            # Fit the model
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, data, groups=data["participant_id"])
                    result = model.fit(method=['powell'])
            except Exception as fit_error:
                self.logger.error(f"  Error fitting model: {str(fit_error)}")
                return None
            
            # Get model indices
            within_idx = result.params.index.get_loc(f"{predictor}_within")
            between_idx = result.params.index.get_loc(f"{predictor}_between")
            
            # Extract coefficients and p-values
            within_coef = result.params.iloc[within_idx]
            within_se = result.bse.iloc[within_idx]
            within_p = result.pvalues.iloc[within_idx]
            
            between_coef = result.params.iloc[between_idx]
            between_se = result.bse.iloc[between_idx]
            between_p = result.pvalues.iloc[between_idx]
            
            # Handle NaN p-values
            if np.isnan(within_p):
                within_p = 1.0
            if np.isnan(between_p):
                between_p = 1.0
            
            # Create result dictionary
            result_dict = {
                'dependent_var': outcome,
                'predictor': predictor,
                'model': 'Base Model',
                'n_obs': data.shape[0],
                'n_groups': len(data['participant_id'].unique()),
                'AIC': result.aic,
                'BIC': result.bic,
                'within_coef': within_coef,
                'within_se': within_se,
                'within_p': within_p,
                'within_sig': self._pval_to_stars(within_p),
                'between_coef': between_coef,
                'between_se': between_se,
                'between_p': between_p,
                'between_sig': self._pval_to_stars(between_p)
            }
            
            # Log result
            self.logger.info(f"  Base Model Results:")
            self.logger.info(f"    Within-person effect: b={within_coef:.4f}, SE={within_se:.4f}, p={within_p:.4f} {self._pval_to_stars(within_p)}")
            self.logger.info(f"    Between-person effect: b={between_coef:.4f}, SE={between_se:.4f}, p={between_p:.4f} {self._pval_to_stars(between_p)}")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"  Error in base model for {outcome} ~ {predictor}: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def _fit_dataset_control_model(self, outcome, predictor, data):
        """Fit multilevel model controlling for dataset source."""
        try:
            # Define the formula with dataset control
            formula = f"{outcome} ~ 1 + {predictor}_within + {predictor}_between + dataset_source"
            
            # Add random intercept for participant
            formula += " + (1|participant_id)"
            
            # Log the formula
            self.logger.info(f"  Formula: {formula}")
            
            # Fit the model
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, data, groups=data["participant_id"])
                    result = model.fit(method=['powell'])
            except Exception as fit_error:
                self.logger.error(f"  Error fitting dataset control model: {str(fit_error)}")
                return None
            
            # Get model indices
            within_idx = result.params.index.get_loc(f"{predictor}_within")
            between_idx = result.params.index.get_loc(f"{predictor}_between")
            
            # Extract coefficients and p-values
            within_coef = result.params.iloc[within_idx]
            within_se = result.bse.iloc[within_idx]
            within_p = result.pvalues.iloc[within_idx]
            
            between_coef = result.params.iloc[between_idx]
            between_se = result.bse.iloc[between_idx]
            between_p = result.pvalues.iloc[between_idx]
            
            # Handle NaN p-values
            if np.isnan(within_p):
                within_p = 1.0
            if np.isnan(between_p):
                between_p = 1.0
            
            # Create result dictionary
            result_dict = {
                'dependent_var': outcome,
                'predictor': predictor,
                'model': 'Dataset Control',
                'n_obs': data.shape[0],
                'n_groups': len(data['participant_id'].unique()),
                'AIC': result.aic,
                'BIC': result.bic,
                'within_coef': within_coef,
                'within_se': within_se,
                'within_p': within_p,
                'within_sig': self._pval_to_stars(within_p),
                'between_coef': between_coef,
                'between_se': between_se,
                'between_p': between_p,
                'between_sig': self._pval_to_stars(between_p)
            }
            
            # Log result
            self.logger.info(f"  Dataset Control Model Results:")
            self.logger.info(f"    Within-person effect: b={within_coef:.4f}, SE={within_se:.4f}, p={within_p:.4f} {self._pval_to_stars(within_p)}")
            self.logger.info(f"    Between-person effect: b={between_coef:.4f}, SE={between_se:.4f}, p={between_p:.4f} {self._pval_to_stars(between_p)}")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"  Error in dataset control model for {outcome} ~ {predictor}: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def _fit_demographics_model(self, outcome, predictor, data):
        """Fit multilevel model controlling for demographics."""
        try:
            # Define the formula with demographic controls
            formula = f"{outcome} ~ 1 + {predictor}_within + {predictor}_between + dataset_source"
            
            # Add demographic controls
            # Determine which demographic variables to include based on data availability
            available_demographics = []
            for var in ['gender_standardized', 'location_type', 'age_group']:
                if var in data.columns and data[var].nunique() > 1:
                    available_demographics.append(var)
            
            # Add available demographics to formula
            if available_demographics:
                formula += " + " + " + ".join(available_demographics)
            
            # Add random intercept for participant
            formula += " + (1|participant_id)"
            
            # Log the formula
            self.logger.info(f"  Formula: {formula}")
            
            # Fit the model
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, data, groups=data["participant_id"])
                    result = model.fit(method=['powell'])
            except Exception as fit_error:
                self.logger.error(f"  Error fitting demographics model: {str(fit_error)}")
                return None
            
            # Get model indices
            within_idx = result.params.index.get_loc(f"{predictor}_within")
            between_idx = result.params.index.get_loc(f"{predictor}_between")
            
            # Extract coefficients and p-values
            within_coef = result.params.iloc[within_idx]
            within_se = result.bse.iloc[within_idx]
            within_p = result.pvalues.iloc[within_idx]
            
            between_coef = result.params.iloc[between_idx]
            between_se = result.bse.iloc[between_idx]
            between_p = result.pvalues.iloc[between_idx]
            
            # Handle NaN p-values
            if np.isnan(within_p):
                within_p = 1.0
            if np.isnan(between_p):
                between_p = 1.0
            
            # Create result dictionary
            result_dict = {
                'dependent_var': outcome,
                'predictor': predictor,
                'model': 'Demographics Control',
                'n_obs': data.shape[0],
                'n_groups': len(data['participant_id'].unique()),
                'AIC': result.aic,
                'BIC': result.bic,
                'within_coef': within_coef,
                'within_se': within_se,
                'within_p': within_p,
                'within_sig': self._pval_to_stars(within_p),
                'between_coef': between_coef,
                'between_se': between_se,
                'between_p': between_p,
                'between_sig': self._pval_to_stars(between_p)
            }
            
            # Log result
            self.logger.info(f"  Demographics Control Model Results:")
            self.logger.info(f"    Within-person effect: b={within_coef:.4f}, SE={within_se:.4f}, p={within_p:.4f} {self._pval_to_stars(within_p)}")
            self.logger.info(f"    Between-person effect: b={between_coef:.4f}, SE={between_se:.4f}, p={between_p:.4f} {self._pval_to_stars(between_p)}")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"  Error in demographics model for {outcome} ~ {predictor}: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def _run_interaction_model(self, outcome, predictor):
        """Run model with dataset interaction to test for differential effects."""
        try:
            # Filter data to non-missing values
            data = self.data.dropna(subset=[outcome, predictor, f'{predictor}_within', f'{predictor}_between', 'dataset_source'])
            
            if data.shape[0] < 30:
                self.logger.warning(f"  Insufficient observations ({data.shape[0]}) for interaction model. Skipping.")
                return None
            
            # Check if there are at least 10 observations per dataset
            datasets = data['dataset_source'].unique()
            skip = False
            for ds in datasets:
                n_ds = data[data['dataset_source'] == ds].shape[0]
                if n_ds < 10:
                    self.logger.warning(f"  Insufficient observations for dataset {ds} ({n_ds}). Skipping interaction.")
                    skip = True
            
            if skip:
                return None
            
            self.logger.info(f"  Running dataset interaction model with {data.shape[0]} observations")
            
            try:
                # Create a formula with within-person effect and interaction with dataset
                # We'll use only the within-person effect to reduce complexity
                formula = f"{outcome} ~ 1 + {predictor}_within * dataset_source"
                
                # Add the between-person effect
                formula += f" + {predictor}_between"
                
                # Add random intercept
                formula += " + (1|participant_id)"
                
                self.logger.info(f"  Formula: {formula}")
                
                # Fit the model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = smf.mixedlm(formula, data, groups=data["participant_id"])
                    model_fit = model.fit(method=['powell'])
                
                # Process results
                result_dict = {
                    'dependent_var': outcome,
                    'predictor': predictor,
                    'model': 'Dataset Interaction',
                    'n_obs': data.shape[0],
                    'n_groups': len(data['participant_id'].unique()),
                    'AIC': model_fit.aic,
                    'BIC': model_fit.bic
                }
                
                # Extract main effect coefficient
                within_idx = model_fit.params.index.get_loc(f"{predictor}_within")
                within_coef = model_fit.params.iloc[within_idx]
                within_se = model_fit.bse.iloc[within_idx]
                within_p = model_fit.pvalues.iloc[within_idx]
                
                # Handle NaN p-values
                if np.isnan(within_p):
                    within_p = 1.0
                
                result_dict.update({
                    'within_coef': within_coef,
                    'within_se': within_se,
                    'within_p': within_p,
                    'within_sig': self._pval_to_stars(within_p)
                })
                
                # Find and extract interaction term
                interaction_pattern = f"{predictor}_within:dataset_source"
                interaction_idx = None
                
                for i, param in enumerate(model_fit.params.index):
                    if interaction_pattern in param:
                        interaction_idx = i
                        break
                
                if interaction_idx is not None:
                    interaction_coef = model_fit.params.iloc[interaction_idx]
                    interaction_se = model_fit.bse.iloc[interaction_idx]
                    interaction_p = model_fit.pvalues.iloc[interaction_idx]
                    
                    # Handle NaN p-values
                    if np.isnan(interaction_p):
                        interaction_p = 1.0  # Set to non-significant
                    
                    result_dict.update({
                        'interaction_coef': interaction_coef,
                        'interaction_se': interaction_se,
                        'interaction_p': interaction_p,
                        'interaction_sig': self._pval_to_stars(interaction_p)
                    })
                
                # Add to results
                self.results.append(result_dict)
                
                # Log results
                self.logger.info(f"  Results for dataset interaction model:")
                self.logger.info(f"    Main effect: b={result_dict['within_coef']:.4f}, p={result_dict['within_p']:.4f} {result_dict['within_sig']}")
                if 'interaction_coef' in result_dict:
                    self.logger.info(f"    Interaction effect: b={result_dict['interaction_coef']:.4f}, p={result_dict['interaction_p']:.4f} {result_dict['interaction_sig']}")
                
                return result_dict
                
            except np.linalg.LinAlgError:
                self.logger.error(f"  Error fitting interaction model: Singular matrix")
                return None
                
        except Exception as e:
            self.logger.error(f"  Error fitting interaction model: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def _pval_to_stars(self, p):
        """Convert p-value to significance stars."""
        if np.isnan(p):
            return ""
        elif p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "†"
        else:
            return ""

    def save_results(self):
        """Save analysis results to Excel and CSV."""
        if not self.results:
            self.logger.warning("No results to save")
            return None
        
        try:
            # Create directories
            results_dir = self.output_dir
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Create timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.results)
            
            # Format p-values for Excel
            for col in ['within_p', 'between_p', 'interaction_p']:
                if col in results_df.columns:
                    # Replace NaN with 1.0 (not significant)
                    results_df[col] = results_df[col].fillna(1.0)
            
            # Save Excel file with all results
            excel_path = results_dir / f'pooled_multilevel_results_{timestamp}.xlsx'
            with pd.ExcelWriter(excel_path) as writer:
                results_df.to_excel(writer, sheet_name='All Models', index=False)
                
                # Add significant results sheet
                sig_results = results_df[
                    (results_df['within_p'] < 0.05) | 
                    (results_df['between_p'] < 0.05 if 'between_p' in results_df.columns else False) | 
                    (results_df['interaction_p'] < 0.05 if 'interaction_p' in results_df.columns else False)
                ]
                
                if not sig_results.empty:
                    sig_results.to_excel(writer, sheet_name='Significant Results', index=False)
            
            self.logger.info(f"Saved detailed results to {excel_path}")
            
            # Save CSV for easier processing
            csv_path = results_dir / f'pooled_multilevel_results_{timestamp}.csv'
            results_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV results to {csv_path}")
            
            # Create separate emotion-specific files
            for emotion in self.emotion_metrics:
                emotion_results = results_df[results_df['dependent_var'] == emotion]
                if not emotion_results.empty:
                    emotion_display = emotion.replace('_score_std', '')
                    emotion_path = results_dir / f'{emotion_display}_results_{timestamp}.xlsx'
                    with pd.ExcelWriter(emotion_path) as writer:
                        emotion_results.to_excel(writer, sheet_name=f'All {emotion_display}', index=False)
                        
                        # Add significant results
                        sig_emotion = emotion_results[
                            (emotion_results['within_p'] < 0.05) | 
                            (emotion_results['between_p'] < 0.05 if 'between_p' in emotion_results.columns else False) |
                            (emotion_results['interaction_p'] < 0.05 if 'interaction_p' in emotion_results.columns else False)
                        ]
                        if not sig_emotion.empty:
                            sig_emotion.to_excel(writer, sheet_name='Significant Results', index=False)
                    
                    self.logger.info(f"Saved {emotion_display} results to {emotion_path}")
            
            return excel_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

    def print_summary(self):
        """Print a summary of the results."""
        if not self.results:
            self.logger.warning("No results to summarize")
            return
        
        try:
            self.logger.info("\n" + "="*50)
            self.logger.info("MULTILEVEL ANALYSIS SUMMARY")
            self.logger.info("="*50)
            
            # Count significant findings by type
            sig_within = 0
            sig_between = 0
            sig_interaction = 0
            
            for result in self.results:
                if 'within_p' in result and result['within_p'] < 0.05:
                    sig_within += 1
                if 'between_p' in result and result['between_p'] < 0.05:
                    sig_between += 1
                if 'interaction_p' in result and result['interaction_p'] < 0.05:
                    sig_interaction += 1
            
            self.logger.info(f"Total models analyzed: {len(self.results)}")
            self.logger.info(f"Significant within-person effects: {sig_within}")
            self.logger.info(f"Significant between-person effects: {sig_between}")
            self.logger.info(f"Significant interaction effects: {sig_interaction}")
            
            # Show most significant findings
            if self.results:
                self.logger.info("\nMOST SIGNIFICANT FINDINGS:")
                
                # Filter for significant results and sort by p-value
                sig_results_within = [r for r in self.results if 'within_p' in r and r['within_p'] < 0.05]
                sig_results_between = [r for r in self.results if 'between_p' in r and r['between_p'] < 0.05]
                sig_results_interaction = [r for r in self.results if 'interaction_p' in r and r['interaction_p'] < 0.05]
                
                # Sort by p-value
                sig_results_within.sort(key=lambda x: x['within_p'])
                sig_results_between.sort(key=lambda x: x['between_p'])
                sig_results_interaction.sort(key=lambda x: x['interaction_p'])
                
                # Show top findings
                if sig_results_within:
                    self.logger.info("\nTOP WITHIN-PERSON EFFECTS:")
                    for i, result in enumerate(sig_results_within[:3]):
                        self.logger.info(f"  {i+1}. {result['dependent_var']} ~ {result['predictor']} (Model: {result['model']})")
                        self.logger.info(f"     b={result['within_coef']:.4f}, p={result['within_p']:.4f} {result['within_sig']}")
                
                if sig_results_between:
                    self.logger.info("\nTOP BETWEEN-PERSON EFFECTS:")
                    for i, result in enumerate(sig_results_between[:3]):
                        self.logger.info(f"  {i+1}. {result['dependent_var']} ~ {result['predictor']} (Model: {result['model']})")
                        self.logger.info(f"     b={result['between_coef']:.4f}, p={result['between_p']:.4f} {result['between_sig']}")
                
                if sig_results_interaction:
                    self.logger.info("\nTOP INTERACTION EFFECTS:")
                    for i, result in enumerate(sig_results_interaction[:3]):
                        self.logger.info(f"  {i+1}. {result['dependent_var']} ~ {result['predictor']} × Dataset (Model: {result['model']})")
                        self.logger.info(f"     b={result['interaction_coef']:.4f}, p={result['interaction_p']:.4f} {result['interaction_sig']}")
            
            self.logger.info("="*50)
        
        except Exception as e:
            self.logger.error(f"Error printing summary: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())


def main():
    """Main function to run the multilevel analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run multilevel regression analysis on pooled SURREAL-TLV data')
    
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save analysis results')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Run analysis
    try:
        analysis = MultilevelAnalysis(
            output_dir=args.output_dir,
            debug=args.debug
        )
        
        # Load and preprocess data
        df = analysis.load_and_preprocess_data()
        
        if df is None or df.empty:
            print("Error: Failed to load or preprocess data")
            return 1
        
        # Store the data in the analysis object
        analysis.data = df
        
        # Run the multilevel analysis
        results = analysis.run_multilevel_analysis()
        
        # Save results
        analysis.save_results()
        
        # Print summary
        analysis.print_summary()
        
        print("Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 