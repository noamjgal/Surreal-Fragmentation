#!/usr/bin/env python3
"""
Simplified Stepwise Regression Analysis for Fragmentation Metrics

This script performs 12 separate stepwise regressions (3 fragmentation metrics × 4 outcomes).
Each regression follows a fixed 4-step process with pre-determined control variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging
from datetime import datetime

class FixedStepwiseRegression:
    def __init__(self, debug=False):
        """Initialize the stepwise regression analysis class with hardcoded paths."""
        # Hardcoded paths
        self.participant_file = "/Users/noamgal/DSProjects/Fragmentation/SURREAL/processed/merged_data/ema_fragmentation_demographics_participant_norm.csv"
        self.output_dir = Path("SURREAL/results/regression_analysis")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        self.regression_results = []
        
        # The fragmentation metrics we want to focus on
        self.fragmentation_predictors = [
            'frag_digital_fragmentation_index', 
            'frag_mobility_fragmentation_index', 
            'frag_overlap_fragmentation_index'
        ]
        
        # Outcome variables (stress/anxiety measures)
        self.outcome_variables = [
            'ema_STAI_Y_A_6_zstd',  # Standardized anxiety score
            'ema_STAI_Y_A_6_raw',   # Raw anxiety score
            'ema_CES_D_8_zstd',     # Standardized mood/depression score
            'ema_CES_D_8_raw'       # Raw mood/depression score
        ]
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'fixed_stepwise_{timestamp}.log'
        
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
        self.logger.info(f"Initializing fixed stepwise regression analysis")
        self.logger.info(f"Participant-normalized data: {self.participant_file}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load participant-normalized data.
        
        Returns:
            DataFrame: Participant-normalized dataset
        """
        # Load participant-normalized data
        self.logger.info(f"Loading participant-normalized data from {self.participant_file}")
        try:
            df = pd.read_csv(self.participant_file)
            self.logger.info(f"Data loaded successfully with shape: {df.shape}")
            
            # Replace hyphens in column names with underscores for easier referencing
            df.columns = [col.replace('-', '_') for col in df.columns]
            
            # Update outcome variables to match the new column names
            self.outcome_variables = [var.replace('-', '_') for var in self.outcome_variables]
            
            # Handle problematic column names
            if 'City.center' in df.columns:
                # Rename City.center to avoid patsy formula issues
                df['city_center'] = df['City.center']
                self.logger.info("Renamed 'City.center' to 'city_center' to avoid patsy formula issues")
            
            # Check for required columns
            required_controls = ['is_weekend', 'Gender', 'city_center']
            missing_controls = [col for col in required_controls if col not in df.columns]
            
            if 'is_weekend' in missing_controls:
                self.logger.warning("'is_weekend' not found in dataset. Creating dummy variable.")
                # Create a dummy is_weekend variable (replace with actual logic if known)
                df['is_weekend'] = False
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

    def _run_regression(self, df, formula, outcome_var, step_name, frag_predictor):
        """Run a single regression model and extract results.
        
        Args:
            df (DataFrame): Dataset
            formula (str): Regression formula
            outcome_var (str): Outcome variable name
            step_name (str): Step description
            frag_predictor (str): Main fragmentation predictor to focus on
            
        Returns:
            dict: Regression results
        """
        self.logger.info(f"Running regression: {formula}")
        
        try:
            # Extract all variables from formula
            all_vars = formula.replace(f"{outcome_var} ~", "").split("+")
            all_vars = [var.strip() for var in all_vars] + [outcome_var]
            
            # Clean data for regression (remove missing values)
            regression_data = df[all_vars].dropna()
            n_obs = len(regression_data)
            
            if n_obs < 10:
                self.logger.warning(f"Too few observations ({n_obs}) for reliable regression")
                return {
                    'outcome': outcome_var,
                    'frag_type': frag_predictor,
                    'step': step_name,
                    'formula': formula,
                    'n_obs': n_obs,
                    'converged': False,
                    'error': "Too few observations"
                }
            
            # Fit the model
            model = ols(formula, data=regression_data)
            results = model.fit()
            
            # Extract key statistics
            r_squared = results.rsquared
            adj_r_squared = results.rsquared_adj
            aic = results.aic
            bic = results.bic
            f_stat = results.fvalue
            f_pvalue = results.f_pvalue
            
            # Extract coefficient, standard error, t-value, and p-value for the fragmentation predictor
            # Instead of creating separate columns for each predictor, use generic column names
            if frag_predictor in results.params.index:
                coef = results.params[frag_predictor]
                se = results.bse[frag_predictor]
                t_value = results.tvalues[frag_predictor]
                p_value = results.pvalues[frag_predictor]
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            else:
                coef = np.nan
                se = np.nan
                t_value = np.nan
                p_value = np.nan
                sig = ''
            
            # Get predictors list from formula
            predictors = [var.strip() for var in formula.split("~")[1].split("+")]
            
            # Prepare results dictionary with generic column names
            result_dict = {
                'outcome': outcome_var,
                'frag_type': frag_predictor,
                'step': step_name,
                'formula': formula,
                'n_obs': n_obs,
                'converged': True,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'aic': aic,
                'bic': bic,
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'predictors': predictors,
                'coef_frag': coef,  # Generic column for fragmentation coefficient
                'se_frag': se,      # Generic column for standard error
                't_frag': t_value,  # Generic column for t-value
                'p_frag': p_value,  # Generic column for p-value
                'sig_frag': sig     # Generic column for significance
            }
            
            # Log key results for main predictor
            self.logger.info(
                f"Regression results for {step_name}: "
                f"n={n_obs}, R²={r_squared:.4f}, "
                f"{frag_predictor} coef={coef:.4f}, p={p_value:.4f}"
            )
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in regression: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
                
            return {
                'outcome': outcome_var,
                'frag_type': frag_predictor,
                'step': step_name,
                'formula': formula,
                'n_obs': 0,
                'converged': False,
                'error': str(e)
            }
    
    def run_fixed_stepwise_model(self, df, outcome_var, frag_predictor):
        """Run a 4-step regression model for a specific outcome and fragmentation predictor.
        
        Args:
            df (DataFrame): Dataset
            outcome_var (str): Outcome variable
            frag_predictor (str): Fragmentation predictor to use as main IV
            
        Returns:
            list: List of regression results for each step
        """
        self.logger.info(f"Running 4-step regression for {outcome_var} with {frag_predictor}")
        
        # Determine which duration metric to use based on the fragmentation predictor
        if 'digital' in frag_predictor:
            duration_var = 'frag_digital_total_duration'
        elif 'mobility' in frag_predictor:
            duration_var = 'frag_mobility_total_duration'
        else:  # overlap
            duration_var = 'frag_overlap_total_duration'
        
        # Define the 4 steps with fixed control variables
        steps = [
            {
                'name': f"Step 1: {frag_predictor}",
                'formula': f"{outcome_var} ~ {frag_predictor}"
            },
            {
                'name': f"Step 2: Added {duration_var}",
                'formula': f"{outcome_var} ~ {frag_predictor} + {duration_var}"
            },
            {
                'name': f"Step 3: Added is_weekend",
                'formula': f"{outcome_var} ~ {frag_predictor} + {duration_var} + is_weekend"
            },
            {
                'name': f"Step 4: Added Gender",
                'formula': f"{outcome_var} ~ {frag_predictor} + {duration_var} + is_weekend + Gender"
            }
        ]
        
        # Run each step
        results = []
        for step in steps:
            step_result = self._run_regression(
                df, 
                step['formula'], 
                outcome_var, 
                step['name'],
                frag_predictor
            )
            results.append(step_result)
        
        return results
    
    def run_all_regressions(self, df):
        """Run all 12 regression models (3 fragmentation predictors × 4 outcomes).
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            bool: Success status
        """
        if df is None or df.empty:
            self.logger.error("No data available for analysis")
            return False
            
        # Check if required columns exist
        for frag_predictor in self.fragmentation_predictors:
            if frag_predictor not in df.columns:
                self.logger.error(f"Missing required predictor: {frag_predictor}")
                return False
                
        for outcome_var in self.outcome_variables:
            if outcome_var not in df.columns:
                self.logger.error(f"Missing required outcome: {outcome_var}")
                return False
        
        all_results = []
        
        # Run regressions for each combination
        for outcome_var in self.outcome_variables:
            for frag_predictor in self.fragmentation_predictors:
                results = self.run_fixed_stepwise_model(df, outcome_var, frag_predictor)
                all_results.extend(results)
        
        self.regression_results = all_results
        self.logger.info(f"Completed all regressions. Generated {len(all_results)} models.")
        
        return len(all_results) > 0
        
    def save_results(self):
        """Save regression results to Excel file"""
        if not self.regression_results:
            self.logger.warning("No results to save")
            return None
        
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.regression_results)
            
            # Round numeric columns for cleaner output
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].round(4)
            
            # Save to Excel
            output_path = self.output_dir / f'fixed_stepwise_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(output_path) as writer:
                # Save all results to first sheet
                results_df.to_excel(writer, sheet_name='All Results', index=False)
                
                # Create separate sheets for each outcome variable
                for outcome_var in self.outcome_variables:
                    outcome_results = results_df[results_df['outcome'] == outcome_var]
                    if not outcome_results.empty:
                        # Create a readable sheet name
                        sheet_name = outcome_var.replace('ema_', '').replace('_', ' ')
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[-30:]
                        outcome_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create a compact summary table focused on fragmentation effects
                summary_data = []
                for outcome_var in self.outcome_variables:
                    outcome_name = outcome_var.replace('ema_', '').replace('_', ' ')
                    
                    for frag_predictor in self.fragmentation_predictors:
                        # Extract just the type (digital, mobility, overlap)
                        frag_type = frag_predictor.split('_')[1]  # Extract 'digital', 'mobility', or 'overlap'
                        
                        # Get all steps for this combination
                        steps = results_df[
                            (results_df['outcome'] == outcome_var) & 
                            (results_df['frag_type'] == frag_predictor)
                        ]
                        
                        for _, row in steps.iterrows():
                            if not row['converged']:
                                continue
                            
                            # Get step number (1-4)
                            step_num = row['step'].split(':')[0].strip()
                            
                            # Create more compact summary row
                            summary_data.append({
                                'Outcome': outcome_name,
                                'Fragmentation Type': frag_type,
                                'Step': step_num,
                                'Controls': row['step'].split('Added ')[-1] if 'Added' in row['step'] else 'None',
                                'N': row['n_obs'],
                                'R²': row['r_squared'],
                                'Coefficient': row['coef_frag'],
                                'P-value': row['p_frag'],
                                'Sig': row['sig_frag']
                            })
                
                if summary_data:
                    # Create pivot-style table for easier comparison
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Sort by outcome, fragmentation type, then step
                    summary_df = summary_df.sort_values(['Outcome', 'Fragmentation Type', 'Step'])
                    
                    # Save to Excel
                    summary_df.to_excel(writer, sheet_name='Compact Summary', index=False)
                    
                    # Also create a pivot table for easier interpretation
                    pivot_data = []
                    
                    # Group by outcome and fragmentation type
                    for outcome in summary_df['Outcome'].unique():
                        for frag_type in summary_df['Fragmentation Type'].unique():
                            # Get data for this combination
                            subset = summary_df[
                                (summary_df['Outcome'] == outcome) &
                                (summary_df['Fragmentation Type'] == frag_type)
                            ]
                            
                            if not subset.empty:
                                # Get values for each step
                                steps = {}
                                for _, row in subset.iterrows():
                                    step = row['Step']
                                    steps[f'Step {step} Coef'] = row['Coefficient']
                                    steps[f'Step {step} P'] = row['P-value']
                                    steps[f'Step {step} Sig'] = row['Sig']
                                    
                                # Add summary row
                                pivot_data.append({
                                    'Outcome': outcome,
                                    'Fragmentation Type': frag_type,
                                    'N': subset['N'].iloc[0],  # Use N from first step
                                    **steps
                                })
                    
                    if pivot_data:
                        pivot_df = pd.DataFrame(pivot_data)
                        pivot_df.to_excel(writer, sheet_name='Pivot Summary', index=False)
            
            self.logger.info(f"Saved regression results to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                import traceback
                self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run the fixed stepwise regression analysis."""
    try:
        # Create analyzer
        analyzer = FixedStepwiseRegression(debug=True)
        
        # Load data
        df = analyzer.load_data()
        
        if df is None or df.empty:
            print("Error: Failed to load data")
            return 1
        
        # Run regressions
        if analyzer.run_all_regressions(df):
            # Save results
            results_path = analyzer.save_results()
            
            if results_path:
                print(f"Fixed stepwise regression analysis completed successfully!")
                print(f"Results saved to: {results_path}")
                return 0
            else:
                print("Error: Failed to save results")
                return 1
        else:
            print("Error: Failed to run regressions")
            return 1
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())