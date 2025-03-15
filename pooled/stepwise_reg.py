#!/usr/bin/env python3
"""
Pooled Stepwise Regression Analysis

This script performs stepwise regressions across pooled datasets (SURREAL and TLV),
examining the relationship between 3 fragmentation metrics and 4 emotion outcomes.
Each regression follows a fixed 4-step process with pre-determined control variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging
from datetime import datetime
import warnings

class PooledStepwiseRegression:
    def __init__(self, output_dir=None, debug=False):
        """Initialize the pooled stepwise regression analysis.
        
        Args:
            output_dir (str): Directory to save outputs
            debug (bool): Enable debug logging
        """
        # Set paths for participant-level standardized data
        self.pooled_data_path = Path("pooled/processed/pooled_stai_data_population.csv")
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = Path(__file__).parent
            self.output_dir = script_dir / "results" / "regression_analysis"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        self.regression_results = []
        
        # Define fragmentation predictors - updated for pooled dataset
        self.fragmentation_predictors = [
            'digital_fragmentation', 
            'mobility_fragmentation', 
            'overlap_fragmentation'
        ]
        
        # Define outcome variables - updated to only use standardized scores
        self.outcome_variables = [
            'anxiety_score_std',   # Standardized anxiety score
            'mood_score_std'       # Standardized mood/depression score
        ]
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pooled_stepwise_{timestamp}.log'
        
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
        self.logger.info(f"Initializing pooled stepwise regression analysis with participant-level standardization")
        self.logger.info(f"Pooled data (participant standardized): {self.pooled_data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load participant-normalized pooled data.
        
        Returns:
            DataFrame: Participant-normalized pooled dataset
        """
        # Load participant-normalized data
        self.logger.info(f"Loading participant-normalized pooled data from {self.pooled_data_path}")
        try:
            df = pd.read_csv(self.pooled_data_path)
            self.logger.info(f"Pooled data loaded successfully with shape: {df.shape}")
            
            # Replace hyphens in column names with underscores for easier referencing
            df.columns = [col.replace('-', '_') for col in df.columns]
            
            # Update variables to match the new column names if needed
            self.outcome_variables = [var.replace('-', '_') for var in self.outcome_variables]
            self.fragmentation_predictors = [var.replace('-', '_') for var in self.fragmentation_predictors]
            
            # Check for required columns
            required_vars = self.fragmentation_predictors + self.outcome_variables
            missing_vars = [var for var in required_vars if var not in df.columns]
            if missing_vars:
                self.logger.error(f"Missing critical variables: {missing_vars}")
                self.logger.error("Available columns: " + ", ".join(df.columns.tolist()))
                return None
            
            # Check for required control variables - using location_type
            required_controls = ['location_type', 'gender_standardized', 'age_group']
            missing_controls = [col for col in required_controls if col not in df.columns]
            if missing_controls:
                self.logger.warning(f"Missing control variables: {missing_controls}")
                
                # Create gender_standardized if missing but Gender exists (transform, not generate)
                if 'gender_standardized' not in df.columns:
                    gender_cols = ['Gender', 'gender', 'sex']
                    found_gender = False
                    for gender_col in gender_cols:
                        if gender_col in df.columns:
                            df['gender_standardized'] = df[gender_col].apply(
                                lambda x: 'female' if str(x).strip().lower() in ['f', 'female', 'נקבה'] 
                                else 'male' if str(x).strip().lower() in ['m', 'male', 'זכר'] 
                                else 'other'
                            )
                            self.logger.info(f"Created 'gender_standardized' from '{gender_col}'")
                            found_gender = True
                            break
                    if not found_gender:
                        self.logger.error("No gender column found for standardization")
                
                # Create location_type if missing but city_center exists (transform, not generate)
                if 'location_type' not in df.columns:
                    location_cols = ['city_center', 'City.center', 'location', 'School']
                    found_location = False
                    for location_col in location_cols:
                        if location_col in df.columns:
                            df['location_type'] = df[location_col].apply(
                                lambda x: 'city_center' if str(x).strip().lower() in ['yes', 'center', 'city_center'] 
                                else 'suburb'
                            )
                            self.logger.info(f"Created 'location_type' from '{location_col}'")
                            found_location = True
                            break
                    if not found_location:
                        self.logger.error("No location column found for standardization")
                
                # Create age_group if missing but can be inferred from dataset_source (transform, not generate)
                if 'age_group' not in df.columns and 'dataset_source' in df.columns:
                    df['age_group'] = df['dataset_source'].apply(
                        lambda x: 'adult' if x == 'surreal' else 'adolescent'
                    )
                    self.logger.info("Created 'age_group' from dataset_source")
                
                # Check again after transformations
                still_missing = [col for col in required_controls if col not in df.columns]
                if still_missing:
                    self.logger.error(f"Still missing required control variables after transformation: {still_missing}")
            
            # Log location type distribution
            if 'location_type' in df.columns:
                location_counts = df['location_type'].value_counts()
                self.logger.info(f"Location type distribution:")
                for location, count in location_counts.items():
                    location_pct = 100 * count / len(df)
                    self.logger.info(f"  {location}: {count} ({location_pct:.1f}%)")
            
            # Check fragmentation duration variables - but don't create defaults
            for frag_type in ['digital', 'mobility', 'overlap']:
                duration_var = f'{frag_type}_duration'
                if duration_var not in df.columns:
                    self.logger.warning(f"{duration_var} not found - models will use only fragmentation metrics")
                    # Do NOT create synthetic duration variables with default values
                else:
                    # Log statistics on duration variables
                    valid_count = df[duration_var].notna().sum()
                    self.logger.info(f"{duration_var} has {valid_count} valid values ({100*valid_count/len(df):.1f}%)")
                    if valid_count > 0:
                        self.logger.info(f"  Mean: {df[duration_var].mean():.2f}, Min: {df[duration_var].min():.2f}, Max: {df[duration_var].max():.2f}")
            
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
            # Extract variables from formula, but handle categorical variables correctly
            # For formulas like "outcome ~ pred1 + C(pred2)", we just need the base columns
            base_vars = []
            formula_parts = formula.replace(f"{outcome_var} ~", "").split("+")
            
            for part in formula_parts:
                part = part.strip()
                if part.startswith('C(') and part.endswith(')'):
                    # Extract the variable name from C(variable_name)
                    var_name = part[2:-1].strip()
                    base_vars.append(var_name)
                else:
                    base_vars.append(part)
            
            # Add the outcome variable
            all_vars = base_vars + [outcome_var]
            
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
        frag_type = frag_predictor.split('_')[0]  # Extract 'digital', 'mobility', or 'overlap'
        duration_var = f'{frag_type}_duration'
        
        # Check if duration variable exists - if not, skip steps that use it
        has_duration = duration_var in df.columns and df[duration_var].notna().sum() > 5
        
        # Define the steps
        if has_duration:
            # Define all steps including duration if available
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
                    'name': f"Step 3: Added location_type",
                    'formula': f"{outcome_var} ~ {frag_predictor} + {duration_var} + C(location_type)"
                },
                {
                    'name': f"Step 4: Added gender_standardized",
                    'formula': f"{outcome_var} ~ {frag_predictor} + {duration_var} + C(location_type) + C(gender_standardized)"
                }
            ]
        else:
            # Skip steps with duration if not available
            self.logger.warning(f"{duration_var} not available - skipping duration control steps")
            steps = [
                {
                    'name': f"Step 1: {frag_predictor}",
                    'formula': f"{outcome_var} ~ {frag_predictor}"
                },
                {
                    'name': f"Step 2: Added location_type",
                    'formula': f"{outcome_var} ~ {frag_predictor} + C(location_type)"
                },
                {
                    'name': f"Step 3: Added gender_standardized",
                    'formula': f"{outcome_var} ~ {frag_predictor} + C(location_type) + C(gender_standardized)"
                }
            ]
        
        # Optional final step - add age_group
        if 'age_group' in df.columns:
            if has_duration:
                steps.append({
                    'name': f"Step 5: Added age_group",
                    'formula': f"{outcome_var} ~ {frag_predictor} + {duration_var} + C(location_type) + C(gender_standardized) + C(age_group)"
                })
            else:
                steps.append({
                    'name': f"Step 4: Added age_group",
                    'formula': f"{outcome_var} ~ {frag_predictor} + C(location_type) + C(gender_standardized) + C(age_group)"
                })
        
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
        """Run all regression models (3 fragmentation predictors × 4 outcomes).
        
        Args:
            df (DataFrame): Dataset
            
        Returns:
            bool: Success status
        """
        if df is None or df.empty:
            self.logger.error("No data available for analysis")
            return False
        
        # Validate required columns
        missing_predictors = [pred for pred in self.fragmentation_predictors if pred not in df.columns]
        if missing_predictors:
            self.logger.error(f"Missing required predictors: {missing_predictors}")
            return False
                
        missing_outcomes = [outcome for outcome in self.outcome_variables if outcome not in df.columns]
        if missing_outcomes:
            self.logger.error(f"Missing required outcomes: {missing_outcomes}")
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
            output_path = self.output_dir / f'pooled_stepwise_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(output_path) as writer:
                # Save all results to first sheet
                results_df.to_excel(writer, sheet_name='All Results', index=False)
                
                # Create separate sheets for each outcome variable
                for outcome_var in self.outcome_variables:
                    # Skip if this outcome wasn't in the results
                    if outcome_var not in results_df['outcome'].values:
                        continue
                        
                    outcome_results = results_df[results_df['outcome'] == outcome_var]
                    if not outcome_results.empty:
                        # Create a readable sheet name
                        sheet_name = outcome_var.replace('_std', '').replace('_raw', '')
                        if len(sheet_name) > 30:  # Excel sheet name limit
                            sheet_name = sheet_name[-30:]
                        outcome_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create a compact summary table focused on fragmentation effects
                summary_data = []
                for outcome_var in self.outcome_variables:
                    # Skip if this outcome wasn't in the results
                    if outcome_var not in results_df['outcome'].values:
                        continue
                        
                    outcome_name = outcome_var.replace('_std', '').replace('_raw', '')
                    
                    for frag_predictor in self.fragmentation_predictors:
                        # Skip if this predictor wasn't in the results
                        if frag_predictor not in results_df['frag_type'].values:
                            continue
                            
                        # Extract just the type (digital, mobility, overlap)
                        frag_type = frag_predictor.split('_')[0]  # Extract 'digital', 'mobility', or 'overlap'
                        
                        # Get all steps for this combination
                        steps = results_df[
                            (results_df['outcome'] == outcome_var) & 
                            (results_df['frag_type'] == frag_predictor)
                        ]
                        
                        for _, row in steps.iterrows():
                            if not row['converged']:
                                continue
                            
                            # Get step number
                            step_num = row['step'].split(':')[0].strip()
                            
                            # Create more compact summary row
                            summary_data.append({
                                'Outcome': outcome_name,
                                'Standardization': 'Standardized' if '_std' in outcome_var else 'Raw',
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
                    
                    # Sort by outcome, standardization, fragmentation type, then step
                    summary_df = summary_df.sort_values(['Outcome', 'Standardization', 'Fragmentation Type', 'Step'])
                    
                    # Save to Excel
                    summary_df.to_excel(writer, sheet_name='Compact Summary', index=False)
                    
                    # Also create a pivot table for easier interpretation
                    pivot_data = []
                    
                    # Group by outcome, standardization, and fragmentation type
                    summary_groups = summary_df.groupby(['Outcome', 'Standardization', 'Fragmentation Type'])
                    
                    for (outcome, std_type, frag_type), group in summary_groups:
                        # Get values for each step
                        steps = {}
                        for _, row in group.iterrows():
                            step = row['Step']
                            steps[f'Step {step} Coef'] = row['Coefficient']
                            steps[f'Step {step} P'] = row['P-value']
                            steps[f'Step {step} Sig'] = row['Sig']
                            
                        # Add summary row
                        pivot_data.append({
                            'Outcome': outcome,
                            'Standardization': std_type,
                            'Fragmentation Type': frag_type,
                            'N': group['N'].iloc[0],  # Use N from first step
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
    """Main function to run the pooled stepwise regression analysis."""
    try:
        # Create analyzer
        analyzer = PooledStepwiseRegression(debug=True)
        
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
                print(f"Pooled stepwise regression analysis completed successfully!")
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
    # Ignore certain warnings
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    
    exit(main())