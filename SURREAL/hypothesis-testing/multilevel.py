#!/usr/bin/env python3
"""
Multilevel Regression Analysis for Fragmentation-Emotion Relationships

This script performs multilevel regression analysis to examine the relationship
between digital fragmentation metrics and emotional outcomes (from EMA data),
while controlling for demographic variables.

Usage:
    python multilevel.py --input_file /path/to/merged_data.csv --output_dir /path/to/results
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
import matplotlib.pyplot as plt
import seaborn as sns
import re
import traceback

class MultilevelAnalysis:
    def __init__(self, input_path, output_dir, debug=False):
        """Initialize the multilevel analysis class.
        
        Args:
            input_path (str): Path to merged data file
            output_dir (str): Directory to save analysis results
            debug (bool): Enable debug logging
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self._setup_logging()
        self.results = []
        self.col_name_map = {}  # For storing column name mappings
        
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
        self.logger.info(f"Initializing multilevel analysis with input: {self.input_path}")

    def load_and_preprocess_data(self):
        """Load and preprocess data for analysis."""
        try:
            self.logger.info(f"Loading data from {self.input_path}")
            
            # Load data
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Data loaded successfully with shape: {df.shape}")
            
            # Create safe column names for formula-based modeling
            safe_cols = []
            
            for col in df.columns:
                # Check if column has special characters (periods, hyphens, spaces)
                if re.search(r'[.\-\s]', col):
                    # Create safe name by replacing special chars with underscore
                    safe_name = re.sub(r'[.\-\s]', '_', col)
                    self.col_name_map[col] = safe_name
                    safe_cols.append(safe_name)
                else:
                    safe_cols.append(col)
            
            # Apply the mapping to rename columns
            if self.col_name_map:
                df = df.rename(columns=self.col_name_map)
                self.logger.info(f"Renamed {len(self.col_name_map)} columns to avoid formula issues")
                if self.debug:
                    self.logger.debug(f"Column mapping: {self.col_name_map}")
            
            # Convert date column to datetime if it's not already
            if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                self.logger.info("Converted date column to datetime")
            
            # Add weekend indicator if not already present
            if 'date' in df.columns and 'is_weekend' not in df.columns:
                df['is_weekend'] = df['date'].dt.dayofweek >= 5
                df['is_weekend'] = df['is_weekend'].astype(int)
                self.logger.info("Created weekend indicator variable")
            
            # Make sure participant ID columns are consistently named
            for col in df.columns:
                if 'participant' in col.lower() and 'id' in col.lower():
                    df[col] = df[col].astype(str)
            
            # Create a clean/consistent participant ID column
            if 'participant_id_clean' not in df.columns:
                if 'participant_id_ema' in df.columns:
                    df['participant_id_clean'] = df['participant_id_ema'].astype(str)
                elif 'participant_id_frag' in df.columns:
                    df['participant_id_clean'] = df['participant_id_frag'].astype(str)
                elif 'Participant_ID' in df.columns:
                    df['participant_id_clean'] = df['Participant_ID'].astype(str)
                else:
                    # Try to find any column with 'participant' and 'id'
                    for col in df.columns:
                        if 'participant' in col.lower() and 'id' in col.lower():
                            df['participant_id_clean'] = df[col].astype(str)
                            self.logger.info(f"Using {col} as participant ID column")
                            break
                    else:
                        self.logger.warning("No participant ID column found!")
            
            # Identify variable groups based on available columns
            self._define_variable_groups(df)
            
            # Calculate z-scores for fragmentation metrics if not already present
            for metric in self.fragmentation_metrics['raw']:
                z_col = f"{metric}_zstd"
                if z_col not in df.columns and metric in df.columns:
                    # Create a copy to avoid SettingWithCopyWarning
                    group_means = df.groupby('participant_id_clean')[metric].transform('mean')
                    group_stds = df.groupby('participant_id_clean')[metric].transform('std')
                    
                    # Avoid division by zero
                    mask = (group_stds > 0)
                    df.loc[mask, z_col] = (df.loc[mask, metric] - group_means[mask]) / group_stds[mask]
                    # For groups with 0 std, set z-score to 0
                    df.loc[~mask, z_col] = 0
                    
                    self.logger.info(f"Created z-score column: {z_col}")
            
            # Store the data for analysis - make a copy to avoid SettingWithCopyWarning
            self.data = df.copy()
            
            self.logger.info(f"Data preprocessing complete. Final shape: {df.shape}")
            self.logger.info(f"Emotion metrics: {self.emotion_metrics}")
            self.logger.info(f"Fragmentation metrics: {self.fragmentation_metrics}")
            self.logger.info(f"Episode metrics: {self.episode_metrics}")
            self.logger.info(f"Duration metrics: {self.duration_metrics}")
            self.logger.info(f"Demographic variables: {self.demographic_vars}")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading or preprocessing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _define_variable_groups(self, df):
        """Define groups of columns for analysis based on available data."""
        # Define groups of columns to check (at least one from each group must exist)
        column_groups = {
            'emotion': {
                'stai': ['ema_STAI_Y_A_6_zstd', 'ema_STAI_Y_A_6_raw', 'STAI_Y_A_6_zstd', 'STAI_Y_A_6_raw'],
                'cesd': ['ema_CES_D_8_zstd', 'ema_CES_D_8_raw', 'CES_D_8_zstd', 'CES_D_8_raw']
            },
            'fragmentation': {
                'raw': ['frag_digital_fragmentation_index', 'frag_mobility_fragmentation_index', 
                       'frag_overlap_fragmentation_index', 'digital_fragmentation_index', 'mobility_fragmentation_index', 
                       'overlap_fragmentation_index'],
                'zstd': []  # Will be filled with z-standardized columns
            },
            'episodes': ['frag_digital_episode_count', 'frag_mobility_episode_count', 'frag_overlap_episode_count',
                        'digital_episode_count', 'mobility_episode_count', 'overlap_episode_count'],
            'duration': ['frag_digital_total_duration', 'frag_mobility_total_duration', 'frag_overlap_total_duration',
                        'digital_total_duration', 'mobility_total_duration', 'overlap_total_duration'],
            'demographics': ['Gender', 'gender', 'age', 'City.center', 'gender_code']
        }
        
        # Find available columns that match each category
        self.emotion_metrics = []
        for emotion_type, candidates in column_groups['emotion'].items():
            for col in candidates:
                if col in df.columns:
                    self.emotion_metrics.append(col)
                    self.logger.info(f"Found emotion metric: {col}")
        
        # Sort fragmentation metrics
        self.fragmentation_metrics = {'raw': [], 'zstd': []}
        for col in column_groups['fragmentation']['raw']:
            if col in df.columns:
                self.fragmentation_metrics['raw'].append(col)
                # Add corresponding z-standardized column if it exists
                z_col = f"{col}_zstd"
                if z_col in df.columns:
                    self.fragmentation_metrics['zstd'].append(z_col)
        
        # Find episode and duration metrics
        self.episode_metrics = [col for col in column_groups['episodes'] if col in df.columns]
        self.duration_metrics = [col for col in column_groups['duration'] if col in df.columns]
        
        # Find demographic variables
        self.demographic_vars = [col for col in column_groups['demographics'] if col in df.columns]
        
        # Validate that we have enough variables to proceed
        if not self.emotion_metrics:
            self.logger.warning("No emotion metrics found!")
        
        if not self.fragmentation_metrics['raw']:
            self.logger.warning("No fragmentation metrics found!")
    
    def run_multilevel_analysis(self):
        """Run the full multilevel analysis with various model specifications."""
        
        if not hasattr(self, 'data'):
            self.logger.error("No data loaded. Call load_and_preprocess_data() first.")
            return
        
        self.logger.info("Starting multilevel analysis")
        
        # Define model steps with progressive controls
        control_steps = [
            [],  # Base model with no controls
            ['is_weekend'],  # Control for weekend effect
            self.duration_metrics[:2] if len(self.duration_metrics) >= 2 else self.duration_metrics,  # Control for activity duration
        ]
        
        # Add demographic controls if available
        if self.demographic_vars:
            demographic_controls = control_steps[-1] + self.demographic_vars
            control_steps.append(demographic_controls)
        
        # Run primary analysis: fragmentation effects on emotions
        for emotion in self.emotion_metrics:
            for frag_metric in self.fragmentation_metrics['raw']:
                frag_metric_z = f"{frag_metric}_zscore"
                
                self.logger.info(f"\nAnalyzing {emotion} ~ {frag_metric}")
                
                for step_idx, controls in enumerate(control_steps):
                    model_name = f"Step {step_idx + 1}"
                    controls_desc = ', '.join(controls) if controls else 'None'
                    self.logger.info(f"  Model {model_name} with controls: {controls_desc}")
                    
                    try:
                        self._run_multilevel_model(
                            dv=emotion,
                            pred_raw=frag_metric,
                            pred_z=frag_metric_z,
                            controls=controls,
                            model_name=model_name
                        )
                    except Exception as e:
                        self.logger.error(f"Error in model {model_name}: {str(e)}")
                        if self.debug:
                            self.logger.error(traceback.format_exc())
        
        # Run additional analyses for episode counts and durations
        for metric in self.episode_metrics + self.duration_metrics:
            for emotion in self.emotion_metrics:
                self.logger.info(f"\nAnalyzing {emotion} ~ {metric}")
                
                try:
                    self._run_multilevel_model(
                        dv=emotion,
                        pred_raw=metric,
                        controls=['is_weekend'],
                        model_name="Secondary Analysis"
                    )
                except Exception as e:
                    self.logger.error(f"Error in secondary analysis model: {str(e)}")
                    if self.debug:
                        self.logger.error(traceback.format_exc())
        
        self.logger.info("Multilevel analysis completed")
        return self.results
    
    def _run_multilevel_model(self, dv, pred_raw, pred_z=None, controls=None, model_name=""):
        """Run a single multilevel model with within-person and between-person effects.
        
        Args:
            dv (str): Dependent variable name
            pred_raw (str): Predictor variable name
            pred_z (str): Z-standardized version of predictor (if available)
            controls (list): List of control variables to include
            model_name (str): Name/identifier for this model
        
        Returns:
            dict: Results dictionary with model parameters and statistics
        """
        controls = controls or []
        
        try:
            # Filter data to only include rows with both DV and predictor
            current_data = self.data.dropna(subset=[dv, pred_raw]).copy()  # Create copy to avoid warnings
            
            if len(current_data) < 10:
                self.logger.warning(f"Too few observations ({len(current_data)}) for {dv} ~ {pred_raw}")
                return None
                
            n_participants = current_data['participant_id_clean'].nunique()
            if n_participants < 5:
                self.logger.warning(f"Too few participants ({n_participants}) for {dv} ~ {pred_raw}")
                return None
            
            # Calculate within-person and between-person variables
            within_var, between_var = self._calculate_within_between(current_data, pred_raw)
            
            # Prepare control variables part of the formula
            control_formula = ""
            if controls:
                for var in controls:
                    if var in current_data.columns:
                        control_formula += f" + {var}"
                    else:
                        self.logger.warning(f"Control variable {var} not found in data")
            
            # Build formula for mixed model (within + between)
            formula = f"{dv} ~ within_pred + between_pred + {control_formula[3:]}" if control_formula else f"{dv} ~ within_pred + between_pred"
            
            self.logger.debug(f"Model formula: {formula}")
            
            # Create variables for model
            current_data.loc[:, 'within_pred'] = within_var
            current_data.loc[:, 'between_pred'] = between_var
            
            # Try different optimization methods if the model doesn't converge
            model = smf.mixedlm(
                formula=formula,
                data=current_data,
                groups=current_data['participant_id_clean']
            )
            
            # Try different methods for fitting the model
            methods = ['lbfgs', 'cg', 'bfgs', 'powell', 'nm']
            converged = False
            results = None
            
            for method in methods:
                try:
                    results = model.fit(reml=True, method=method)
                    if results.converged:
                        self.logger.info(f"Model converged using {method} method")
                        converged = True
                        break
                    self.logger.warning(f"Model did not converge with {method} method, trying another...")
                except Exception as e:
                    self.logger.warning(f"Error fitting model with {method} method: {str(e)}")
            
            # If the model still didn't converge, try with a simpler random effects structure
            if not converged and results is not None:
                self.logger.warning("Trying simpler random effects structure")
                try:
                    model = smf.mixedlm(
                        formula=formula,
                        data=current_data,
                        groups=current_data['participant_id_clean'],
                        re_formula="1"  # Only random intercept, no random slopes
                    )
                    results = model.fit(reml=True)
                    if results.converged:
                        self.logger.info("Model converged with simpler random effects structure")
                        converged = True
                except Exception as e:
                    self.logger.warning(f"Error fitting simplified model: {str(e)}")
            
            # If no model converged, use the last results (with warning)
            if not converged:
                self.logger.warning(f"Model did not converge for {dv} ~ {pred_raw} with any method")
                if results is None:
                    self.logger.error("Failed to fit model")
                    return None
            
            # Get results
            result_dict = {
                'dependent_var': dv,
                'predictor': pred_raw,
                'model': model_name,
                'n_observations': len(current_data),
                'n_participants': n_participants,
                'within_coef': results.params['within_pred'],
                'within_se': results.bse['within_pred'],
                'within_t': results.tvalues['within_pred'],
                'within_p': results.pvalues['within_pred'],
                'between_coef': results.params['between_pred'],
                'between_se': results.bse['between_pred'],
                'between_t': results.tvalues['between_pred'],
                'between_p': results.pvalues['between_pred'],
                'aic': results.aic,
                'bic': results.bic,
                'log_likelihood': results.llf,
                'converged': results.converged,
                'controls': ', '.join(controls) if controls else 'None'
            }
            
            # Add significance indicators
            result_dict['within_sig'] = self._get_significance_marker(result_dict['within_p'])
            result_dict['between_sig'] = self._get_significance_marker(result_dict['between_p'])
            
            # Add the results to the collection
            self.results.append(result_dict)
            
            # Log key results
            self.logger.info(f"Results for {dv} ~ {pred_raw} ({model_name}):")
            self.logger.info(f"  Within-person effect: b={result_dict['within_coef']:.4f}, p={result_dict['within_p']:.4f} {result_dict['within_sig']}")
            self.logger.info(f"  Between-person effect: b={result_dict['between_coef']:.4f}, p={result_dict['between_p']:.4f} {result_dict['between_sig']}")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error fitting model: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None
    
    def _calculate_within_between(self, data, var):
        """Calculate within-person and between-person components of a variable.
        
        Args:
            data (pd.DataFrame): Data frame with observations
            var (str): Variable name
            
        Returns:
            tuple: (within_var, between_var) containing components
        """
        # Group by participant
        grouped = data.groupby('participant_id_clean')[var]
        
        # Calculate person means
        person_means = grouped.transform('mean')
        
        # Calculate within-person deviation (person-centered)
        within_var = data[var].values - person_means.values
        
        # Between-person component is simply the person mean
        between_var = person_means.values
        
        return within_var, between_var
    
    def _get_significance_marker(self, p_value):
        """Get significance marker based on p-value.
        
        Args:
            p_value (float): P-value to check
            
        Returns:
            str: Significance marker (*, **, ***, or empty string)
        """
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""
    
    def save_results(self):
        """Save results to files."""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        try:
            # Create a timestamp for file names
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create directory for results if it doesn't exist
            results_dir = self.output_dir
            
            # Create a DataFrame from results
            results_df = pd.DataFrame(self.results)
            
            # Round numeric columns for readability
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].round(4)
            
            # Save detailed results to Excel
            excel_path = results_dir / f'multilevel_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(excel_path) as writer:
                # All results
                results_df.to_excel(writer, sheet_name='All Models', index=False)
                
                # Create sheets for different dependent variables
                for dv in results_df['dependent_var'].unique():
                    # Replace underscores with hyphens for sheet name if this was a remapped column
                    display_dv = dv
                    for orig, remapped in self.col_name_map.items():
                        if remapped == dv:
                            display_dv = orig
                    
                    dv_results = results_df[results_df['dependent_var'] == dv]
                    sheet_name = f"{display_dv}"[:31]  # Excel sheet name length limit
                    dv_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create summary sheet with only significant results
                sig_results = results_df[(results_df['within_p'] < 0.05) | (results_df['between_p'] < 0.05)]
                if not sig_results.empty:
                    sig_results.to_excel(writer, sheet_name='Significant Results', index=False)
            
            self.logger.info(f"Saved detailed results to {excel_path}")
            
            # Save a CSV with just the essential results
            csv_path = results_dir / f'multilevel_results_{timestamp}.csv'
            results_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV results to {csv_path}")
            
            # Create sheets for STAI and CESD separately
            for emotion_type in ['STAI', 'CES']:
                emotion_results = results_df[results_df['dependent_var'].str.contains(emotion_type)]
                if not emotion_results.empty:
                    emotion_path = results_dir / f'{emotion_type.lower()}_results_{timestamp}.xlsx'
                    with pd.ExcelWriter(emotion_path) as writer:
                        emotion_results.to_excel(writer, sheet_name=f'All {emotion_type}', index=False)
                        
                        # Add significant results
                        sig_emotion = emotion_results[(emotion_results['within_p'] < 0.05) | 
                                                      (emotion_results['between_p'] < 0.05)]
                        if not sig_emotion.empty:
                            sig_emotion.to_excel(writer, sheet_name='Significant Results', index=False)
                    
                    self.logger.info(f"Saved {emotion_type} results to {emotion_path}")
                
            return excel_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run the multilevel analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run multilevel regression analysis on SURREAL data')
    
    parser.add_argument('--input_file', type=str, 
                        default='/Users/noamgal/DSProjects/Fragmentation/SURREAL/processed/merged_data/ema_fragmentation_daily_demographics.csv',
                        help='Path to merged data file')
    
    parser.add_argument('--output_dir', type=str,
                        default='/Users/noamgal/DSProjects/Fragmentation/SURREAL/results/multilevel_analysis',
                        help='Directory to save analysis results')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Run analysis
    try:
        analysis = MultilevelAnalysis(
            input_path=args.input_file,
            output_dir=args.output_dir,
            debug=args.debug
        )
        
        # Load and preprocess data
        df = analysis.load_and_preprocess_data()
        
        if df is None or df.empty:
            print("Error: Failed to load or preprocess data")
            return 1
        
        # Run the multilevel analysis
        results = analysis.run_multilevel_analysis()
        
        # Save results
        analysis.save_results()
        
        print("Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())