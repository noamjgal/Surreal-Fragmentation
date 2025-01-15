import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class StatisticalAnalyzer:
    def __init__(self, metrics_path: str, output_dir: str):
        """
        Initialize statistical analyzer
        
        Args:
            metrics_path: Path to combined metrics CSV
            output_dir: Directory for output files
        """
        self.metrics_path = Path(metrics_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging with both console and file output"""
        log_path = self.output_dir / 'statistical_analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare metrics data for analysis"""
        self.logger.info(f"Loading data from {self.metrics_path}")
        
        try:
            # Load data
            df = pd.read_csv(self.metrics_path)
            
            # Log initial data shape and columns
            self.logger.info(f"Loaded data shape: {df.shape}")
            self.logger.info("\nColumns in dataset:")
            self.logger.info("\n".join(f"- {col}" for col in df.columns))
            
            # Filter for rows with good data quality
            df = df[df['data_quality'] == 'good'].copy()
            
            # Ensure user column exists
            if 'user' not in df.columns and 'Participant_ID' in df.columns:
                df['user'] = df['Participant_ID']
            
            # Convert numeric columns
            numeric_cols = [
                'digital_fragmentation_index',
                'moving_fragmentation_index',
                'digital_frag_during_mobility',
                'digital_total_duration_minutes',
                'moving_total_duration_minutes',
                'STAI6_score',
                'HAPPY'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert boolean columns
            if 'is_weekend' in df.columns:
                df['is_weekend'] = df['is_weekend'].astype(int)
            
            # Log data summary after preparation
            self.logger.info("\nData after preparation:")
            self.logger.info(f"Number of participants: {df['user'].nunique()}")
            self.logger.info(f"Number of observations: {len(df)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def prepare_within_between_data(self, data: pd.DataFrame, var: str) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare within and between-person components for a variable
        
        Args:
            data: DataFrame containing the variable
            var: Name of the variable to decompose
            
        Returns:
            Tuple of (within-person, between-person) components
        """
        participant_means = data.groupby('user')[var].transform('mean')
        within_var = data[var] - participant_means
        between_var = participant_means - participant_means.mean()
        return within_var, between_var
    
    def run_multilevel_analysis(self, 
                              data: pd.DataFrame,
                              dependent_vars: List[str],
                              predictor_vars: List[str],
                              control_vars: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run multilevel analysis for specified variables
        """
        results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Check if data is empty
        if data.empty:
            self.logger.error("Input data is empty")
            return pd.DataFrame()
        
        # Rename participant_id to user for consistency
        data = data.copy()
        if 'participant_id' in data.columns:
            data['user'] = data['participant_id']
        
        for dv in dependent_vars:
            for pred in predictor_vars:
                self.logger.info(f"\nAnalyzing {dv} ~ {pred}")
                
                # Prepare model data
                model_vars = [dv, pred, 'user']
                if control_vars:
                    model_vars.extend(control_vars)
                
                model_data = data[model_vars].dropna()
                
                if len(model_data) == 0:
                    self.logger.warning(f"Insufficient data for {dv} ~ {pred}")
                    continue
                
                # Calculate within and between components
                for var in [pred] + (control_vars or []):
                    within, between = self.prepare_within_between_data(model_data, var)
                    model_data[f'{var}_within'] = within
                    model_data[f'{var}_between'] = between
                
                # Create model formula
                within_terms = [f'{var}_within' for var in [pred] + (control_vars or [])]
                between_terms = [f'{var}_between' for var in [pred] + (control_vars or [])]
                formula = f"{dv} ~ {' + '.join(within_terms + between_terms)} + (1|user)"
                
                try:
                    # Fit multilevel model
                    model = smf.mixedlm(formula, data=model_data, groups='user')
                    fit = model.fit()
                    
                    # Extract results
                    result = {
                        'dependent_var': dv,
                        'predictor_var': pred,
                        'n_observations': len(model_data),
                        'n_participants': model_data['user'].nunique(),
                        'aic': fit.aic,
                        'bic': fit.bic,
                        'within_coef': fit.params[f'{pred}_within'],
                        'within_se': fit.bse[f'{pred}_within'],
                        'within_p': 2 * (1 - stats.t.cdf(
                            abs(fit.params[f'{pred}_within'] / fit.bse[f'{pred}_within']),
                            df=len(model_data) - len(fit.params)
                        )),
                        'between_coef': fit.params[f'{pred}_between'],
                        'between_se': fit.bse[f'{pred}_between'],
                        'between_p': 2 * (1 - stats.t.cdf(
                            abs(fit.params[f'{pred}_between'] / fit.bse[f'{pred}_between']),
                            df=len(model_data) - len(fit.params)
                        ))
                    }
                    
                    # Add control variable information
                    if control_vars:
                        result['controls'] = ', '.join(control_vars)
                        
                        # Add control variable coefficients
                        for control in control_vars:
                            for level in ['within', 'between']:
                                var_name = f'{control}_{level}'
                                result[f'{control}_{level}_coef'] = fit.params[var_name]
                                result[f'{control}_{level}_p'] = 2 * (1 - stats.t.cdf(
                                    abs(fit.params[var_name] / fit.bse[var_name]),
                                    df=len(model_data) - len(fit.params)
                                ))
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error in analysis of {dv} ~ {pred}: {str(e)}")
        
        # Handle case where no results were generated
        if not results:
            self.logger.warning("No valid results were generated")
            empty_results = pd.DataFrame(columns=[
                'dependent_var', 'predictor_var', 'n_observations', 'n_participants',
                'aic', 'bic', 'within_coef', 'within_se', 'within_p',
                'between_coef', 'between_se', 'between_p'
            ])
            
            # Save empty results with summary
            output_file = self.output_dir / f'multilevel_results_{timestamp}.xlsx'
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                empty_results.to_excel(writer, sheet_name='Results', index=False)
                
                summary_stats = pd.DataFrame({
                    'Metric': ['Status', 'Total Observations', 'Variables Analyzed'],
                    'Value': [
                        'No valid results generated',
                        0,
                        f"DV: {', '.join(dependent_vars)}\nPred: {', '.join(predictor_vars)}"
                    ]
                })
                summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            
            self.logger.info(f"\nSaved empty results to {output_file}")
            return empty_results
        
        # If we have results, process them
        results_df = pd.DataFrame(results)
        
        # Save results to Excel with multiple sheets
        output_file = self.output_dir / f'multilevel_results_{timestamp}.xlsx'
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main results sheet
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Summary statistics sheet
            total_obs = results_df['n_observations'].sum() if 'n_observations' in results_df.columns else 0
            total_participants = results_df['n_participants'].mean() if 'n_participants' in results_df.columns else 0
            
            summary_stats = pd.DataFrame({
                'Metric': ['Total Observations', 'Total Participants', 'Variables Analyzed'],
                'Value': [
                    total_obs,
                    total_participants,
                    f"DV: {', '.join(dependent_vars)}\nPred: {', '.join(predictor_vars)}"
                ]
            })
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            
            # Significant findings sheet (only if we have results)
            if not results_df.empty:
                sig_results = results_df[
                    (results_df['within_p'] < 0.05) | 
                    (results_df['between_p'] < 0.05)
                ]
                if not sig_results.empty:
                    sig_results.to_excel(writer, sheet_name='Significant Findings', index=False)
        
        self.logger.info(f"\nSaved results to {output_file}")
        
        return results_df
    
    def generate_visualization(self, 
                             data: pd.DataFrame,
                             results_df: pd.DataFrame,
                             significant_only: bool = True):
        """Generate visualization of significant relationships"""
        if data.empty or results_df.empty:
            self.logger.warning("No data available for visualization")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Filter for significant results if requested
        if significant_only:
            sig_results = results_df[
                (results_df['within_p'] < 0.05) | 
                (results_df['between_p'] < 0.05)
            ]
            if sig_results.empty:
                self.logger.info("No significant relationships found for visualization")
                return
        else:
            sig_results = results_df
        
        for _, row in sig_results.iterrows():
            dv = row['dependent_var']
            pred = row['predictor_var']
            
            # Create within and between components for visualization
            model_data = data[[dv, pred, 'user']].dropna()
            if len(model_data) == 0:
                continue
            
            # Calculate within and between components
            within_var, between_var = self.prepare_within_between_data(model_data, pred)
            model_data[f'{pred}_within'] = within_var
            model_data[f'{pred}_between'] = between_var
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            try:
                # Within-person plot
                sns.regplot(
                    data=model_data,
                    x=f'{pred}_within',
                    y=dv,
                    ax=ax1,
                    scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'}
                )
                ax1.set_title(f'Within-person: {pred} → {dv}\n(p={row["within_p"]:.3f})')
                ax1.set_xlabel(f'{pred} (within-person)')
                ax1.set_ylabel(dv)
                
                # Between-person plot (using participant means)
                participant_means = model_data.groupby('user').mean()
                sns.regplot(
                    data=participant_means,
                    x=pred,
                    y=dv,
                    ax=ax2,
                    scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'blue'}
                )
                ax2.set_title(f'Between-person: {pred} → {dv}\n(p={row["between_p"]:.3f})')
                ax2.set_xlabel(f'{pred} (between-person)')
                ax2.set_ylabel(dv)
                
                # Adjust layout and save
                plt.tight_layout()
                fig_path = self.output_dir / f'relationship_{dv}_{pred}_{timestamp}.png'
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved visualization to {fig_path}")
                
            except Exception as e:
                self.logger.error(f"Error generating visualization for {dv} ~ {pred}: {str(e)}")
                plt.close()
                continue

def main():
    # Configure paths - update to use combined_metrics.csv
    metrics_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/metrics/combined_metrics.csv'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(metrics_path, output_dir)
    
    # Load and prepare data
    data = analyzer.load_and_prepare_data()
    
    # Define variables for analysis based on actual column names
    dependent_vars = [
        'STAI6_score', 'HAPPY'  # Starting with just these two as they're confirmed in the sample
    ]
    
    predictor_vars = [
        'digital_fragmentation_index',
        'moving_fragmentation_index',
        'digital_frag_during_mobility'
    ]
    
    control_vars = [
        'digital_total_duration_minutes',
        'moving_total_duration_minutes',
        'is_weekend'  # Adding this as a control variable
    ]
    
    # Run analysis
    results = analyzer.run_multilevel_analysis(
        data=data,
        dependent_vars=dependent_vars,
        predictor_vars=predictor_vars,
        control_vars=control_vars
    )
    
    # Generate visualizations
    analyzer.generate_visualization(data, results)
    
    print("Analysis completed successfully")

if __name__ == "__main__":
    main() 

