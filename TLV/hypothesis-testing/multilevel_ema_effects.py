import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import logging
from datetime import datetime

class FragmentationAnalysis:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.results = []
        
    def _setup_logging(self):
        log_path = self.output_dir / 'analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess_data(self):
        """Load and preprocess data with comprehensive metrics"""
        self.logger.info(f"Loading data from {self.input_path}")
        df = pd.read_csv(self.input_path)
        print(df.columns)  # Keep this for debugging
        
        # Log initial data shape
        self.logger.info(f"Initial data shape: {df.shape}")
        
        # Define metric groups for analysis
        self.fragmentation_metrics = {
            'raw': [
                'digital_fragmentation_index',
                'moving_fragmentation_index',
                'digital_frag_during_mobility'
            ],
            'zscore': [
                'digital_fragmentation_index_zscore',
                'moving_fragmentation_index_zscore',
                'digital_frag_during_mobility_zscore'
            ]
        }
        
        self.episode_metrics = [
            'digital_episode_count',
            'moving_episode_count',
            'digital_num_episodes',
            'moving_num_episodes',
            'overlap_num_episodes'
        ]
        
        self.duration_metrics = [
            'digital_total_duration',
            'moving_total_duration',
            'digital_total_duration_minutes',
            'moving_total_duration_minutes',
            'overlap_total_duration_minutes'
        ]
        
        # Define outcome variables
        self.emotion_components = [
            'TENSE',
            'RELAXATION_R',  # Note: Using reversed score directly
            'WORRY',
            'PEACE_R',      # Note: Using reversed score directly
            'IRRITATION',
            'SATISFACTION_R', # Note: Using reversed score directly
            'STAI6_score',
            'HAPPY'
        ]
        
        # Verify all required columns exist
        missing_cols = [col for col in self.emotion_components if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate z-scores for fragmentation metrics
        for metric in self.fragmentation_metrics['raw']:
            if metric in df.columns:
                df[f'{metric}_zscore'] = stats.zscore(df[metric], nan_policy='omit')
        
        self.data = df
        return df

    def run_comprehensive_analysis(self):
        """Run multiple analyses exploring different relationships"""
        
        # Model configurations
        control_steps = [
            [],  # Base model
            ['digital_total_duration'],  # Step 1
            ['digital_total_duration', 'moving_total_duration'],  # Step 2
            ['digital_total_duration', 'moving_total_duration', 'is_weekend']  # Step 3
        ]
        
        # 1. Primary Analysis: Fragmentation Effects on Emotions
        for metric_raw, metric_z in zip(
            self.fragmentation_metrics['raw'],
            self.fragmentation_metrics['zscore']
        ):
            for emotion in self.emotion_components:
                self.logger.info(f"\nAnalyzing {emotion} ~ {metric_raw}")
                
                for step, controls in enumerate(control_steps):
                    try:
                        # Use both raw and z-scored versions
                        self._run_single_model(
                            dv=emotion,
                            pred_raw=metric_raw,
                            pred_z=metric_z,
                            controls=controls,
                            model_name=f"Step {step}"
                        )
                    except Exception as e:
                        self.logger.error(f"Error in model: {str(e)}")
        
        # 2. Secondary Analysis: Episode Counts and Duration Effects
        for metric in self.episode_metrics + self.duration_metrics:
            for emotion in self.emotion_components:
                self.logger.info(f"\nAnalyzing {emotion} ~ {metric}")
                try:
                    self._run_single_model(
                        dv=emotion,
                        pred_raw=metric,
                        controls=['is_weekend'],
                        model_name="Episode/Duration Analysis"
                    )
                except Exception as e:
                    self.logger.error(f"Error in episode/duration model: {str(e)}")

    def _run_single_model(self, dv, pred_raw, pred_z=None, controls=None, model_name=""):
        """Run a single multilevel model with given parameters"""
        if controls is None:
            controls = []
            
        # Prepare data
        model_vars = [dv, pred_raw] + controls + ['user']
        current_data = self.data[model_vars].dropna()
        
        # FIXED: Use raw predictor for within/between calculations
        # Don't try to use z-score version as it might not exist
        within_pred, between_pred = self.prepare_within_between_data(
            current_data, pred_raw
        )
        
        current_data['within_pred'] = within_pred
        current_data['between_pred'] = between_pred
        
        # Prepare control variables
        control_terms = []
        for control in controls:
            within_control, between_control = self.prepare_within_between_data(
                current_data, control
            )
            current_data[f'within_{control}'] = within_control
            current_data[f'between_{control}'] = between_control
            control_terms.extend([f'within_{control}', f'between_{control}'])
        
        # Build formula
        formula_parts = ['within_pred', 'between_pred'] + control_terms
        formula = f"{dv} ~ {' + '.join(formula_parts)} + (1|user)"
        
        # Log model information for debugging
        self.logger.info(f"Fitting model for {dv} ~ {pred_raw}")
        self.logger.info(f"Formula: {formula}")
        self.logger.info(f"Data shape: {current_data.shape}")
        
        # Fit model with more robust settings
        try:
            # Add check for variance in random effects
            group_means = current_data.groupby('user')[dv].mean()
            group_var = group_means.var()
            if group_var < 1e-6:  # threshold for meaningful variance
                self.logger.warning(f"Very low between-person variance ({group_var:.6f}) for {dv}")
            
            # Add correlation check between predictors
            if controls:
                pred_vars = [pred_raw] + controls
                corr_matrix = current_data[pred_vars].corr()
                high_corr = np.where(np.abs(corr_matrix) > 0.7)
                if len(high_corr[0]) > len(pred_vars):  # more correlations than just diagonal
                    self.logger.warning(f"High correlations detected between predictors")
                    for i, j in zip(*high_corr):
                        if i < j:  # avoid printing both (i,j) and (j,i)
                            self.logger.warning(f"Correlation between {pred_vars[i]} and {pred_vars[j]}: {corr_matrix.iloc[i,j]:.3f}")
            
            model = smf.mixedlm(formula, data=current_data, groups='user')
            results = model.fit(reml=True, method=['lbfgs', 'nm', 'cg'], maxiter=1000)
            
            # Check convergence
            if not results.converged:
                self.logger.warning(f"Model did not converge for {dv} ~ {pred_raw}")
                return
            
            # Verify model results are valid
            if np.isnan(results.llf) or np.isinf(results.llf):
                self.logger.warning(f"Invalid log-likelihood for {dv} ~ {pred_raw}")
                return
                
            # Calculate AIC and BIC manually if needed
            n = len(current_data)
            k = len(results.params)  # number of parameters
            llf = results.llf  # log-likelihood
            
            aic = 2 * k - 2 * llf
            bic = k * np.log(n) - 2 * llf
            
            # Log detailed model diagnostics
            self.logger.info(f"""
            Model diagnostics for {dv} ~ {pred_raw}:
            - Log-likelihood: {llf:.2f}
            - Number of parameters: {k}
            - Sample size: {n}
            - AIC (calculated): {aic:.2f}
            - BIC (calculated): {bic:.2f}
            - Converged: {results.converged}
            """)
            
            # Store results
            result_dict = {
                'dependent_var': dv,
                'predictor': pred_raw,
                'model': model_name,
                'n_observations': n,
                'n_participants': current_data['user'].nunique(),
                'within_coef': results.params['within_pred'],
                'within_se': results.bse['within_pred'],
                'within_p': results.pvalues['within_pred'],
                'between_coef': results.params['between_pred'],
                'between_se': results.bse['between_pred'],
                'between_p': results.pvalues['between_pred'],
                'aic': float(aic),  # Use manually calculated AIC
                'bic': float(bic),  # Use manually calculated BIC
                'converged': results.converged,
                'controls': ', '.join(controls) if controls else 'None'
            }
            
            self.results.append(result_dict)
            
            # Add relative model comparison metrics
            if hasattr(self, 'previous_aic'):
                delta_aic = aic - self.previous_aic
                self.logger.info(f"Change in AIC from previous model: {delta_aic:.2f}")
            self.previous_aic = aic
            
        except Exception as e:
            self.logger.error(f"Model fitting failed: {str(e)}")
            self.logger.error(f"Variables in data: {current_data.columns.tolist()}")

    def prepare_within_between_data(self, data, var):
        """Prepare within and between person components"""
        participant_means = data.groupby('user')[var].transform('mean')
        within_var = data[var] - participant_means
        between_var = participant_means - participant_means.mean()
        return within_var, between_var

    def save_results(self):
        """Save comprehensive analysis results"""
        if not self.results:
            self.logger.warning("No results to save")
            return
            
        results_df = pd.DataFrame(self.results)
        
        # Round numeric columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        results_df[numeric_cols] = results_df[numeric_cols].round(4)
        
        # Add significance indicators
        for p_col in [col for col in results_df.columns if '_p' in col]:
            results_df[f'{p_col}_sig'] = results_df[p_col].apply(
                lambda x: '***' if x < 0.001 else 
                         '**' if x < 0.01 else 
                         '*' if x < 0.05 else ''
            )
        
        # Save results with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f'comprehensive_results_{timestamp}.xlsx'
        
        with pd.ExcelWriter(output_path) as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Create summary sheets by analysis type
            for analysis_type in ['Fragmentation', 'Episodes', 'Duration']:
                mask = results_df['model'].str.contains(analysis_type, case=False, na=False)
                if mask.any():
                    subset = results_df[mask]
                    subset.to_excel(writer, sheet_name=f'{analysis_type} Summary', index=False)
            
            # Create emotion component summary
            emotion_summary = results_df.pivot_table(
                index=['predictor', 'model'],
                columns='dependent_var',
                values=['within_coef', 'within_p_sig'],
                aggfunc='first'
            )
            emotion_summary.to_excel(writer, sheet_name='Emotion Components')
        
        self.logger.info(f"Comprehensive results saved to {output_path}")

def main():
    
    input_path = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/metrics/combined_metrics.csv'
    output_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/analysis_results'
    
    
    analyzer = FragmentationAnalysis(input_path, output_dir)
    df = analyzer.load_and_preprocess_data()
    analyzer.run_comprehensive_analysis()
    analyzer.save_results()
    
    print("Comprehensive analysis completed successfully")

if __name__ == "__main__":
    main()