# main.py
from config import Config
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from statistical_tests import StatisticalAnalyzer
from results_manager import ResultsManager
from analysis_runner import AnalysisRunner
from digital_usage_processor import DigitalUsageProcessor
import pandas as pd
import logging

def setup_logging():
    """Configure logging for the analysis pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )

def validate_data(data):
    """
    Validate input data meets minimum requirements
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data to validate
    
    Returns:
    --------
    bool
        True if data is valid, False otherwise
    """
    min_participants = 3
    min_observations = 30
    
    if len(data['participant_id'].unique()) < min_participants:
        logging.warning(f"Insufficient participants: {len(data['participant_id'].unique())}")
        return False
    if len(data) < min_observations:
        logging.warning(f"Insufficient observations: {len(data)}")
        return False
    return True

def main():
    """
    Main analysis pipeline
    
    Pipeline Steps:
    1. Initialize components and logging
    2. Load raw data
    3. Preprocess data
    4. Add digital usage metrics
    5. Run analyses:
       - Mobility analysis
       - Emotional analysis
       - Population analysis
       - Usage group analysis
    6. Save results
    
    Error Handling:
    - Validates data at each step
    - Logs errors and warnings
    - Gracefully handles missing data
    """
    # 1. Setup
    setup_logging()
    logging.info("Starting analysis pipeline")
    
    try:
        # Initialize components
        config = Config()
        loader = DataLoader(config)
        preprocessor = DataPreprocessor()
        analyzer = StatisticalAnalyzer(config)
        results_manager = ResultsManager(config)
        digital_processor = DigitalUsageProcessor()
        runner = AnalysisRunner(config, analyzer, results_manager)
        
        # 2. Load Data
        logging.info("Loading raw data")
        raw_data = loader.load_data()
        if not validate_data(raw_data):
            raise ValueError("Data validation failed")
            
        # 3. Preprocess
        logging.info("Preprocessing data")
        processed_data = preprocessor.preprocess(raw_data)
        
        # 4. Digital Usage Processing
        logging.info("Processing digital usage metrics")
        try:
            user_avg_usage = digital_processor.calculate_usage_metrics(processed_data)
            processed_data = digital_processor.add_usage_metrics(processed_data, user_avg_usage)
        except Exception as e:
            logging.error(f"Error in digital usage processing: {str(e)}")
            raise
        
        # 5. Run Analyses
        analysis_functions = {
            'mobility': runner.run_mobility_analysis,
            'emotional': runner.run_emotional_analysis,
            'population': runner.run_population_analysis,
            'usage_group': runner.run_usage_group_analysis
        }
        
        all_results = {}
        for analysis_type, analysis_func in analysis_functions.items():
            logging.info(f"Running {analysis_type} analysis")
            try:
                results_df = analysis_func(processed_data)
                all_results[analysis_type] = results_df
                
                # Save results by analysis type
                if 'type' in results_df.columns:
                    for test_type in results_df['type'].unique():
                        test_results = results_df[results_df['type'] == test_type]
                        results_manager.save_results(
                            test_results,
                            f'{analysis_type}_{test_type}_analysis.csv'
                        )
                else:
                    # Handle cases where 'type' column doesn't exist
                    results_manager.save_results(
                        results_df,
                        f'{analysis_type}_analysis.csv'
                    )
                
                # Save significant findings
                if 'p_value' in results_df.columns:
                    significant_results = results_df[results_df['p_value'] < 0.05]
                    results_manager.save_results(
                        significant_results,
                        f'{analysis_type}_significant_findings.csv'
                    )
            
            except Exception as e:
                logging.error(f"Error in {analysis_type} analysis: {str(e)}")
                continue
        
        # 6. Generate Summary
        logging.info("Generating analysis summary")
        generate_summary(all_results, config)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

def generate_summary(results_dict, config):
    """
    Generate summary of analysis results
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results DataFrames
    config : Config
        Configuration object
    """
    summary = []
    for analysis_type, results_df in results_dict.items():
        if 'p_value' in results_df.columns:
            sig_count = len(results_df[results_df['p_value'] < 0.05])
            summary.append({
                'analysis_type': analysis_type,
                'total_tests': len(results_df),
                'significant_results': sig_count,
                'significance_rate': sig_count / len(results_df) if len(results_df) > 0 else 0
            })
    
    summary_df = pd.DataFrame(summary)
    results_manager = ResultsManager(config)
    results_manager.save_results(summary_df, 'analysis_summary.csv')

if __name__ == "__main__":
    main()