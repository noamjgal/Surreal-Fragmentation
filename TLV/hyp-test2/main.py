# main.py
import pandas as pd
import logging
import os
from config import Config
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from digital_usage_processor import DigitalUsageProcessor
from tests import StatisticalAnalyzer, AnalysisRunner

def setup_logging(config):
    """Configure logging with output directory"""
    log_file = os.path.join(config.output_dir, 'analysis.log')
    os.makedirs(config.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_data(data, config):
    """Extended data validation including column checks"""
    logging.info("\nValidating data structure...")
    logging.info(f"Available columns: {sorted(data.columns.tolist())}")
    
    # Split population factors into required now vs required later
    immediate_required = [col for col in config.population_factors 
                         if col != 'digital_usage_group']  # This gets added later
    
    # Check for immediately required columns
    immediate_required_columns = (
        config.frag_indices + 
        config.emotional_outcomes + 
        immediate_required +
        config.control_variables
    )
    
    missing_columns = [col for col in immediate_required_columns 
                      if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Basic validation
    min_participants = 3
    min_observations = 30
    
    n_participants = len(data['participant_id'].unique())
    if n_participants < min_participants:
        logging.warning(f"Insufficient participants: {n_participants}")
        return False
        
    if len(data) < min_observations:
        logging.warning(f"Insufficient observations: {len(data)}")
        return False
    
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"Number of participants: {n_participants}")
    logging.info(f"Average observations per participant: {len(data)/n_participants:.1f}")
    
    return True

def main():
    # Setup
    config = Config()
    setup_logging(config)
    logging.info("Starting analysis pipeline")
    
    try:
        # Initialize components
        loader = DataLoader(config)
        preprocessor = DataPreprocessor()
        digital_processor = DigitalUsageProcessor(config.output_dir)
        
        # Load and validate data
        logging.info("Loading data")
        raw_data = loader.load_data()
        logging.info("\nRaw data columns:")
        logging.info(sorted(raw_data.columns.tolist()))
        # Change the order in main() function
        logging.info("\nPreprocessing data")
        processed_data = preprocessor.preprocess(raw_data)

        # First process digital usage metrics
        user_metrics = digital_processor.calculate_usage_metrics(processed_data)
        processed_data = digital_processor.add_usage_metrics(processed_data, user_metrics)

        # Then validate the complete dataset
        if not validate_data(processed_data, config):
            raise ValueError("Data validation failed")

        # Initialize analysis components
        analyzer = StatisticalAnalyzer(config)
        runner = AnalysisRunner(analyzer)
        
        # Run analyses
        logging.info("\nRunning statistical analyses")
        results_df = runner.run_analyses(
            data=processed_data,
            predictors=config.frag_indices + config.population_factors,
            outcomes=config.emotional_outcomes,
            control_vars=config.control_variables
        )
        
        # Save all results
        output_file = os.path.join(config.output_dir, 'analysis_results.csv')
        results_df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
        # Save significant results separately
        sig_results = results_df[results_df['p_value'] < 0.05].copy()
        if len(sig_results) > 0:
            sig_file = os.path.join(config.output_dir, 'significant_results.csv')
            sig_results.to_csv(sig_file, index=False)
            
            summary_cols = ['test_type', 'predictor', 'outcome', 
                          'coefficient', 'p_value', 'effect_size']
            logging.info("\nSignificant Findings:")
            logging.info(sig_results[summary_cols].to_string())
            
            logging.info(f"\nTotal tests run: {len(results_df)}")
            logging.info(f"Significant results: {len(sig_results)}")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()