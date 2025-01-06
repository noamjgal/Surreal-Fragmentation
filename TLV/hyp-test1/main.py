from config import Config
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from statistical_tests import StatisticalAnalyzer
from results_manager import ResultsManager
from analysis_runner import AnalysisRunner

# main.py
def main():
    # Initialize components
    config = Config()
    loader = DataLoader(config)
    preprocessor = DataPreprocessor()
    analyzer = StatisticalAnalyzer(config)
    results_manager = ResultsManager(config)
    
    # Initialize analysis runner
    runner = AnalysisRunner(config, analyzer, results_manager)
    
    # Load and preprocess data
    raw_data = loader.load_data()
    processed_data = preprocessor.preprocess(raw_data)
    
    # Set control variables based on available data columns
    analyzer.set_control_variables(processed_data)
    
    # Run analyses
    analysis_functions = {
        'mobility': runner.run_mobility_analysis,
        'emotional': runner.run_emotional_analysis,
        'population': runner.run_population_analysis
    }
    
    for analysis_type, analysis_func in analysis_functions.items():
        # Run analysis
        results_df = analysis_func(processed_data)
        
        # Split results by test type
        for test_type in results_df['type'].unique():
            test_results = results_df[results_df['type'] == test_type]
            results_manager.save_results(
                test_results,
                f'{analysis_type}_{test_type}_analysis.csv'
            )
        
        # Save significant findings
        significant_results = results_df[results_df['p_value'] < 0.05]
        results_manager.save_results(
            significant_results,
            f'{analysis_type}_significant_findings.csv'
        )

if __name__ == "__main__":
    main()