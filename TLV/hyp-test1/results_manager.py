import os
import pandas as pd

# results_manager.py
class ResultsManager:
    def __init__(self, config):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
    
    def save_results(self, results_dict, filename):
        output_path = os.path.join(self.config.output_dir, filename)
        
        if isinstance(results_dict, pd.DataFrame):
            results_dict.to_csv(output_path, index=False)
        else:
            pd.DataFrame(results_dict).to_csv(output_path, index=False)
            
        print(f"Results saved to: {output_path}")
