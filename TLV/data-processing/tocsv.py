import pandas as pd
import os
from pathlib import Path
import logging
from typing import Dict, List
import time

class ExcelConverter:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize Excel to CSV converter
        
        Args:
            input_dir: Directory containing Excel files
            output_dir: Directory for output CSV files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def convert_gps_data(self, gps_file: str, sheet_name: str = 'gpsappS_8') -> str:
        """
        Convert GPS Excel file to CSV
        
        Args:
            gps_file: Name of GPS Excel file
            sheet_name: Name of sheet to convert
            
        Returns:
            Path to converted CSV file
        """
        start_time = time.time()
        self.logger.info(f"Converting GPS data from {gps_file}")
        
        input_path = self.input_dir / gps_file
        output_path = self.output_dir / f"{input_path.stem}.csv"
        
        if output_path.exists() and output_path.stat().st_mtime > input_path.stat().st_mtime:
            self.logger.info(f"CSV already up to date: {output_path}")
            return str(output_path)
            
        try:
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            df.to_csv(output_path, index=False)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Converted {len(df)} rows in {elapsed:.2f} seconds")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error converting {gps_file}: {str(e)}")
            raise

    def convert_ema_data(self, ema_file: str) -> str:
        """
        Convert EMA questionnaire Excel file to CSV
        
        Args:
            ema_file: Name of EMA Excel file
            
        Returns:
            Path to converted CSV file
        """
        start_time = time.time()
        self.logger.info(f"Converting EMA data from {ema_file}")
        
        input_path = self.input_dir / ema_file
        output_path = self.output_dir / f"{input_path.stem}.csv"
        
        if output_path.exists() and output_path.stat().st_mtime > input_path.stat().st_mtime:
            self.logger.info(f"CSV already up to date: {output_path}")
            return str(output_path)
            
        try:
            df = pd.read_excel(input_path)
            df.to_csv(output_path, index=False)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Converted {len(df)} rows in {elapsed:.2f} seconds")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error converting {ema_file}: {str(e)}")
            raise

    def get_converted_paths(self, 
                          gps_file: str, 
                          ema_file: str, 
                          sheet_name: str = 'gpsappS_8') -> Dict[str, str]:
        """
        Convert both files and return paths to CSVs
        
        Args:
            gps_file: Name of GPS Excel file
            ema_file: Name of EMA Excel file
            sheet_name: Name of sheet in GPS file
            
        Returns:
            Dictionary with paths to converted CSV files
        """
        gps_csv = self.convert_gps_data(gps_file, sheet_name)
        ema_csv = self.convert_ema_data(ema_file)
        
        return {
            'gps_path': gps_csv,
            'ema_path': ema_csv
        }

def main():
    # Define paths
    INPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey'
    OUTPUT_DIR = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon/Survey/csv'
    
    GPS_FILE = 'gpsappS_9.1_excel.xlsx'
    EMA_FILE = 'End_of_the_day_questionnaire.xlsx'
    
    # Initialize and run converter
    converter = ExcelConverter(INPUT_DIR, OUTPUT_DIR)
    
    # Convert files
    converted_paths = converter.get_converted_paths(GPS_FILE, EMA_FILE)
    
    print("\nConverted file paths:")
    for file_type, path in converted_paths.items():
        print(f"{file_type}: {path}")

if __name__ == "__main__":
    main()