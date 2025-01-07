import pandas as pd
import logging
from pathlib import Path
import sys

# Setup
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_variable_mappings():
    """Add Variable column to raw EMA data based on existing mappings."""
    try:
        # Load raw EMA data
        raw_data_path = Path(project_root) / "data" / "raw" / "comprehensive_ema_data.csv"
        raw_data = pd.read_csv(raw_data_path, sep='|', quotechar='"')
        logging.info(f"Loaded raw data: {len(raw_data)} rows")
        logging.info("Raw data columns:")
        logging.info(raw_data.columns)

        # Load mappings
        mapping_path = Path(project_root) / "data" / "raw" / "Corrected-Response-Mappings.xlsx"
        mapping_data = pd.read_excel(mapping_path)
        logging.info("Mapping data columns:")
        logging.info(mapping_data.columns)

        # Debug: Print first few rows of both dataframes
        logging.info("\nFirst few rows of raw data:")
        logging.info(raw_data.head())
        logging.info("\nFirst few rows of mapping data:")
        logging.info(mapping_data.head())

        # Create question to variable mapping with normalization
        question_to_variable = {}
        for _, row in mapping_data.iterrows():
            # Normalize Hebrew question by removing spaces and special characters
            hebrew_question = row['Question'].strip().replace('/', '').replace('?', '').replace('  ', ' ')
            variable = row['Variable']
            question_to_variable[hebrew_question] = variable
            
            # Also store a simplified version
            simple_question = ''.join(c for c in hebrew_question if c.isalnum())
            question_to_variable[simple_question] = variable

        # Debug mapping
        logging.info("\nMapping statistics:")
        logging.info(f"Total mappings created: {len(question_to_variable)}")
        
        # Find the question column
        question_col = 'Question name'  # Use explicit column name instead of searching
        if question_col not in raw_data.columns:
            raise ValueError(f"Could not find '{question_col}' in raw data columns: {raw_data.columns}")
        
        # Create normalized versions of the questions in raw data
        raw_data['Question_normalized'] = raw_data[question_col].apply(
            lambda x: x.strip().replace('/', '').replace('?', '').replace('  ', ' ') if isinstance(x, str) else x
        )
        raw_data['Question_simple'] = raw_data[question_col].apply(
            lambda x: ''.join(c for c in x if c.isalnum()) if isinstance(x, str) else x
        )
        
        # Try matching with both normalized and simplified versions
        raw_data['Variable'] = raw_data['Question_normalized'].map(question_to_variable)
        mask = raw_data['Variable'].isna()
        raw_data.loc[mask, 'Variable'] = raw_data.loc[mask, 'Question_simple'].map(question_to_variable)
        
        # Debug matching results
        logging.info("\nMatching Results:")
        logging.info(f"Total rows: {len(raw_data)}")
        logging.info(f"Matched variables: {raw_data['Variable'].notna().sum()}")
        logging.info(f"Unmatched rows: {raw_data['Variable'].isna().sum()}")
        
        # Check for unmapped questions with more detail
        unmapped = raw_data[raw_data['Variable'].isna()][question_col].unique()
        if len(unmapped) > 0:
            logging.warning(f"\nFound {len(unmapped)} unmapped questions:")
            for q in unmapped:
                # Show similar questions from mapping
                similar_qs = [mq for mq in mapping_data['Question'] if similar_text(q, mq) > 0.8]
                logging.warning(f"\nUnmapped: {q}")
                if similar_qs:
                    logging.warning(f"Similar questions in mapping:")
                    for sq in similar_qs:
                        logging.warning(f"  - {sq} -> {question_to_variable.get(sq)}")

        # Remove temporary columns and save
        raw_data = raw_data.drop(['Question_normalized', 'Question_simple'], axis=1)
        output_path = Path(project_root) / "data" / "raw" / "comprehensive_ema_data_var.csv"
        raw_data.to_csv(output_path, index=False)
        logging.info(f"\nSaved processed data to: {output_path}")
        
        return raw_data

    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise

def similar_text(a, b):
    """Calculate similarity ratio between two strings."""
    from difflib import SequenceMatcher
    if not isinstance(a, str) or not isinstance(b, str):
        return 0
    return SequenceMatcher(None, a, b).ratio()

if __name__ == "__main__":
    create_variable_mappings()