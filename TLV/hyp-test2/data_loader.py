# data_loader.py
import pandas as pd
import os
from config import Config
class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        frag_summary = pd.read_csv(os.path.join(self.config.input_dir, 'fragmentation/fragmentation_summary.csv'))
        survey_responses = pd.read_excel(self.config.survey_file)
        participant_info = pd.read_csv(self.config.participant_info_file)
        return self._merge_datasets(frag_summary, survey_responses, participant_info)
    
    def _merge_datasets(self, frag_summary, survey_responses, participant_info):
        frag_summary['date'] = pd.to_datetime(frag_summary['date']).dt.date
        survey_responses['date'] = pd.to_datetime(survey_responses['StartDate']).dt.date
        
        for df, id_col in [(frag_summary, 'participant_id'), 
                          (survey_responses, 'Participant_ID'),
                          (participant_info, 'user')]:
            df[id_col] = df[id_col].astype(str)
        
        merged = pd.merge(
            frag_summary, 
            survey_responses,
            left_on=['participant_id', 'date'],
            right_on=['Participant_ID', 'date'],
            how='inner'
        )
        
        merged = pd.merge(
            merged,
            participant_info,
            left_on='participant_id',
            right_on='user',
            how='left'
        )
        
        return merged
