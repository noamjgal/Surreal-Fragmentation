# config.py
class Config:
    def __init__(self):
        self.input_dir = '/Users/noamgal/Downloads/Research-Projects/SURREAL/Amnon'
        self.output_dir = self.input_dir + '/output'
        self.survey_file = self.input_dir + '/Survey/End_of_the_day_questionnaire.xlsx'
        self.participant_info_file = self.input_dir + '/participant_info.csv'
        self.frag_indices = ['digital_fragmentation_index', 'moving_fragmentation_index', 'digital_frag_during_mobility']
        self.emotional_outcomes = ['TENSE', 'RELAXATION', 'WORRY', 'PEACE', 'IRRITATION', 'SATISFACTION', 'STAI6_score', 'HAPPY']
        self.population_factors = ['Gender', 'Class', 'School']
