#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:48:10 2024

@author: noamgal
"""

import pandas as pd
import glob
import os

# file name of directory of participant data on your computer
directory = "participant-file-loc"
# file name of processed data on your computer
processed = "processed-target-file-loc" 

# Initialize all_participants_data and error list
all_participants_data = []
error = []

for participant in os.listdir(directory):  # loop through all participant folders
    pID = participant.split("_")[1]
    
    if participant.startswith('.'):
        continue 
    try:  
        print(pID)
        # access data files
        files = glob.glob(directory + participant + "/7 - Dreem EEG/*.csv", recursive=True)
        print(files)
        
        participant_data = []
        for file in files:
            if "depreciated_report" not in file:
                # access data file
                source = pd.read_csv(file, header=None)
                
                # Transpose the dataframe
                transposed_df = source.transpose()
                
                # Set the first row as column headers
                transposed_df.columns = transposed_df.iloc[0]
                transposed_df = transposed_df.drop(transposed_df.index[0])
                
                # Reset index to start from 0
                transposed_df = transposed_df.reset_index(drop=True)
                
                # Add Participant_ID column
                transposed_df['Participant_ID'] = pID
                
                participant_data.append(transposed_df)
        
        # Combine all data for this participant
        if participant_data:
            participant_combined = pd.concat(participant_data, ignore_index=True)
            
            # Move Participant_ID to the first column
            cols = participant_combined.columns.tolist()
            cols = ['Participant_ID'] + [col for col in cols if col != 'Participant_ID']
            participant_combined = participant_combined[cols]
            
            # Deduplicate the data for this participant
            participant_combined = participant_combined.drop_duplicates(subset=['record'], keep='first')
            
            # Add to the list for all participants output
            all_participants_data.append(participant_combined)
            
            # Export processed file for individual participant
            output_filename = f"{pID}_7_DreemEEG_processed.csv"
            participant_combined.to_csv(os.path.join(processed, output_filename), index=False)
                
    except Exception as e:
        error.append(f"{pID}: {str(e)}")

# Combine all participants' data
if all_participants_data:
    combined_df = pd.concat(all_participants_data, ignore_index=True)
    
    # Deduplicate the data across all participants
    combined_df = combined_df.drop_duplicates(subset=['Participant_ID', 'record'], keep='first')
   
    # Export combined data
    combined_df.to_csv(os.path.join(processed, "All_Participants_7_DreemEEG_processed.csv"), index=False)

print("There were errors with the following participants:")
for err in error:
    print(err)
