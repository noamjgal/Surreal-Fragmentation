import pandas as pd
import glob
import os

directory = "Participants"

# file name of directory of participant data on your computer
directory = "participant-file-location"
# file name of processed data on your computer
processed = "processed-file-location" 

for participant in os.listdir(directory): # loop through all the participant folders
    # skips file that start with a . in order to avoid the hidden files such as .DS_Store
    if participant.startswith('.'):
        continue 
    try:    
        # access data file
        file = glob.glob(directory + participant +"/4 - PCE-HT 72 Hygrometer/*.CSV", recursive = True)
        df = pd.read_csv(file[0])
        # remove unnecessary column
        del df["Number"]
        # add column with participant id
        df.insert(0, "Participant ID", participant)
        df.to_csv(processed + participant +"_4_Hygrometer_processed.csv", index=False)
        
    except:
        print('Error for:', participant)       
