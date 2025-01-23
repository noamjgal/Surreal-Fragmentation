import pandas as pd
import os
import glob

# file name of directory of participant data on your computer
directory = "participant-file-loc"
# file name of processed data on your computer
processed = "processed-target-file-loc"   
# error list
error = []

for participant in os.listdir(directory): # loop through all the participant folders
    # skips file that start with a . in order to avoid the hidden files such as .DS_Store
    if participant.startswith('.'):
        continue 
    pID = participant.split("_")[1] 
    result = pd.DataFrame()
    
    try:
        print(pID)
        # access data file
        files = glob.glob(directory + "/Pilot_" + pID + '/2 - wGT3X_Accelerometer/*', recursive = True)
        # Boolean to check if the bouts file is present
        empty = True
        # iterates through files and adds bouts files to df
        for file in files:
            print(file)
            if "bou" in file.lower():
                empty = False
                try:
                    df = pd.read_csv(file)
                except:
                    df = pd.read_excel(file)
                result = result._append(df)
        # reports error if no bouts file
        if empty:
            raise
        # add column with participant id
        result.insert(0, "Participant ID", participant)
        
    
        result.to_csv(processed + "/"+pID+"_2_Accelerometer_processed.csv", index=False)
    except:
        error.append(pID)
        
print("There were errors with the following participants:")
print(error)
