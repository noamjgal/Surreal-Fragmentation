import pandas as pd
import datetime as dt
import glob
import os


# file name of directory of participant data on your computer
directory = "participant-file-loc"
# file name of processed data on your computer
processed = "processed-target-file-loc" 

error = []
for participant in os.listdir(directory): # loop through all participant folders
    pID = participant.split("_")[1]
    if participant.startswith('.'):
        continue 
    try:     
        # access data files
        file = glob.glob(directory + participant + "/5 - HOBO Photometer/*.csv", recursive = True)
        print(pID)
        df = pd.read_csv(file[0], skiprows=[0])
        # convert to 24 hour time
        df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2]).dt.strftime('%H:%M:%S')
        try:
            # joins the date and time columns for days before clock change
            df['DATE TIME, GMT +03:00'] = df["Date"] + " " + df["Time, GMT+03:00"]
            del df["Time, GMT+03:00"]
        except:
            # adjusts time for daylight savings
            df['Time, GMT+02:00'] += pd.Timedelta(hours=1)
            df['Time, GMT+03:00'] = df['Time, GMT+02:00'].astype(str).map(lambda x: x[7:])
            del df["Time, GMT+02:00"]
            # joins date and time columns
            df['DATE TIME, GMT +03:00'] = df["Date"] + " " + df["Time, GMT+03:00"]            
            
            
        #df['DATE TIME, GMT +03:00'] = df["Date"] + " " + df["Time, GMT+03:00"]
        
        # remove redundant columns  
        del df["Date"]        
        del df["#"]
        # add column with participant id
        df.insert(0, "Participant ID", participant)
        df.to_csv(processed+ pID +"_5_Photometer_processed.csv", index=False)
                
    except:
        error.append(pID) 
        
print('The following participants had errors')
print(error)
