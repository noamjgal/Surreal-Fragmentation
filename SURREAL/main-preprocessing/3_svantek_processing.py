import pandas as pd
import os
import glob
import datetime as dt

# file name of directory of participant data on your computer
directory = "Paritipant-data-location"
# file name of processed data on your computer
processed = "target-processed-output-location" 




# lists for use in creating file of participant start/end times
p_IDs = []
start_times = []
end_times = []

errors = []
for participant in os.listdir(directory): #loop through all the participant folders
    
    # skips file that start with a . in order to avoid the hidden files such as .DS_Store
    if participant.startswith('.'):
        continue
    pID = participant.split("_")[1]

    try:   
    
        # access data files
        files = glob.glob(directory + "/Pilot_" + pID + '/3 - Svantek SV104/*.csv', recursive = True)
        
        big_df = pd.DataFrame()
        for file in files: # loop through data files
        
            print("reading" + pID)
            # read data in
            source = pd.read_csv(file, sep=';',skiprows=6, skipfooter=1)
    
            # reverse order of data by date
            df = source.iloc[::-1]
         
            
            # code to detect where the metadata ends in the time column
            # code assumes that time entries are all of length 18
            # when two sequential entries are length 18, this means the metadata is over and time data has begun
            count = 0
            a = False
            for t in df['Time']:
                if a:
                    if len(t) == 18:
                        break
                if len(t)==18:
                    a = True
                count += 1          
            
                        # code to save the metadata
            #meta_df = df.iloc[:count,0:2]
            # adjusts the dataframe to remove the metadata and overload column
            df = df.iloc[count:,:-4]
            
            # adjusts time for participant 1 which had an error
            if pID == '001': 
                df['Time'] = pd.to_datetime(df['Time']) + pd.Timedelta(days=1603, hours=17, minutes=1, seconds=0)   
                

            
            # concatenates all files for each participant
            big_df =  pd.concat([big_df, df], axis=0)



        # add participant start and end times to lists
        start_times.append(big_df.iloc[0,0])
        end_times.append(big_df.iloc[-1,0])
        p_IDs.append(participant) 
        print(big_df['Time'].head())
        # write data to new file
        big_df.to_csv(processed + "/" + pID + "_3_Svantek_processed.csv", index=False)
        print("done" + pID)   
        
    # reports participants with problems
    except:
        print('There was a problem with Participant: ',pID)
        errors.append(pID)
        continue
    
# create file for all participants start and end times
d = {'Participant IDs': p_IDs, 'Start Time': start_times, 'End Time': end_times}
start_end_df = pd.DataFrame(data=d)
start_end_df.to_csv(processed + "/participants_start_end.csv")

print('All files are processed except for the following participants with errors:', errors)

