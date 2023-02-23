import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import ruptures as rpt
import os

data_path = 'drug_free_growth.xlsx'
save_folder = 'test' # place to save figures for spot-checking

def parse_data_file(df):
    """Strips metadata from raw data file to obtain timeseries OD data

    Args:
        p (str): path to data file

    Returns:
        pandas dataframe: dataframe of raw data
    """

    # get the first column (leftmost) of the data
    # cycle nr is always in the leftmost column
    time_col = df[df.keys()[0]]
    time_array = np.array(time_col)
        
    # raw data starts after cycle nr.
    if any(time_array == 'Cycle Nr.'):
        data_start_indx = np.argwhere(time_array == 'Cycle Nr.')
    elif any(time_array == 'Time [s]'):
        data_start_indx = np.argwhere(time_array == 'Time [s]')
    else:
        raise Exception('Unknown file format. Expected either Cycle Nr. or Time [s] as column headings.')

    #sometimes the data gets passed in very unraw
    if len(data_start_indx) == 0:
        return df
    
    data_start_indx = data_start_indx[0][0] # get scalar from numpy array

    # filter header from raw data file
    df_filt = df.loc[data_start_indx:,:]

    # change the column names
    df_filt.columns = df_filt.iloc[0] # get the top row from the dataframe and make it the column names
    df_filt = df_filt.drop(df_filt.index[0]) # remove the top row from the dataframe

    # find data end and filter empty cells
    first_col = df_filt[df_filt.keys()[0]] # get the first column of data
    x = pd.isna(first_col) # find where na occurs (empty cell)
    data_end_indx = x[x].index[0] 

    df_filt = df_filt.loc[:data_end_indx-1,:] # filter out the empty cells

    return df_filt

def est_growth_rates(data_path,prev_action,save_folder) -> dict:
    """Takes in optical density data and returns growth rates

    Time series optical density data in a 96-well plate -> a dict of output
    arrays. The output array for a given well is an array of size 16 with the 
    first 15 entries representing the drug action of the previous day and the 
    last entry representing the estimated growth rate.

    Also generates a figure for each curve fit and saves the figures to a
    folder.

    Args:
        data_path (str): .csv or .xlsx file with OD data
        prev_policy (dict): Dict of drugs used on the previous day for each well
        save_folder (str): Folder to save images of fits for spot-checking
    """

    df = pd.read_excel(data_path)

    data = parse_data_file(df)

    row_list = ['A','B','C','D','E','F','G','H']

    results = {}

    # Create the save folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for row in row_list:
        for col in range(1,13): # 12 columns

            fig,ax_list = plt.subplots(ncols=2,figsize=(8,3)) # figures for spot-checking

            key = row + str(col)

            OD = np.array(data[key]).astype(float)
            time = np.array(data['Time [s]']).astype(float)

            if OD[0] > OD[1]:
                OD = OD[1:]
                time = time[1:]


            lnOD = np.log(OD/np.min(OD)) # normalize OD data

            # Get the break points using pelt
            # jump, min_size, and pen are hyperparameters that may need tuning
            algo = rpt.Pelt(model='l2',jump=2,min_size=5).fit(lnOD)
            breaks = algo.predict(pen=0.1)

            # plot the break points
            break1 = time[[breaks[0],breaks[0]]]
            break2 = time[[breaks[1],breaks[1]]]
            y = [lnOD[0],np.max(lnOD)]

            ax = ax_list[0]

            ax.plot(time,lnOD,linewidth=2)
            ax.plot(break1,y,linewidth=2)
            ax.plot(break2,y,linewidth=2)

            # Get the linear segment of data determined by the break points
            lin_seg = lnOD[breaks[0]:breaks[1]]
            time_t = np.array(time[breaks[0]:breaks[1]])

            # linear regression to get the slope

            res = scipy.stats.linregress(time_t,lin_seg)

            fit = time_t*res.slope + res.intercept

            ax = ax_list[1]

            # plot the linear regression
            ax.plot(time_t,lin_seg,linewidth=2)
            ax.plot(time_t,fit,linewidth=2)

            # add the results to the results dict

            # output is an array representing the previous day's drug concatenated
            # with the growth rate
            output = np.zeros(16)
            output[prev_action[key]] = 1
            output[-1] = res.slope*1000

            results[key] = output

            ax.set_title(key)

            # save the figure for spot checking
            fig.savefig(save_folder + os.sep + key + '.png',bbox_inches='tight')
            plt.close(fig)
    
    return results