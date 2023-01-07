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

df = pd.read_excel(data_path)

data = parse_data_file(df)

row_list = ['A','B','C','D','E','F','G','H']

results = {}

# Create the save folder
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

for row in row_list:
    for col in range(1,13):

        fig,ax_list = plt.subplots(ncols=2,figsize=(8,3))

        key = row + str(col)

        OD = np.array(data[key]).astype(float)
        time = np.array(data['Time [s]']).astype(float)

        if OD[0] > OD[1]:
            OD = OD[1:]
            time = time[1:]


        lnOD = np.log(OD/np.min(OD))

        # Get the break points using pelt
        algo = rpt.Pelt(model='l1').fit(lnOD)
        breaks = algo.predict(pen=2)

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
        results[key] = res.slope

        ax.set_title(key)

        # save the figure for spot checking
        fig.savefig(save_folder + os.sep + key + '.png',bbox_inches='tight')
        plt.close(fig)