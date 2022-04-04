# -*- coding: utf-8 -*-
"""
Created on Tues May 18 11:48:30 2021

@author: Bronte Mckeown

This script creates wordclouds for CAP neurosynth terms. 
"""

import os
import pandas as pd
import numpy as np
from collections import OrderedDict

from wordcloud import WordCloud
import matplotlib.cm as cm
import matplotlib.colors as mcolor

# %% Wordcloud function (adapted from pca_baby)

def wordclouder(neurosynth_dict, display,key, savefile=False):
    """
    Function to return 1) wordclouds.pngs (saved by default) 2) .csvs containg colour codes & weightings used to make wordclouds 
    """
     # Loop over loading dictionaries - 1 dataframe per iteration
    df = pd.DataFrame(neurosynth_dict) 
    df=(df-df.min())/(df.max()-df.min())
    principle_vector = np.array(df, dtype =float) # turn df into array
    pv_in_hex= []
    vmax = np.abs(principle_vector).max() #get the maximum absolute value in array
    vmin = -vmax #minimu 
    for i in range(principle_vector.shape[1]): # loop through each column (cap)
        rescale = (principle_vector  [:,i] - vmin) / (vmax - vmin) # rescale scores 
        colors_hex = []
        for c in cm.RdBu_r(rescale): 
            colors_hex.append(mcolor.to_hex(c)) # adds colour codes (hex) to list
        pv_in_hex.append(colors_hex) # add all colour codes for each item on all caps 
    colors_hex = np.array(pv_in_hex ).T 
    df_v_color = pd.DataFrame(colors_hex)
    if savefile:
        df_v_color.to_csv("neurosynth_colour_codes_for_wordclouds.csv", index = False, header = False)
    else:
        pass

    # loops over loadings for each cap
    for col_index in df:
        absolute = df[col_index].abs() # make absolute 
        integer = 100 * absolute # make interger 
        integer = integer.astype(int) 
        concat = pd.concat([integer, df_v_color[col_index]], axis=1) # concatanate loadings and colours 
        concat.columns = ['freq', 'colours']
        concat.insert(1, 'labels', display) # add labels (items) from display df
        if savefile:
            concat.to_csv("loadings_and_colour_codes_{}.csv".format(col_index+5), index = False, header = True)
        else:
            pass

        freq_dict = dict(zip(concat.labels, concat.freq)) # where key: item and value: weighting
        colour_dict = dict(zip(concat.labels, concat.colours))# where key: itemm and value: colour
        def color_func(word, *args, **kwargs): #colour function to supply to wordcloud function.. don't ask !
            try:
                color = colour_dict[word]
            except KeyError:
                color = '#000000' # black
            return color
        # create wordcloud object
        wc = WordCloud(background_color="white", color_func=color_func, 
                    width=400, height=400, prefer_horizontal=1, 
                    min_font_size=8, max_font_size=200
                    )
        # generate wordcloud from loadings in frequency dict
        wc = wc.generate_from_frequencies(freq_dict)
        wc.to_file('wordcloud_caps_component_{}.png'.format(key))
            

# %% Read in data
# read in neurosynth term loading data for caps 5 & 6
if __name__ == "__main__":
    folder_path = "C:\\Users\\bront\\OneDrive\\Documents\\PhD\\Projects\\caps\\"
    data_name = "results\\neurosynth\\neurosynth_loadings_caps_5_6_for_wordcloud_script.csv"
    data_path = folder_path + data_name
    df = pd.read_csv(data_path, header=None)

    # read in display labels
    label_path = folder_path + "results\\neurosynth\\neurosynth_terms_caps_5_6_for_wordcloud_script.csv"
    display = pd.read_csv(label_path, header=None)


    # transform to dictionary for word cloud function 
    neurosynth_dict = OrderedDict()
    neurosynth_dict['neurosynth'] = df

    # change directory for saving results
    os.chdir("C:\\Users\\bront\\OneDrive\\Documents\\PhD\\Projects\\caps\\results\\wordclouds")

    # call word cloud function 
    wordclouder(neurosynth_dict, display, savefile=False)
