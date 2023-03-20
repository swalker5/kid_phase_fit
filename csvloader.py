'''
csvloader.py

SW 3/2023
'''


import math
import cmath
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


# change filename as needed
filename = 'drive_atten_toltec0_017068_test.csv'


# read in file as pandas DataFrame
df = pd.read_csv(filename, delimiter= ',')

# find nans in drive_atten column
loc_nan = np.isnan(df['drive_atten'])

# set nans to 0.
df.loc[loc_nan, 'drive_atten'] = 0.

# check that there are no more nans
loc_nan_check = np.isnan(df['drive_atten']) 
print('checking number of nans in drive_atten after setting to 0: ',\
 len(df.loc[loc_nan_check]))

# convert flags from one string to a list of strings
df['fit_success'] = df['fit_success'].str.strip('[]').str.split(",")
df['fit_flags'] = df['fit_flags'].str.strip('[]').str.split(",")

# find empty lists or failed fits
loc_empty = (df['fit_success'].str.len() == 1)

# find negative drive_atten values
loc_neg = df['drive_atten'] < 0.
# set negative drive_atten values to 0.
df.loc[loc_neg, 'drive_atten'] = 0.

# find drive_atten values equal to 0.
loc_zero = df['drive_atten'] == 0.
# find drive_atten values that are not 0. (want to use)
loc_notzero = df['drive_atten'] != 0.

# find mean, median, and std of drive_atten values not 0.
mean_drive_atten = np.mean(df['drive_atten'][loc_notzero])
median_drive_atten = np.median(df['drive_atten'][loc_notzero])
std_drive_atten = np.std(df['drive_atten'][loc_notzero])

# find drive_atten values greater than two times the median
loc_too_large = df['drive_atten'] > 2.*median_drive_atten

# set large drive_atten values (probably bad fits) to 0.
df.loc[loc_too_large, 'drive_atten'] = 0.

# find new list of drive_atten value equal to 0.
loc_zero_new = df['drive_atten'] == 0.
# find new list of drive_atten value greater than 0. (want to use)
loc_notzero_new = df['drive_atten'] != 0.

# find new mean, median, and std of drive_atten values not 0.
mean_drive_atten_new = np.mean(df['drive_atten'][loc_notzero_new])
median_drive_atten_new = np.median(df['drive_atten'][loc_notzero_new])
std_drive_atten_new = np.std(df['drive_atten'][loc_notzero_new])

# use median or mean drive_atten for global value
print('For global drive atten, the mean value is ' +\
 str(mean_drive_atten_new) + ' dBm')
print('For global drive atten, the median value is ' +\
 str(median_drive_atten_new) + ' dBm')
