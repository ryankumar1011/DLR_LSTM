# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

# we focus on one specific file for EDA. Here we take a mixed dataset for Li-ion cell tested at 25DegC
file_path = 'dataset/25degC/552_Mixed3.csv'

# reading file won't work without skiprows parameter since the first 30 lines of the csv are invalid
df = pd.read_csv(file_path, skiprows=30)

# look at the values
print(df.head())

# we see that we will have an extra empty column because of trailing commas in CSV file
df.columns = ['Time Stamp','Step','Status','Prog Time','Step Time','Cycle',
              'Cycle Level','Procedure','Voltage','Current','Temperature','Capacity','WhAccu','Cnt', 'Empty']

# drop the empty column
df.drop('Empty', axis=1, inplace=True)

# change display settings so we can see all columns
pd.set_option('display.max_columns', None)

# there are no NULL values
print(df.head(), df.info(), df.describe().transpose(), df.isnull().sum())

# we see the Status values are all TABLE
print(df['Status'].value_counts())

# note:
# for the charging datasets STATUS has values PAU and CHA which means paused and charging
# for the discharging datasets STATUS is DCH which means discharging
# for the mixed datasets STATUS is 'TABLE'
# we are interested in the mixed datasets for SOH estimation (not charging and not constant fixed discharging)

# in dataframe Prog Time is a string with format HH:MM:SS.MS (like 02:14:39.314)
# we first convert the Prog time string to ints hours (h), minutes (m), seconds (s), milliseconds (ms)
# and then create a timedelta object that allows us to get time in seconds
def get_seconds(timestamp):
    temp = timestamp.split(':')
    h = int(temp[0])
    m = int(temp[1])
    temp = temp[2].split('.')
    s = int(temp[0])
    ms = int(temp[1])

    # we can convert time to second using timedelta
    return float(timedelta(hours=h, minutes=m, seconds=s, microseconds=ms*1000).total_seconds())

# we can now make column Elapsed Time which is time from start of experiment
# we do this by using timedelta method
def get_elapsed_time(series):
    timestamps = series.to_numpy()
    res= []
    start_time = get_seconds(timestamps[0])

    for time in timestamps:
        res.append(get_seconds(time) - start_time)

    return np.array(res)

# set the Elapsed Time column
df['Elapsed Time'] = get_elapsed_time(df['Prog Time'])

# we can now get SOC (the target) from capacity.
# in the dataset, capacity starts at 0 and decreases into the negative as the battery discharges
# since 0 is set as 100% SOC we shift capacity values by min capacity to make 0 the min
df['SOC Capacity'] = (df['Capacity'] - df['Capacity'].min())

# we find SOC percentage by comparing it to the max
df['SOC'] = df['SOC Capacity'] / df['SOC Capacity'].max()

# look at the new values
print(df.head())

# GRAPHING

# correlation_series tells us how much each variable is correlates to SOC (from 0-1 linear correlation)
correlation_series = df[['SOC', 'Elapsed Time', 'Voltage', 'Current', 'Temperature', 'WhAccu']].corr()['SOC']
correlation_series.drop('SOC')

# plot correlation_series as a barplot
sns.barplot(x=correlation_series.index, y=correlation_series.to_numpy())
plt.show()

# SOC against elapsed time
sns.lineplot(data=df, x='Elapsed Time', y='SOC')
plt.show()

# plot some dataframe variables against SOC (lineplot and scatterplot used)
# we = invert SOC axis by getting current axis and inverting x_axis (SOC goes from 1 to 0)
for feature in ['Temperature', 'Voltage', 'Current', 'WhAccu']:
    sns.lineplot(data=df, x='SOC', y=feature)
    plt.gca().invert_xaxis()

    plt.show()

    sns.scatterplot(data=df, x='SOC', y=feature)
    plt.gca().invert_xaxis()

    plt.show()

# plot the kernel density estimation (kde) plots for current and voltage so we can see their distribution
# of values
sns.kdeplot(data=df, x='Voltage')
plt.show()

sns.kdeplot(data=df, x='Current')
plt.show()
