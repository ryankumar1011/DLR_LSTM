# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# file 3
file_path = '/Users/ryankumar/Library/Application Support/JetBrains/PyCharm2024.3/light-edit/CALB_ds/Temperature_40C/59627/Driving_cycles/L148N58A_59627_CH3_DrivingCycle_T40_Channel_3_Wb_1.xlsx'

# file 2- '/Users/ryankumar/Library/Application Support/JetBrains/PyCharm2024.3/light-edit/CALB_ds/Temperature_25C/60710/Driving_cycles/L148N58A_60710_CH4_DrivingCycle_T25_Channel_4_Wb_1.xlsx'
# file 1 - '/Users/ryankumar/Library/Application Support/JetBrains/PyCharm2024.3/light-edit/CALB_ds/Temperature_25C/60644/Driving_cycles/L148N58A_60644_CH6_DrivingCycle_T25_Channel_6_Wb_1.xlsx'

df = pd.read_excel(file_path, sheet_name=1)

# change display settings so we can see all columns
pd.set_option('display.max_columns', None)

# check data
print (df.head(5))

# select columns of interest
df = df[['Date Time', 'Test Time (s)','Current (A)', 'Voltage (V)', 'Power (W)', 'Aux_Temperature_1 (C)', 'Aux_Temperature_2 (C)', 'Charge Capacity (Ah)', 'Discharge Capacity (Ah)']]

# change column names to match current naming
df.columns = ['Time Stamp', 'Elapsed Time', 'Current','Voltage','Power','Temperature1','Temperature2', 'Charge Capacity', 'Discharge Capacity']

# get temperature average
df['Temperature'] = (df['Temperature1'] + df['Temperature2']) / 2

# trim df by voltage and get SOC
df = df.iloc[df['Voltage'].idxmax() : df['Voltage'].idxmin()+1]
df['SOC'] = (df['Discharge Capacity'].max() - df['Discharge Capacity']) / df['Discharge Capacity'].max()

# check how capacity changed
sns.lineplot(x='Elapsed Time', y='Charge Capacity', data=df, label='Charge Capacity')
sns.lineplot(x='Elapsed Time', y='Discharge Capacity', data=df, label='Discharge Capacity')
plt.legend()
plt.ylabel('Capacity (Ah)')
plt.show()

# plot features against time
for feature in ['Temperature', 'Voltage', 'Current', 'Power', 'SOC']:
    sns.lineplot(data=df, x='Elapsed Time', y=feature)
    plt.show()

# Note: we can see that charge and discharge capacity are recorded cumulatively

# Note: power and current oscillate because of power recovery from braking

# In our power profile, would we ignore power recovery?
