import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read files
df_power = pd.read_excel('/Users/yorkgong/Downloads/OPA_Power.xlsx')
df_temp = pd.read_excel('/Users/yorkgong/Downloads/OPA_Temp.xlsx')

#name each column and create structure
df_temp.columns = ['T', 'date', 'time', 'temp', 'Humid', 'TH1']
df_power.columns = ['sps', 'date', 'time', 'power']
df_temp1 = pd.DataFrame(df_temp)
df_power1 = pd.DataFrame(df_power)

#change datatype and check
df_temp1['time'] = pd.to_datetime(df_temp1['time'], format='%H:%M:%S').dt.time
df_temp1['date'] = pd.to_datetime(df_temp1['date'], format='%b %d %Y')
df_temp1['datetime'] = pd.to_datetime(df_temp1['date'].astype(str)+' '+df_temp1['time'].astype(str))
df_power1['time'] = pd.to_datetime(df_power1['time'], format='%H:%M:%S.%f').dt.time
df_power1['date'] = pd.to_datetime(df_power1['date'], format='%m/%d/%Y')
df_power1['datetime'] = pd.to_datetime(df_power1['date'].astype(str)+' '+df_power1['time'].astype(str))
print(df_power1)
print(df_power1.dtypes)
print(df_temp1)
print(df_temp1.dtypes)

#plot part
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df_power1['datetime'], df_power1['power'], color='red', label='Power')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()

ax2.plot(df_temp1['datetime'], df_temp1['temp'], color='blue', label='Temperature')
ax2.set_ylabel('Temperature', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Power and Temperature over Time')

fig.tight_layout()
plt.show()