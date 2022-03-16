import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_full_data=pd.read_csv('../processed_data/full_data.csv')

print(type(df_full_data['Datum'][1])) #the 'Datum' column has str-format, not date-format
df_full_data['Datum'] = df_full_data['Datum'].map(lambda x: str(x)[:-7]) #in this process we tried to remove '00+0100' and adding '30' to get a good date format working with
df_full_data['Datum'] = df_full_data['Datum'].astype(str) + '30'
df_full_data['Datum'] = pd.to_datetime(df_full_data['Datum'], format="%Y-%m-%dT%H:%M")

#plotting figure 1 (PM2.5 and CO all year)
x=df_full_data['Datum']

fig, ax1 = plt.subplots()
ax2=ax1.twinx()
ax1.scatter(x,df_full_data['CO'],s=0.1,c='black',label='CO',alpha=1)
ax2.scatter(x,df_full_data['NO2'], s=0.1,c='b', label='PM2.5',alpha=0.2)

ax1.set_xlabel('Year')
ax1.set_ylabel('PM2.5 concentration (in µg/m3)', color='black')
ax2.set_ylabel('NO2 concentration (in µg/m3)', color='b')
plt.show()

#plotting figure 2 (10.01.2008 vs 10.01.2021 cars)
subdf_full_data=df_full_data.loc['Datum']== '2008-01-10T18:30' #dos't work
