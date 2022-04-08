import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_full_data = pd.read_csv('../processed_data/full_data.csv')
print(type(df_full_data['Datum'][1]))  # the 'Datum' column has str-format, not date-format
df_full_data['Datum'] = df_full_data['Datum'].map(lambda x: str(x)[
                                                            :-7])  # in this process we tried to remove '00+0100' and adding '30' to get a good date format working with
df_full_data['Datum'] = df_full_data['Datum'].astype(str) + '30'
df_full_data['Datum'] = pd.to_datetime(df_full_data['Datum'], format="%Y-%m-%dT%H:%M")
print('format changed to DATE')


# plotting figure 1 (PM2.5 and CO all year)
def plot_fig1(save_file=False,file_name=None):
    x = df_full_data['Datum']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(x, df_full_data['CO'], s=0.1, c='black', label='CO', alpha=1)
    ax2.scatter(x, df_full_data['PM2.5'], s=0.1, c='b', label='PM2.5', alpha=0.2)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('PM2.5 concentration (in µg/m3)', color='black')
    ax2.set_ylabel('NO2 concentration (in µg/m3)', color='b')

    if save_file == False:
        plt.show()

    if save_file == True:
        plt.savefig(f"../processed_data/{file_name}.png",dpi=300)

#pairplot
def plot_pairplot(list_of_vars,save_file=False,file_name=None):
    pairplot=sns.pairplot(df_full_data          # if sample, do df_full_data.sample(size_of_sample)
                 , vars=list_of_vars
                 , kind="scatter"               # reg, hist, scatter, kde
                 , plot_kws=dict(s=1)           # size of points
                 , height=2.5
                 )
    if save_file == True:

        plt.close(pairplot.fig)
        pairplot.figure.savefig(f"../processed_data/{file_name}.png",dpi=300)

    plt.show()


plot_fig1()