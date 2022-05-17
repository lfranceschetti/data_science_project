###############################################
###                Imports                  ###
###############################################
import pandas as pd
import numpy as np
df = pd.read_csv('../../processed_data/full_data.csv')


# Values where we need to take the sum over 24h
add_vals = ["Tagesdatum", "Zweirad",
            "Personenwagen", "Lastwagen", "RainDur", ]

# Values where we need to take the mean over 24h
mean_vals = ["Tagesdatum", "Hr", "T", "WVs", "StrGlo", "p", "NO2",
             "NO", "NOx", "O3", "CO", "PM10", "SO2"]

cols = df.drop(columns=["Datum", "Zeit"]).columns

df["Tagesdatum"] = df.apply(lambda x: str(x["Datum"])[:10], axis=1)

add_df = df[add_vals]
mean_df = df[mean_vals]


mean_df = mean_df.groupby("Tagesdatum").mean()
add_df = add_df.groupby("Tagesdatum").sum(min_count=1)


df_daily = pd.merge(
    add_df, mean_df, how="outer", on="Tagesdatum")


df_daily = df_daily.reset_index()

df_daily["Jahr"] = df_daily.apply(lambda x: x["Tagesdatum"][:4], axis=1)
df_daily["Monat"] = df_daily.apply(lambda x: x["Tagesdatum"][5:7], axis=1)
df_daily["Tag"] = df_daily.apply(lambda x: x["Tagesdatum"][8:10], axis=1)

cols = list(cols)
cols.insert(3, "Tagesdatum")

df_daily = df_daily[cols]

df_daily.to_csv('../../processed_data/full_data_daily.csv',
                index=False, float_format='%.3f')

