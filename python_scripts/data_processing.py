# https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options
import pandas as pd
years = [f'{i}' for i in range(2007, 2022)]

#test alisha
def get_df_of_year(year):
    df_air = pd.read_csv(f'../data/air_quality/ugz_ogd_air_h1_{year}.csv')
    df_traffic = pd.read_csv(f'../data/traffic/ugz_ogd_traffic_h1_{year}.csv')
    df_meteo = pd.read_csv(f'../data/meteo/ugz_ogd_meteo_h1_{year}.csv')

    df_air = df_air.loc[df_air['Standort'] == 'Zch_Stampfenbachstrasse']
    df_meteo = df_meteo.loc[df_meteo['Standort'] == 'Zch_Stampfenbachstrasse']

    # Handle the airquality data
    df_air2 = df_air.loc[:, ['Datum']]
    df_air2.drop_duplicates(subset="Datum", keep='first',
                            inplace=True, ignore_index=True)

    gases = ["NO2", "NO", "NOx", "O3", "CO", "PM10", "SO2"]
    for gas in gases:
        df_gas = df_air.loc[df_air['Parameter'] == gas][["Datum", "Wert"]]
        df_air2 = pd.merge(df_air2, df_gas, how="outer", on="Datum")
        df_air2 = df_air2.rename(columns={'Wert': gas})

    # Handle the traffic data
    df_traffic2 = df_traffic.loc[:, ['Datum']]
    df_traffic2.drop_duplicates(subset="Datum", keep='first',
                                inplace=True, ignore_index=True)

    vehicles = ["Zweirad", "Personenwagen", "Lastwagen"]
    for vehicle in vehicles:
        df_vehicle = df_traffic.loc[df_traffic["Klasse.Text"] == vehicle]
        df_schaffhauser = df_vehicle.loc[df_vehicle["Richtung"]
                                         == "Schaffhauserplatz"][["Datum", "Anzahl"]]
        df_stampfenbach = df_vehicle.loc[df_vehicle["Richtung"]
                                         == "Stampfenbachplatz"][["Datum", "Anzahl"]]
        df_traffic2 = pd.merge(
            df_traffic2, df_schaffhauser, how="outer", on="Datum")
        df_traffic2 = df_traffic2.rename(columns={'Anzahl': vehicle +
                                                  "_schaff"})
        df_traffic2 = pd.merge(
            df_traffic2, df_stampfenbach, how="outer", on="Datum")
        df_traffic2 = df_traffic2.rename(columns={'Anzahl': vehicle +
                                                  "_stampf"})

    # Handle the meteo data
    df_meteo2 = df_meteo.loc[:, ['Datum']]
    df_meteo2.drop_duplicates(subset="Datum", keep='first',
                              inplace=True, ignore_index=True)

    props = ["Hr", "RainDur", "T", "WD", "WVv", "WVs", "StrGlo", "p"]
    for prop in props:
        df_prop = df_meteo.loc[df_meteo["Parameter"] == prop]
        df_prop = df_prop[["Datum", "Wert"]]
        # To-do should be done like this also for the other datasets
        df_meteo2 = pd.merge(
            df_meteo2, df_prop, how="outer", on="Datum")
        df_meteo2 = df_meteo2.rename(columns={'Wert': prop})

    df_traffic_meteo = pd.merge(df_traffic2, df_meteo2,
                                how="outer", on="Datum", sort=True,
                                )

    df_full = pd.merge(df_traffic_meteo, df_air2,
                       how="outer", on="Datum", sort=True,)

    return df_full


df_list = []
for year in years:
    df_list.append(get_df_of_year(year))
df_final = pd.concat(df_list)
df_final.to_csv('../processed_data/full_data.csv', index=False)
