import pandas as pd

dt_air_2012 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2012.csv', sep=',')
dt_air_2013 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2013.csv', sep=',')
dt_air_2014 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2014.csv', sep=',')
dt_air_2015 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2015.csv', sep=',')
dt_air_2016 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2016.csv', sep=',')
dt_air_2017 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2017.csv', sep=',')
dt_air_2018 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2018.csv', sep=',')
dt_air_2019 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2019.csv', sep=',')
dt_air_2020 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2020.csv', sep=',')
dt_air_2021 = pd.read_csv(
    './data/air_quality/ugz_ogd_air_d1_2021.csv', sep=',')
print('success loading dataframes from air_pollution')
dt_air_all_unfiltered = pd.concat([dt_air_2012, dt_air_2013, dt_air_2014, dt_air_2015,
                                  dt_air_2015, dt_air_2016, dt_air_2017, dt_air_2018, dt_air_2019, dt_air_2020, dt_air_2021])
print('success appending all dataframes')
dt_air_stampf_all = dt_air_all_unfiltered.loc[dt_air_all_unfiltered['Standort']
                                              == 'Zch_Stampfenbachstrasse']
print('success filtering Standort == Stampfenbachstrasse')
dt_air_stampf_all.to_csv('./data/air_quality/ugz_ogd_air_stampf_all.csv')
print('CSV saved to ./data/air_quality/ugz_ogd_air_stampf_all.csv')

dt_car_2012 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2012.csv', sep=',', low_memory=False)
dt_car_2013 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2013.csv', sep=',', low_memory=False)
dt_car_2014 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2014.csv', sep=',', low_memory=False)
dt_car_2015 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2015.csv', sep=',', low_memory=False)
dt_car_2016 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2016.csv', sep=',', low_memory=False)
dt_car_2017 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2017.csv', sep=',', low_memory=False)
dt_car_2018 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2018.csv', sep=',', low_memory=False)
dt_car_2019 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2019.csv', sep=',', low_memory=False)
dt_car_2020 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2020.csv', sep=',', low_memory=False)
dt_car_2021 = pd.read_csv(
    './data/cars/sid_dav_verkehrszaehlung_miv_OD2031_2021.csv', sep=',', low_memory=False)
print('success loading dataframes from dav_verkehrszaehlung')
dt_car_all_unfiltered = pd.concat([dt_car_2012, dt_car_2013, dt_car_2014, dt_car_2015,
                                  dt_car_2016, dt_car_2017, dt_car_2018, dt_car_2019, dt_car_2020, dt_car_2021])
print('success appending all dataframes')
dt_car_all_stampf = dt_car_all_unfiltered.loc[dt_car_all_unfiltered['Achse']
                                              == 'Stampfenbachstrasse']
print('success filtering Standort == Stampfenbachstrasse')
dt_car_all_stampf.to_csv(
    './data/cars/sid_dav_verkehrszaehlung_stampf_all', index=False)
print('CSV saved to ./data/cars/sid_dav_verkehrszaehlung_stampf_all.csv')

print('--- Task ended ---')