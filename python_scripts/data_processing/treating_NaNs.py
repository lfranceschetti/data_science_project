import pandas as pd
import csv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import MissingIndicator
from sklearn.linear_model import BayesianRidge


paths = ['../../processed_data/full_data_daily.csv',
         '../../processed_data/full_data.csv']
save = ['../../processed_data/full_data_imputed_daily.csv',
        '../../processed_data/full_data_imputed.csv']

for i, path in enumerate(paths):
    df = pd.read_csv(path)
    # Drop columns only if they have no a single value
    df.dropna(thresh=16)

    print(df)

    datum_cols = ["Jahr", "Monat", "Tag", "Tagesdatum"]
    if(i == 1):
        datum_cols.pop()
        datum_cols.append("Zeit")
        datum_cols.append("Datum")
    date = df[datum_cols]
    df = df.drop(columns=date)

    # indicator = MissingIndicator(features="all")
    # res = indicator.fit_transform(df_mean)
    # cols_missing = [col+ "_miss" for col in cols]
    # df_miss = pd.DataFrame(res , index=df_mean.index, columns=cols_missing)
    # df_miss.replace({False: 0, True: 1}, inplace=True)

    # df_mean_miss = pd.concat([df_mean, df_miss], axis=1)

    # Iterative Inputer
    iterativeImputer = IterativeImputer(
        estimator=BayesianRidge(), random_state=0)
    df_imputed = pd.DataFrame(iterativeImputer.fit_transform(
        df), index=df.index, columns=df.columns)

    df_imputed[["Zweirad",
                "Personenwagen", "Lastwagen"]] = df_imputed[["Zweirad",
                                                             "Personenwagen", "Lastwagen"]].round(0).astype(int)

    df_imputed[datum_cols] = date

    cols = df_imputed.columns.to_list()
    new_cols = cols[16:] + cols[:16]
    df_imputed = df_imputed[new_cols]

    df_imputed.to_csv(save[i], index=False, float_format='%.3f')
