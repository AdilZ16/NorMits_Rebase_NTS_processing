# -*- coding: utf-8 -*-
"""
Created on: 6/21/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error




#read in the dataframe which is the input into agg_fill



df = pd.read_csv('dfr_split_before.csv')
df = df[['purpose','tfn_at','hh_type','mode','period','trips']]

#The decision to remove mode 8 (air travel) from the mode was made.
df = df[df['mode'] != 8]
#calculate the total trips at the p,at,hh level
df_total = df.groupby(['purpose','tfn_at','hh_type']).sum()
df_total = df_total[['trips']].reset_index()

#determine whether the combination needs to be modelled or the raw splits can be used, if trips>500 use observed trips to derive splits
df_total['split_method'] = np.where(df_total.trips>=500, 'observed', 'TBD')
df_total = df_total.rename(columns={'trips': 'total_trips'})
#calculate the raw split (trips/(total trips at the p,at,hh level))
df_trip_join = pd.merge(df, df_total, left_on =['purpose','tfn_at','hh_type'] , right_on = ['purpose','tfn_at','hh_type'], how='inner')
df_trip_join['split'] = df_trip_join['trips'] / df_trip_join['total_trips']
df_trip_join = df_trip_join[['purpose','tfn_at','hh_type','mode', 'period','split','trips','total_trips','split_method']]
df_trip_join['mode_period'] = df_trip_join['mode'].astype(str) + '_' + df_trip_join['period'].astype(str)
df_trip_join.to_csv('trip_join.csv')




df_trip_join = df_trip_join.fillna(0)  # ensure that the target variable y (splits) always has a value
audit_df = pd.DataFrame(columns=['purpose', 'area_type', 'agg', 'total_trips', 'status'])


# define aggregation hierarchies for household type and new TfN area type
tfn_at = {1: [1, 2], 2: [1, 2], 3: [3, 4, 5, 6, 7, 8], 4: [3, 4, 5, 6, 7, 8],
          5: [3, 4, 5, 6, 7, 8], 6: [3, 4, 5, 6, 7, 8], 7: [3, 4, 5, 6, 7, 8],
          8: [3, 4, 5, 6, 7, 8],
          9: [9, 10, 11], 10: [9, 10, 11], 11: [9, 10, 11], 12: [12, 13], 13: [12, 13],
          14: [14, 15], 15: [14, 15], 16: [16, 17],
          17: [16, 17], 18: [18, 19], 19: [18, 19], 20: [20]}
list_hh_type = [[1, 3, 6], [2, 5, 8], [4, 7]]
modelled_df = pd.DataFrame()
modelled_at = pd.DataFrame()
for p in range(1, 9):
    for at in range(1, 21):
        df_mode_period_join_f = df_trip_join[
            (df_trip_join['purpose'] == p) & (df_trip_join['tfn_at'] == at)]
        for hh in list_hh_type:
            df_mode_period_join_hh = df_mode_period_join_f[df_mode_period_join_f.hh_type.isin(hh)]
            needs_pred = df_mode_period_join_hh[df_mode_period_join_hh.split_method == 'TBD']
            trips = df_mode_period_join_hh.total_trips.unique().sum()

            if not needs_pred.empty:
                if trips > 500:
                    data_df = [[p, at, hh, trips, 'Need hh_type aggregation']]
                    audit_df = audit_df.append(pd.Series(data_df[0], index=audit_df.columns),
                                               ignore_index=True)
                    if not needs_pred.empty:
                        X_train_cat = df_mode_period_join_hh[['purpose', 'tfn_at', 'mode_period']]
                        X_train_prob = df_mode_period_join_hh[['split']]

                        # Encode categorical features using one-hot encoding
                        encoder = OneHotEncoder(sparse_output=False)
                        X = encoder.fit_transform(X_train_cat)

                        y = X_train_prob['split'].reset_index()
                        y = np.array([y['split']]).T

                        # Initialize and train the model
                        model = LinearRegression()
                        model.fit(X, y)

                        X_predictor = encoder.fit_transform(X_train_cat.drop_duplicates())
                        y_predicted_new_data = model.predict(X_predictor)

                        m_p = X_train_cat.drop_duplicates()[['mode_period']]

                        df_predict = pd.DataFrame(y_predicted_new_data)
                        df_model = df_predict.rename(columns={0: 'modelled'})
                        df_model = df_model.clip(lower=0)

                        df_model['purpose'] = p
                        df_model['tfn_at'] = at
                        df_model = df_model.reset_index()
                        m_p = m_p.reset_index()
                        df_model['mode_period'] = m_p['mode_period']
                        # store DataFrame in list
                        df_model['split_method'] = str(hh)
                        # store df
                        modelled_df = pd.concat([modelled_df, df_model])
                else:
                    if not needs_pred.empty:
                        data_df = [[p, at, hh, trips, 'Need tfn_at aggregation']]
                        audit_df = audit_df.append(pd.Series(data_df[0], index=audit_df.columns),
                                                   ignore_index=True)
                        df_mode_period_join_at = df_trip_join[(df_trip_join['purpose'] == p)]
                        df_mode_period_join_at = df_mode_period_join_at[
                            df_mode_period_join_at.tfn_at.isin(tfn_at[at])]
                        df_mode_period_join_at = df_mode_period_join_at[
                            df_mode_period_join_at.hh_type.isin(hh)]
                        X_train_cat = df_mode_period_join_at[['purpose', 'mode_period']]
                        X_train_prob = df_mode_period_join_at[['split']]

                        # Encode categorical features using one-hot encoding
                        encoder = OneHotEncoder(sparse_output=False)
                        X = encoder.fit_transform(X_train_cat)

                        y = X_train_prob['split'].reset_index()
                        y = np.array([y['split']]).T

                        # Initialize and train the model
                        model = LinearRegression()
                        model.fit(X, y)

                        X_predictor = encoder.fit_transform(X_train_cat.drop_duplicates())
                        y_predicted_new_data = model.predict(X_predictor)

                        m_p = X_train_cat.drop_duplicates()[['mode_period']]

                        df_predict = pd.DataFrame(y_predicted_new_data)
                        df_model = df_predict.rename(columns={0: 'modelled'})
                        df_model = df_model.clip(lower=0)

                        df_model['purpose'] = p
                        df_model['hh'] = str(hh)
                        df_model = df_model.reset_index()
                        m_p = m_p.reset_index()
                        df_model['mode_period'] = m_p['mode_period']
                        # store DataFrame in list
                        df_model['split_method'] = str(tfn_at[at])

                        # store df
                        modelled_at = pd.concat([modelled_at, df_model])
                        modelled_at = modelled_at.drop_duplicates()
            else:
                data_df = [
                    [p, at, hh, trips, 'No aggregation all hh types above 500 trips individually']]
                audit_df = audit_df.append(pd.Series(data_df[0], index=audit_df.columns),
                                           ignore_index=True)

# add column describing the method used to derive the splits
modelled_df['modelled'] = modelled_df['modelled'].clip(lower=0)
modelled_df = modelled_df[['mode_period', 'purpose', 'tfn_at', 'modelled', 'split_method']]
modelled_df = modelled_df.assign(
    hh_type=modelled_df['split_method'].str.strip('[]').str.split(',')).explode(
    'hh_type').reset_index(drop=True)
modelled_df.hh_type = modelled_df.hh_type.astype(int)
modelled_df_groupby = modelled_df.groupby(['purpose', 'tfn_at', 'hh_type']).sum().rename(
    columns={'modelled': 'sum of modelled'}).reset_index()

# constraining the mts to 1 to created the adjusted rho column
all_modelled_hh = pd.merge(modelled_df, modelled_df_groupby, how='inner',
                           on=['purpose', 'tfn_at', 'hh_type'])
all_modelled_hh['adjusted_rho'] = all_modelled_hh['modelled'] / all_modelled_hh['sum of modelled']
all_modelled_hh = all_modelled_hh[
    ['mode_period', 'purpose', 'tfn_at', 'hh_type', 'split_method', 'adjusted_rho']]
all_modelled_hh = all_modelled_hh.fillna(0)
all_modelled_hh = all_modelled_hh.drop_duplicates()

# add column describing the method used to derive the splits
modelled_at['modelled'] = modelled_at['modelled'].clip(lower=0)
modelled_df = modelled_at[['mode_period', 'purpose', 'hh', 'modelled', 'split_method']]
modelled_df = modelled_df.assign(
    tfn_at=modelled_df['split_method'].str.strip('[]').str.split(',')).explode(
    'tfn_at').reset_index(drop=True)
modelled_df.tfn_at = modelled_df.tfn_at.astype(int)
modelled_df = modelled_df.assign(hh_type=modelled_df['hh'].str.strip('[]').str.split(',')).explode(
    'hh_type').reset_index(drop=True)
modelled_df.hh_type = modelled_df.hh_type.astype(int)
modelled_df_groupby = modelled_df.groupby(['purpose', 'tfn_at', 'hh_type']).sum().rename(
    columns={'modelled': 'sum of modelled'}).reset_index()

# constraining the mts to 1 to created the adjusted rho column
all_modelled_at = pd.merge(modelled_df, modelled_df_groupby, how='inner',
                           on=['purpose', 'tfn_at', 'hh_type'])
all_modelled_at['adjusted_rho'] = all_modelled_at['modelled'] / all_modelled_at['sum of modelled']
all_modelled_at = all_modelled_at[
    ['mode_period', 'purpose', 'tfn_at', 'hh_type', 'split_method', 'adjusted_rho']]
all_modelled_at = all_modelled_at.fillna(0)
all_modelled_at = all_modelled_at.drop_duplicates()

# create the split column that defined which hh type agg method would be needed
df_trip_join_hh = df_trip_join[
    ['purpose', 'tfn_at', 'mode_period', 'hh_type', 'split', 'split_method']]
df_trip_join_hh = df_trip_join_hh[df_trip_join_hh.split_method == 'TBD']
df_trip_join_hh = df_trip_join_hh.rename(columns={'split_method': 'split_needed'})

df_trip_join_ob = df_trip_join[
    ['purpose', 'tfn_at', 'mode_period', 'hh_type', 'split', 'split_method']]
df_trip_join_ob = df_trip_join_ob[df_trip_join_ob.split_method == 'observed']
df_trip_join_ob = df_trip_join_ob.rename(columns={'split_method': 'split_needed'})

# merge the observed and modelled information together into a final dataframe
resultant_df_hh = pd.merge(df_trip_join_hh, all_modelled_hh,
                           left_on=['purpose', 'tfn_at', 'mode_period', 'hh_type'],
                           right_on=['purpose', 'tfn_at', 'mode_period', 'hh_type'], how='left')
resultant_df_at = resultant_df_hh[resultant_df_hh.split_method.isnull()]
resultant_df_hh = resultant_df_hh[resultant_df_hh.split_method.notnull()]
resultant_df_at = resultant_df_at[
    ['purpose', 'tfn_at', 'mode_period', 'hh_type', 'split', 'split_needed']]
resultant_df_2 = pd.merge(resultant_df_at, all_modelled_at,
                          left_on=['purpose', 'tfn_at', 'mode_period', 'hh_type'],
                          right_on=['purpose', 'tfn_at', 'mode_period', 'hh_type'], how='left')
resultant_df = pd.concat([resultant_df_hh, resultant_df_2])
resultant_df_final = pd.concat([resultant_df, df_trip_join_ob])

resultant_df_final['split_final'] = np.where(resultant_df_final['split_needed'] == 'TBD',
                                             resultant_df_final['adjusted_rho'],
                                             resultant_df_final['split'])

resultant_df_final = resultant_df_final.drop_duplicates()
resultant_df_final.to_csv('mts_output.csv')
audit_df.to_csv('all_audit_output.csv')
resultant_df_final
