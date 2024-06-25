# -*- coding: utf-8 -*-
"""
Created on: 6/25/2024
Original author: Adil Zaheer
"""
import os

# Local Imports
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




# NHB



def process_data_nhb(data, output_folder):
    # read in the dataframe which is the input into agg_fill
    df = pd.read_csv(data).rename(columns={'tfn_at_o': 'tfn_at'})
    df = df[['purpose', 'tfn_at', 'hh_type', 'mode', 'period', 'trips']]

    # aggregate over hh_type
    df = df.groupby(['purpose', 'tfn_at', 'mode', 'period'])['trips'].sum().reset_index()

    # The decision to remove mode 8 (air travel) from the mode was made.
    df = df[df['mode'] != 8]

    # calculate the total trips at the p,at,hh level
    df_total = df.groupby(['purpose', 'tfn_at', 'mode']).sum()
    df_total = df_total[['trips']].reset_index()

    # determine whether the combination needs to be modelled or the raw splits can be used, if trips>100 use observed trips to derive splits
    df_total['split_method'] = np.where(df_total.trips >= 100, 'observed', 'TBD')
    df_total = df_total.rename(columns={'trips': 'total_trips'})

    # calculate the raw split (trips/(total trips at the p,at,hh,m level))
    df_trip_join = pd.merge(df, df_total, left_on=['purpose', 'tfn_at', 'mode'],
                            right_on=['purpose', 'tfn_at', 'mode'], how='inner')
    df_trip_join['split'] = df_trip_join['trips'] / df_trip_join['total_trips']
    df_trip_join = df_trip_join[
        ['purpose', 'tfn_at', 'mode', 'period', 'split', 'trips', 'total_trips', 'split_method']]
    # df_trip_join['mode_period'] = df_trip_join['mode'].astype(str) + '_' + df_trip_join['period'].astype(str)

    output_filename = 'df_trip_join_nhb_pre_modelling.csv'
    output_path = os.path.join(output_folder, output_filename)
    df_trip_join.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"df_trip_join_nhb_pre_modelling exported to: {output_path}")


    return df_trip_join


def model_unedited_nhb(df_trip_join, output_folder):
    df_trip_join = df_trip_join.fillna(
        0)  # ensure that the target variable y (splits) always has a value
    audit_df = pd.DataFrame(columns=['purpose', 'mode', 'area_type', 'total_trips', 'status'])

    # define aggregation hierarchies for new TfN area type
    tfn_at = {1: [1, 2], 2: [1, 2], 3: [3, 4, 5, 6, 7, 8], 4: [3, 4, 5, 6, 7, 8],
              5: [3, 4, 5, 6, 7, 8], 6: [3, 4, 5, 6, 7, 8], 7: [3, 4, 5, 6, 7, 8],
              8: [3, 4, 5, 6, 7, 8],
              9: [9, 10, 11], 10: [9, 10, 11], 11: [9, 10, 11], 12: [12, 13], 13: [12, 13],
              14: [14, 15], 15: [14, 15], 16: [16, 17],
              17: [16, 17], 18: [18, 19], 19: [18, 19], 20: [20]}
    # list_hh_type = [[1,3,6], [2,5,8], [4,7]]
    modelled_df = pd.DataFrame()
    modelled_at = pd.DataFrame()
    for p in range(1, 9):
        for at in range(1, 21):
            for m in range(1, 8):
                df_mode_period_join_f = df_trip_join[
                    (df_trip_join['purpose'] == p) & (df_trip_join['mode'] == m) & (
                                df_trip_join['tfn_at'] == at)]
                df_mode_period_join_hh = df_mode_period_join_f

                needs_pred = df_mode_period_join_hh[df_mode_period_join_hh.split_method == 'TBD']
                trips = df_mode_period_join_hh.total_trips.unique().sum()

                if not needs_pred.empty:
                    data_df = [[p, m, at, trips, 'Need tfn_at aggregation']]
                    #audit_df = audit_df.append(pd.Series(data_df[0], index=audit_df.columns), ignore_index=True)
                    audit_df = pd.concat([audit_df, pd.DataFrame(data_df, columns=audit_df.columns)], ignore_index=True)

                    df_mode_period_join_at = df_trip_join[(df_trip_join['purpose'] == p)]
                    df_mode_period_join_at = df_mode_period_join_at[
                        (df_mode_period_join_at['mode'] == m)]
                    df_mode_period_join_at = df_mode_period_join_at[
                        df_mode_period_join_at.tfn_at.isin(tfn_at[at])]

                    X_train_cat = df_mode_period_join_at[['purpose', 'period', 'mode']]
                    X_train_prob = df_mode_period_join_at[['split']]

                    # Encode categorical features using one-hot encoding
                    encoder = OneHotEncoder(sparse=False)
                    X = encoder.fit_transform(X_train_cat)

                    y = X_train_prob['split'].reset_index()
                    y = np.array([y['split']]).T

                    # Initialize and train the model
                    model = LinearRegression()
                    model.fit(X, y)

                    X_predictor = encoder.fit_transform(X_train_cat.drop_duplicates())
                    y_predicted_new_data = model.predict(X_predictor)

                    m_p = X_train_cat.drop_duplicates()[['period']]

                    df_predict = pd.DataFrame(y_predicted_new_data)
                    df_model = df_predict.rename(columns={0: 'modelled'})
                    df_model = df_model.clip(lower=0)

                    df_model['purpose'] = p
                    df_model['mode'] = m
                    df_model = df_model.reset_index()
                    m_p = m_p.reset_index()
                    df_model['period'] = m_p['period']
                    # store DataFrame in list
                    df_model['split_method'] = str(tfn_at[at])

                    # store df
                    modelled_at = pd.concat([modelled_at, df_model])
                    modelled_at = modelled_at.drop_duplicates()
                else:
                    data_df = [[p, m, at, trips,
                                'No aggregation all hh types above 100 trips individually']]
                    #audit_df = audit_df.append(pd.Series(data_df[0], index=audit_df.columns), ignore_index=True)
                    audit_df = pd.concat([audit_df, pd.DataFrame(data_df, columns=audit_df.columns)], ignore_index=True)


    # add column describing the method used to derive the splits
    modelled_at['modelled'] = modelled_at['modelled'].clip(lower=0)
    modelled_df = modelled_at[['period', 'purpose', 'mode', 'modelled', 'split_method']]
    modelled_df = modelled_df.assign(
        tfn_at=modelled_df['split_method'].str.strip('[]').str.split(',')).explode(
        'tfn_at').reset_index(drop=True)
    modelled_df.tfn_at = modelled_df.tfn_at.astype(int)
    modelled_df_groupby = modelled_df.groupby(['purpose', 'tfn_at', 'mode']).sum().rename(
        columns={'modelled': 'sum of modelled'}).reset_index()


    #added this line to make code work #todo
    modelled_df_groupby = modelled_df_groupby.rename(columns={'split_method': 'split_method_grouped'})

    # constraining the mts to 1 to created the adjusted rho column
    all_modelled_at = pd.merge(modelled_df, modelled_df_groupby, how='inner',
                               on=['purpose', 'tfn_at', 'mode'])
    all_modelled_at['adjusted_rho'] = all_modelled_at['modelled'] / all_modelled_at[
        'sum of modelled']
    all_modelled_at = all_modelled_at[
        ['period_x', 'purpose', 'tfn_at', 'mode', 'split_method', 'adjusted_rho']]
    all_modelled_at = all_modelled_at.rename(columns={'period_x': 'period'})
    all_modelled_at = all_modelled_at.fillna(0)
    all_modelled_at = all_modelled_at.drop_duplicates()

    # create the split column that defined which hh type agg method would be needed
    df_trip_join_hh = df_trip_join[
        ['purpose', 'tfn_at', 'period', 'mode', 'split', 'split_method']]
    df_trip_join_hh = df_trip_join_hh[df_trip_join_hh.split_method == 'TBD']
    df_trip_join_hh = df_trip_join_hh.rename(columns={'split_method': 'split_needed'})

    df_trip_join_ob = df_trip_join[
        ['purpose', 'tfn_at', 'period', 'mode', 'split', 'split_method']]
    df_trip_join_ob = df_trip_join_ob[df_trip_join_ob.split_method == 'observed']
    df_trip_join_ob = df_trip_join_ob.rename(columns={'split_method': 'split_needed'})

    # merge the observed and modelled information together into a final dataframe
    resultant_df_hh = pd.merge(df_trip_join_hh, all_modelled_at,
                               left_on=['purpose', 'tfn_at', 'period', 'mode'],
                               right_on=['purpose', 'tfn_at', 'period', 'mode'], how='left')
    resultant_df_at = resultant_df_hh[resultant_df_hh.split_method.isnull()]
    resultant_df_hh = resultant_df_hh[resultant_df_hh.split_method.notnull()]

    resultant_df_at = resultant_df_at[
        ['purpose', 'tfn_at', 'period', 'mode', 'split', 'split_needed']]
    resultant_df_2 = pd.merge(resultant_df_at, all_modelled_at,
                              left_on=['purpose', 'tfn_at', 'period', 'mode'],
                              right_on=['purpose', 'tfn_at', 'period', 'mode'], how='left')
    resultant_df = pd.concat([resultant_df_hh, resultant_df_2])
    resultant_df_final = pd.concat([resultant_df, df_trip_join_ob])

    resultant_df_final['split_final'] = np.where(resultant_df_final['split_needed'] == 'TBD',
                                                 resultant_df_final['adjusted_rho'],
                                                 resultant_df_final['split'])

    resultant_df_final = resultant_df_final.drop_duplicates()


    output_filename = 'resultant_df_nhb.csv'
    output_path = os.path.join(output_folder, output_filename)
    resultant_df_final.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"resultant_df_nhb exported to: {output_path}")

    output_filename = 'audit_df_nhb.csv'
    output_path = os.path.join(output_folder, output_filename)
    audit_df.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"audit_df_nhb exported to: {output_path}")

    output_filename = 'df_trip_join_nhb.csv'
    output_path = os.path.join(output_folder, output_filename)
    df_trip_join.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"df_trip_join_nhb exported to: {output_path}")

    output_filename = 'all_modelled_at_nhb.csv'
    output_path = os.path.join(output_folder, output_filename)
    all_modelled_at.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"all_modelled_at_nhb exported to: {output_path}")

    return resultant_df_final, audit_df, df_trip_join, all_modelled_at


def model_evaluation_atkins_method_tfnat_nhb(df_trip_join, all_modelled_at, output_folder):
    p = 1
    m = 2

    df_hh = df_trip_join[df_trip_join.tfn_at.isin([16, 17])]
    df_mode_period_join_f = df_hh[(df_hh['purpose'] == p) & (df_hh['mode'] == m)]

    # create the dataframe which has the observed splits to compare against
    comparison_df = df_mode_period_join_f.pivot_table(index=['purpose', 'tfn_at', 'mode'],
                                                      columns='period', values='split',
                                                      aggfunc='sum', fill_value=0).reset_index()
    pivot_df_T = comparison_df.T.reset_index().loc[3::]
    pivot_df_T['purpose'] = p
    pivot_df_T['mode'] = m
    df_model = all_modelled_at[
        (all_modelled_at['purpose'] == p) & (all_modelled_at['mode'] == m) & (
                    all_modelled_at['split_method'] == str([16, 17]))]

    if not df_model.empty:
        join_model_obs = pd.concat(
            [pivot_df_T.reset_index(), df_model.reset_index()['adjusted_rho']], axis=1)
        join_model_obs = join_model_obs.rename(columns={0: 'area_type_' + str(16)})
        join_model_obs = join_model_obs.rename(columns={1: 'area_type_' + str(17)})
        join_model_obs = join_model_obs[
            ['period', 'area_type_' + str(16), 'area_type_' + str(17), 'adjusted_rho']]

        # plot the observed vs modelled
        dfm = pd.melt(join_model_obs, id_vars="period", var_name="number", value_name="prob")

        output_filename = 'model_scoring_doc_at_nhb.csv'
        output_path = os.path.join(output_folder, output_filename)
        dfm.to_csv(output_path, index=True)
        print('-------------------------------------------------------------')
        print(f"model_scoring_doc_at_nhb exported to: {output_path}")

        return dfm
