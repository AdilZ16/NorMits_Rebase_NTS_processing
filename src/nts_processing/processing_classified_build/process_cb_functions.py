# -*- coding: utf-8 -*-
"""
Created on: 6/26/2024
Original author: Adil Zaheer
"""
import os

import dill
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm
import os
import joblib
import arviz as az
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position



def process_cb_data_atkins_method(data, columns_to_keep, output_folder):
    print('initial data processing')
    # Atkins methodology signified with #AK comment

    # AZ
    df = pd.read_csv(data)
    df = df[columns_to_keep]
    df.columns = [str(col).strip() for col in df.columns]

    # Individual [at, hh] #AK / #AZ
    df = df[df['period'] != 0]
    df = df[~df['mode'].isin([8, 0])]
    #df = df[df['purpose'] != 0]
    df = df[~df['purpose'].isin([0, 2, 3, 4, 5, 6, 7, 8])] # testing purpose 1 only

    df_total = df.groupby(['tfn_at', 'hh_type', 'purpose', 'mode', 'period']).sum()
    df_total = df_total[['trips']].reset_index()

    df_total['split_method'] = np.where(df_total.trips >= 1000, 'observed', 'TBD')
    df_total = df_total.rename(columns={'trips': 'total_trips'})

    df_trip_join = pd.merge(df, df_total, left_on=['tfn_at', 'hh_type', 'purpose', 'mode', 'period'],
                            right_on=['tfn_at', 'hh_type', 'purpose', 'mode', 'period'], how='inner')

    df_trip_join['split'] = df_trip_join['trips'] / df_trip_join['total_trips']
    columns_to_keep_with_additional = columns_to_keep + ['split', 'total_trips', 'split_method']
    df_trip_join = df_trip_join[columns_to_keep_with_additional]
    df_trip_join['mode_period'] = df_trip_join['mode'].astype(str) + '_' + df_trip_join['period'].astype(str)

    df_trip_join = df_trip_join.sort_values(by=['purpose', 'tfn_at', 'hh_type', 'mode', 'period', 'mode_period'])
    df_trip_join = df_trip_join[['purpose', 'tfn_at', 'hh_type', 'mode', 'period', 'trips', 'split', 'total_trips', 'split_method', 'mode_period']]

    df_trip_join = df_trip_join.fillna(0)
    audit_df = pd.DataFrame(columns=['purpose', 'area_type', 'agg', 'total_trips', 'status'])

    output_filename = 'df_trip_join_pre_modelling.csv'
    output_path = os.path.join(output_folder, output_filename)
    df_trip_join.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"df_trip_join exported to: {output_path}")


    return df_trip_join, audit_df


def process_cb_data_tfn_method(data, columns_to_keep, output_folder, purpose_value):
    """
    :param data: data to process
    :param columns_to_keep: which columns in the data you want to keep. Columns not included will
                            still be kept, this function ensures that regardless of any data
                            processing, the column will be kept.
    :param output_folder:   path to output folder. Where you want model outputs to go.
    :param purpose_value:   value from 1 to 8 to dictate which purpose is being modelling.
    :return: processed data which will be exported to your output folder.
    """
    df = pd.read_csv(data)
    df = df[columns_to_keep]
    df.columns = [str(col).strip() for col in df.columns]

    df = df[df['period'] != 0]
    df = df[~df['mode'].isin([8, 0])]
    df = df[df['purpose'] == purpose_value]
    # df = df[~df['purpose'].isin([0, 2, 3, 4, 5, 6, 7, 8])]

    df_total = df.groupby(['tfn_at', 'hh_type', 'purpose', 'mode', 'period']).sum()
    df_total = df_total[['trips']].reset_index()

    df_total['split_method'] = np.where(df_total.trips >= 1000, 'observed', 'TBD')
    df_total = df_total.rename(columns={'trips': 'total_trips'})

    df_total['mode_period'] = df_total['mode'].astype(str) + '_' + df_total['period'].astype(str)
    df_total.columns = [str(col).strip() for col in df_total.columns]


    output_filename = 'df_total.csv'
    output_path = os.path.join(output_folder, output_filename)
    df_total.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"df_trip_join exported to: {output_path}")

    return df_total


def process_data_for_hierarchical_bayesian_model(df, columns_to_keep, reduce_data_size, target_column, output_folder):
    print('processing data to model specifications')

    if reduce_data_size is not None:
        print('Reducing data size')
        df = create_hybrid_sample(df, output_folder=output_folder)

    for var in df.columns:
        if var != target_column:
            # unique values and create new categorical type (excluding continuous target)
            unique_values = df[var].unique()
            df[var] = pd.Categorical(df[var], categories=unique_values)

    #predict_data = df[df['split_method'] == 'TBD'].copy()
    #training_data = df[df['split_method'] != 'TBD'].copy()

    output_filename = 'final_data_to_model.csv'
    output_path = os.path.join(output_folder, output_filename)
    df.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"final_data_to_model exported to: {output_path}")

    return df
