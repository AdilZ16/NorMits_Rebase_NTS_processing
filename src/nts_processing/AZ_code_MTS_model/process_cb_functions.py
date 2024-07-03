# -*- coding: utf-8 -*-
"""
Created on: 6/26/2024
Original author: Adil Zaheer
"""
import os

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm


# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position



def process_cb_data(data, columns_to_keep, output_folder):
    print('initial data processing')
    # Atkins methodology signified with #AK comment

    # AZ
    df = pd.read_csv(data)
    df = df[columns_to_keep]
    df.columns = [str(col).strip() for col in df.columns]
    print()

    # Individual [at, hh] #AK / #AZ
    df = df[df['period'] != 0]
    df = df[~df['mode'].isin([8, 0])]
    df = df[df['purpose'] != 0]

    df_total = df.groupby(['tfn_at', 'hh_type']).sum()
    df_total = df_total[['trips']].reset_index()

    df_total['split_method'] = np.where(df_total.trips >= 1000, 'observed', 'TBD')
    df_total = df_total.rename(columns={'trips': 'total_trips'})

    df_trip_join = pd.merge(df, df_total, left_on=['tfn_at', 'hh_type'],
                            right_on=['tfn_at', 'hh_type'], how='inner')


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


def process_data_for_hierarchical_bayesian_model(df, columns_to_keep):
    print('processing data to model specifications')

    for var in columns_to_keep:
        # unique values and create new categorical type
        unique_values = df[var].unique()
        df[var] = pd.Categorical(df[var], categories=unique_values)

    predict_data = df[df['split_method'] == 'tbd'].copy()
    training_data = df[df['split_method'] != 'tbd'].copy()

    return df, predict_data, training_data


def generate_priors(data):
    print('generating priors')
    start_time = time.time()

    with pm.Model() as trip_model:
        # Hyper-priors (mean)
        sigma_purpose_hc = pm.HalfCauchy('sigma_purpose', beta=5)
        sigma_tfn_at_hc = pm.HalfCauchy('sigma_tfn_at', beta=5)
        sigma_hh_type_hc = pm.HalfCauchy('sigma_hh_type', beta=5)
        sigma_mode_hc = pm.HalfCauchy('sigma_mode', beta=5)
        sigma_period_hc = pm.HalfCauchy('sigma_period', beta=5)

        # Hierarchical priors (mean)
        beta_purpose = pm.Normal('beta_purpose', mu=0, sigma=sigma_purpose_hc, shape=len(data['purpose'].cat.categories))
        beta_tfn_at = pm.Normal('beta_tfn_at', mu=0, sigma=sigma_tfn_at_hc, shape=len(data['tfn_at'].cat.categories))
        beta_hh_type = pm.Normal('beta_hh_type', mu=0, sigma=sigma_hh_type_hc, shape=len(data['hh_type'].cat.categories))
        beta_mode = pm.Normal('beta_mode', mu=0, sigma=sigma_mode_hc, shape=len(data['mode'].cat.categories))
        beta_period = pm.Normal('beta_period', mu=0, sigma=sigma_period_hc, shape=len(data['period'].cat.categories))

        #  intercept
        intercept = pm.Normal('intercept', mu=0, sigma=10)

        # Linear combination (model form)
        mu_mean = (intercept + beta_purpose[data['purpose'].cat.codes] +
              beta_tfn_at[data['tfn_at'].cat.codes] +
              beta_hh_type[data['hh_type'].cat.codes] +
              beta_mode[data['mode'].cat.codes] +
              beta_period[data['period'].cat.codes])

        # Expected value of outcome (must be positive)
        theta = pm.math.exp(mu_mean)

        alpha = pm.Gamma('alpha', alpha=0.01, beta=0.01)

        # Likelihood (sampling distribution) of observations
        trips = pm.NegativeBinomial('trips', mu=theta, alpha=alpha, observed=data['trips'])

    with trip_model:
        # Using NUTS sampler, uses trips likelihood, using metropolis for speed instead of nuts
        print('sampling process beginning')
        trace = pm.sample(500, tune=250, cores=4, random_seed=42, progressbar=True, step=pm.Metropolis())
        # try this instead, can be faster. variational inference instead of mcmc
        #approx = pm.fit(n=50000, method='advi')
        #trace = approx.sample(500)
    end_time = time.time()
    print(f"Sampling took {end_time - start_time:.2f} seconds")

    return trip_model, trace



def predict_model_hierarchical_bayesian_model(columns_to_keep,
                                              data_to_predict,
                                              training_data,
                                              trip_model,
                                              trace,
                                              output_folder):
    print('predictions being made')
    for var in columns_to_keep:
        data_to_predict[var] = pd.Categorical(data_to_predict[var], categories=training_data[var].cat.categories)

    # Make predictions
    with trip_model:
        pred_samples = pm.sample_posterior_predictive(trace, var_names=['trips'])

    # Extract predicted trips
    predicted_trips = pred_samples['trips']

    # Calculate mean predictions and credible intervals
    predictions = predicted_trips.mean(axis=0)
    credible_intervals = np.percentile(predicted_trips, [2.5, 97.5], axis=0)

    # Add predictions to new_data
    data_to_predict['predicted_trips'] = predictions
    data_to_predict['lower_ci'] = credible_intervals[0]
    data_to_predict['upper_ci'] = credible_intervals[1]

    output_filename = 'predictions.csv'
    output_path = os.path.join(output_folder, output_filename)
    predictions.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"predictions exported to: {output_path}")

    return predictions, credible_intervals
