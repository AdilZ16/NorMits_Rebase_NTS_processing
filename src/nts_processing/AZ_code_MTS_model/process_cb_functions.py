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
import os
import joblib
import arviz as az

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

        # mutable data used as batching the predictions is done due to memory issues, mutable data needed
        trips_data = pm.MutableData('trips_data', data['trips'])

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
        trips = pm.NegativeBinomial('trips', mu=theta, alpha=alpha, observed=trips_data)

    with trip_model:
        # Using NUTS sampler, uses trips likelihood, using metropolis for speed instead of nuts
        print('sampling process beginning')
        trace = pm.sample(200, tune=100, cores=8, random_seed=42, progressbar=True, step=pm.Metropolis())
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
                                              output_folder,
                                              batch_size=100):
    print('predictions being made')
    for var in columns_to_keep:
        data_to_predict[var] = pd.Categorical(data_to_predict[var],
                                              categories=training_data[var].cat.categories)

    predictions = []
    lower_cis = []
    upper_cis = []
    # working in batches due to allocating memory error
    for i in range(0, len(data_to_predict), batch_size):
        batch = data_to_predict.iloc[i:i + batch_size]

        with trip_model:
            # pm.set_data to update the observed data for this batch
            pm.set_data({"trips": batch['trips']})

            # Sample from the posterior predictive distribution
            pred_samples = pm.sample_posterior_predictive(trace, var_names=['trips'])

        predicted_trips = pred_samples['trips']
        batch_predictions = predicted_trips.mean(axis=0)
        batch_credible_intervals = np.percentile(predicted_trips, [2.5, 97.5], axis=0)

        predictions.extend(batch_predictions)
        lower_cis.extend(batch_credible_intervals[0])
        upper_cis.extend(batch_credible_intervals[1])

    data_to_predict['predicted_trips'] = predictions
    data_to_predict['lower_ci'] = lower_cis
    data_to_predict['upper_ci'] = upper_cis

    output_filename = 'predictions.csv'
    output_path = os.path.join(output_folder, output_filename)
    data_to_predict.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"predictions exported to: {output_path}")

    return data_to_predict[['predicted_trips', 'lower_ci', 'upper_ci']]


def save_model_and_parameters(model, trace, output_folder):
    model_file_path = output_folder / 'pymc_model.pkl'
    trace_file_path = output_folder / 'pymc_trace.joblib'

    joblib.dump(model, model_file_path)
    joblib.dump(trace, trace_file_path)

    print(f"Model saved to: {model_file_path}")
    print(f"Trace saved to: {trace_file_path}")


def load_model_and_parameters(output_folder):
    model_file_path = output_folder / 'pymc_model.pkl'
    trace_file_path = output_folder / 'pymc_trace.joblib'

    if model_file_path.exists():
        model = joblib.load(model_file_path)
        print(f"Model loaded from: {model_file_path}")
    else:
        model = None
        print(f"Model file not found at: {model_file_path}")

    if trace_file_path.exists():
        trace = joblib.load(trace_file_path)
        print(f"Trace loaded from: {trace_file_path}")
    else:
        trace = None
        print(f"Trace file not found at: {trace_file_path}")

    return model, trace
