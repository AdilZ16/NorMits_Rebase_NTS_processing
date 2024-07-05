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


def process_data_for_hierarchical_bayesian_model(df, columns_to_keep, reduce_data_size, target_column, output_folder):
    print('processing data to model specifications')

    if reduce_data_size is not None:
        print('Reducing data size')
        df = create_hybrid_sample(df, output_folder=output_folder)

    for var in columns_to_keep:
        # unique values and create new categorical type
        unique_values = df[var].unique()
        df[var] = pd.Categorical(df[var], categories=unique_values)

    predict_data = df[df['split_method'] == 'TBD'].copy()
    training_data = df[df['split_method'] != 'TBD'].copy()

    output_filename = 'predict_data.csv'
    output_path = os.path.join(output_folder, output_filename)
    predict_data.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"predict_data exported to: {output_path}")

    output_filename = 'training_data.csv'
    output_path = os.path.join(output_folder, output_filename)
    training_data.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"training_data exported to: {output_path}")

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


'''def predict_model_hierarchical_bayesian_model(data_to_predict, trip_model, trace, target_column, output_folder):
    print('Making predictions')


    with trip_model:
        # make predictions
        pm.set_data({'trips_data': data_to_predict['trips']})
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=[target_column])

    print(type(posterior_pred))
    print(dir(posterior_pred))

    # Extract predictions
    pred_data = posterior_pred.posterior_predictive[target_column].values
    print("Shape of pred_data:", pred_data.shape)

    # Calculate stats across samples
    predictions = pred_data.mean(axis=(0, 1))
    lower_ci = np.percentile(pred_data, 2.5, axis=(0, 1))
    upper_ci = np.percentile(pred_data, 97.5, axis=(0, 1))

    print("Shape of predictions:", predictions.shape)
    print("Length of data_to_predict:", len(data_to_predict))

    if len(predictions) != len(data_to_predict):
        raise ValueError(
            f"Mismatch in prediction length: got {len(predictions)} predictions for {len(data_to_predict)} data points"
        )

    data_to_predict[f'predicted_{target_column}'] = predictions
    data_to_predict['lower_ci'] = lower_ci
    data_to_predict['upper_ci'] = upper_ci

    output_filename = 'predictions.csv'
    output_path = os.path.join(output_folder, output_filename)
    data_to_predict.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"Predictions exported to: {output_path}")

    return data_to_predict'''


def predict_model_hierarchical_bayesian_model(data_to_predict, trip_model, trace,
                                              target_column, output_folder):


    with trip_model:
        print('Model variables:', trip_model.named_vars.keys())
        print("Keys in the trace object:")
        print(trace.keys())
        print("\nSummary of the trace object:")
        print(trace)

        print('Updating model data context for prediction')
        pm.set_data({
            'trips_data': data_to_predict['trips'],

            'sigma_purpose': trace['sigma_purpose'].mean(),
            'beta_purpose': trace['beta_purpose'].mean(axis=0)[data_to_predict['purpose'].cat.codes],

            'sigma_tfn_at': trace['sigma_tfn_at'].mean(),
            'beta_tfn_at': trace['beta_tfn_at'].mean(axis=0)[data_to_predict['tfn_at'].cat.codes],

            'sigma_hh_type': trace['sigma_hh_type'].mean(),
            'beta_hh_type': trace['beta_hh_type'].mean(axis=0)[data_to_predict['hh_type'].cat.codes],

            'sigma_mode': trace['sigma_mode'].mean(),
            'beta_mode': trace['beta_mode'].mean(axis=0)[data_to_predict['mode'].cat.codes],

            'sigma_period': trace['sigma_period'].mean(),
            'beta_period': trace['beta_period'].mean(axis=0)[data_to_predict['period'].cat.codes],
        })

        # Debug
        print("Shape of data_to_predict:", data_to_predict.shape)
        print("Unique values in purpose:", data_to_predict['purpose'].unique())
        print("Unique values in tfn_at:", data_to_predict['tfn_at'].unique())
        print("Unique values in hh_type:", data_to_predict['hh_type'].unique())
        print("Unique values in mode:", data_to_predict['mode'].unique())
        print("Unique values in period:", data_to_predict['period'].unique())

        # Generate posterior predictive samples
        print('Generating posterior predictive samples')
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=[target_column])

        predictions = posterior_pred[target_column].mean(axis=0)

        print("Shape of posterior_pred:", posterior_pred[target_column].shape)
        print("Shape of predictions:", predictions.shape)
        print("Length of data_to_predict:", len(data_to_predict))

        if len(predictions) != len(data_to_predict):
            raise ValueError(
                f"Mismatch in prediction length: got {len(predictions)} predictions for {len(data_to_predict)} data points"
            )

        # (initial model scoring)
        hpd_intervals = pm.stats.hdi(posterior_pred, hdi_prob=0.94)
        credible_intervals = hpd_intervals[target_column]

    output_filename = 'predictions.csv'
    output_path = os.path.join(output_folder, output_filename)
    data_to_predict.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"Predictions exported to: {output_path}")

    return predictions, credible_intervals


def save_model_and_parameters(model, trace, output_folder):
    model_file_path = output_folder / 'pymc_model.pkl'
    trace_file_path = output_folder / 'pymc_trace.joblib'

    with model_file_path.open('wb') as model_file:
        dill.dump(model, model_file)

    with trace_file_path.open('wb') as trace_file:
        dill.dump(trace, trace_file)

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


def create_stratified_sample(df, columns_to_keep, target_column):
    #todo takes too long to run - look into this, method is better than quiker way
    print('Dataframe size before reduction')
    print(df.shape)
    sample_size = 0.02
    stratified_samples = []

    stratify_columns = [col for col in columns_to_keep if col != target_column]

    for column in stratify_columns:
        unique_vals = df[column].unique()
        samples = []
        for val in unique_vals:
            # Filter rows for each unique value in the column
            subset = df[df[column] == val]
            if len(subset) > 1:
                # Sample proportionally from each subset
                sample = subset.sample(frac=sample_size, random_state=42)
                samples.append(sample)
            else:
                samples.append(subset)

        stratified_samples.append(pd.concat(samples))

    combined_sample = pd.concat(stratified_samples).drop_duplicates().reset_index(drop=True)

    combined_sample = combined_sample[columns_to_keep]

    print('Dataframe size post reduction')
    print(combined_sample.shape)
    print(combined_sample['trips'].describe())

    return combined_sample


def create_hybrid_sample(df, output_folder):
    print('Dataframe size before reduction')
    print(df.shape)
    sample_fraction = 0.01

    # Random sampling
    sampled_df = df.sample(frac=sample_fraction, random_state=42)

    # Ensuring 'tbd' is in the data, these are the rows we will predict
    important_value = 'TBD'
    tbd_count = sampled_df[sampled_df['split_method'] == important_value].shape[0]

    if tbd_count < 1000:
        additional_needed = 1000 - tbd_count
        additional_sample = df[df['split_method'] == important_value].sample(n=additional_needed, random_state=42)
        sampled_df = pd.concat([sampled_df, additional_sample], ignore_index=True).reset_index(drop=True)

    print('Dataframe size post reduction')
    print(sampled_df.shape)
    print(sampled_df['trips'].describe())


    output_filename = 'data_to_model_reduced_size.csv'
    output_path = os.path.join(output_folder, output_filename)
    sampled_df.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"Reduced size data exported to: {output_path}")


    return sampled_df
