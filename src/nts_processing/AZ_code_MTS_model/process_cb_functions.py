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


def process_cb_data_tfn_method(data, columns_to_keep, output_folder):
    df = pd.read_csv(data)
    df = df[columns_to_keep]
    df.columns = [str(col).strip() for col in df.columns]

    df = df[df['period'] != 0]
    df = df[~df['mode'].isin([8, 0])]
    df = df[~df['purpose'].isin([0, 2, 3, 4, 5, 6, 7, 8])]

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


def generate_priors(data, target_column): #version 2
    print('generating priors')
    start_time = time.time()

    # logging y to see if we can improve predictions
    data[f'log_{target_column}'] = np.log1p(data[target_column])

    with pm.Model() as trip_model:
        # Hyper-priors
        sigma_purpose = pm.HalfCauchy('sigma_purpose', beta=5)
        sigma_tfn_at = pm.HalfCauchy('sigma_tfn_at', beta=5)
        sigma_hh_type = pm.HalfCauchy('sigma_hh_type', beta=5)
        sigma_mode = pm.HalfCauchy('sigma_mode', beta=5)
        sigma_period = pm.HalfCauchy('sigma_period', beta=5)

        # Hierarchical priors
        beta_purpose = pm.Normal('beta_purpose', mu=0, sigma=sigma_purpose, shape=len(data['purpose'].cat.categories))
        beta_tfn_at = pm.Normal('beta_tfn_at', mu=0, sigma=sigma_tfn_at, shape=len(data['tfn_at'].cat.categories))
        beta_hh_type = pm.Normal('beta_hh_type', mu=0, sigma=sigma_hh_type, shape=len(data['hh_type'].cat.categories))
        beta_mode = pm.Normal('beta_mode', mu=0, sigma=sigma_mode, shape=len(data['mode'].cat.categories))
        beta_period = pm.Normal('beta_period', mu=0, sigma=sigma_period, shape=len(data['period'].cat.categories))

        # Intercept
        intercept = pm.Normal('intercept', mu=0, sigma=10)

        # Linear combination
        mu = (intercept +
              beta_purpose[data['purpose'].cat.codes] +
              beta_tfn_at[data['tfn_at'].cat.codes] +
              beta_hh_type[data['hh_type'].cat.codes] +
              beta_mode[data['mode'].cat.codes] +
              beta_period[data['period'].cat.codes])

        # Model error
        sigma = pm.HalfCauchy('sigma', beta=5)
        sigma_heteroscedastic = pm.Deterministic('sigma_heteroscedastic', sigma * (1 + pm.math.exp(-mu))) # changing variability (to help higher trip value)

        # Likelihood using Normal distribution
        y = pm.Normal('y', mu=mu, sigma=sigma_heteroscedastic, observed=data[f'log_{target_column}'])

        # Inference (NUTS method)
        trace = pm.sample(4000, tune=2000, cores=8, return_inferencedata=True, progressbar=True, target_accept=0.9)
        #trace = pm.sample(1000, tune=500, cores=8, return_inferencedata=True, progressbar=True, step=pm.Metropolis())

    end_time = time.time()
    print(f"Sampling took {end_time - start_time:.2f} seconds")

    return trip_model, trace


def predict_model_hierarchical_bayesian_model(data_to_predict,
                                              trip_model,
                                              trace,
                                              target_column,
                                              output_folder):
    #version 2
    print('Making predictions')

    with trip_model:
        # Make predictions
        posterior_pred = pm.sample_posterior_predictive(trace, progressbar=True)

    # Extract predictions
    pred_data = posterior_pred.posterior_predictive.y.values
    print("Shape of pred_data:", pred_data.shape)

    # Calculate stats across samples
    predictions_log = pred_data.mean(axis=(0, 1))
    lower_ci_log = np.percentile(pred_data, 2.5, axis=(0, 1))
    upper_ci_log = np.percentile(pred_data, 97.5, axis=(0, 1))

    print("Minimum log prediction:", predictions_log.min())
    print("Maximum log prediction:", predictions_log.max())

    # Transform predictions back to original scale
    #predictions = np.expm1(predictions_log)
    #lower_ci = np.expm1(lower_ci_log)
    #upper_ci = np.expm1(upper_ci_log)
    predictions = np.exp(predictions_log) - 1
    lower_ci = np.exp(lower_ci_log) - 1
    upper_ci = np.exp(upper_ci_log) - 1


    # Check for negative values
    neg_pred = np.sum(predictions < 0)
    neg_lower = np.sum(lower_ci < 0)
    neg_upper = np.sum(upper_ci < 0)

    print(f"Number of negative predictions: {neg_pred}")
    print(f"Number of negative lower CI: {neg_lower}")
    print(f"Number of negative upper CI: {neg_upper}")

    # Ensure non-negative predictions
    predictions = np.maximum(predictions, 0)
    lower_ci = np.maximum(lower_ci, 0)
    upper_ci = np.maximum(upper_ci, 0)

    print("Shape of predictions:", predictions.shape)
    print("Length of data_to_predict:", len(data_to_predict))

    if len(predictions) != len(data_to_predict):
        raise ValueError(
            f"Mismatch in prediction length: got {len(predictions)} predictions for {len(data_to_predict)} data points"
        )

    data_to_predict[f'predicted_{target_column}'] = predictions
    data_to_predict['lower_ci'] = lower_ci
    data_to_predict['upper_ci'] = upper_ci

    # Add log pred. for evaluation
    data_to_predict[f'predicted_log_{target_column}'] = np.log1p(predictions)

    output_filename = 'predictions.csv'
    output_path = os.path.join(output_folder, output_filename)
    data_to_predict.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"Predictions exported to: {output_path}")

    return data_to_predict


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

    output_filename = 'data_to_model_reduced_size.csv'
    output_path = os.path.join(output_folder, output_filename)
    sampled_df.to_csv(output_path, index=False)
    print('-------------------------------------------------------------')
    print(f"Reduced size data exported to: {output_path}")

    return sampled_df


def evaluate_model(y_true, y_pred_log, output_folder):
    y_true_log = np.log1p(y_true)
    y_pred = np.exp(y_pred_log) - 1
    y_pred = np.maximum(y_pred, 0)


    metrics_dict = {
        'mse_log': mean_squared_error(y_true_log, y_pred_log),
        'mae_log': mean_absolute_error(y_true_log, y_pred_log),
        'r2_log': r2_score(y_true_log, y_pred_log),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(output_folder, 'model_evaluation_metrics.csv'), index=False)

    plt.figure(figsize=(20, 15))

    # Predicted vs Actual (log)
    plt.subplot(2, 2, 1)
    plt.scatter(y_true_log, y_pred_log, alpha=0.5)
    plt.plot([y_true_log.min(), y_true_log.max()], [y_true_log.min(), y_true_log.max()], 'r--')
    plt.xlabel('Actual Log Total Trips')
    plt.ylabel('Predicted Log Total Trips')
    plt.title('Predicted vs Actual (Log Space)')

    # Predicted vs Actual (non log)
    plt.subplot(2, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Total Trips')
    plt.ylabel('Predicted Total Trips')
    plt.title('Predicted vs Actual (Original Space)')
    plt.xscale('log')
    plt.yscale('log')

    # Residuals plot (log)
    plt.subplot(2, 2, 3)
    residuals_log = y_true_log - y_pred_log
    plt.scatter(y_pred_log, residuals_log, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Log Total Trips')
    plt.ylabel('Residuals (Log Space)')
    plt.title('Residuals Plot (Log Space)')

    # Residuals plot (non log)
    plt.subplot(2, 2, 4)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Total Trips')
    plt.ylabel('Residuals (Original Space)')
    plt.title('Residuals Plot (Original Space)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'diagnostic_plots.png'))
    plt.close()

    # Histograms
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_true, bins=50, alpha=0.5)
    plt.title('Histogram of Actual Total Trips')
    plt.xlabel('Total Trips')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(y_true_log, bins=50, alpha=0.5)
    plt.title('Histogram of Log Actual Total Trips')
    plt.xlabel('Log Total Trips')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'data_distribution.png'))
    plt.close()


    # prediction errors
    plt.figure(figsize=(10, 6))
    error_ratio = np.abs(y_pred - y_true) / y_true
    plt.scatter(y_true, error_ratio, alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Actual Total Trips')
    plt.ylabel('Relative Error')
    plt.title('Relative Prediction Error vs Actual Total Trips')
    plt.savefig(os.path.join(output_folder, 'relative_error_plot.png'))
    plt.close()

    return metrics_df
