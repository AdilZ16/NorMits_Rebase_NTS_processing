# -*- coding: utf-8 -*-
"""
Created on: 7/10/2024
Original author: Adil Zaheer
"""
import multiprocessing
import os
import pickle
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import sys
from typing import Union, List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from caf.ml.functions import data_pipeline_functions as dpf
from caf.ml.functions import process_data_functions as pdf
from src.nts_processing.processing_classified_build import process_cb_functions as mts_process


class TripRate:
    """
    Trip rate class: used to store and call the data processing functions.
    """
    def __init__(self, data, mode, geo_incl, segments_incl, columns_to_keep, output_folder, purpose_value):
        self.data = data
        self.mode = mode
        self.geo_incl = geo_incl
        self.segments_incl = segments_incl
        self.columns_to_keep = columns_to_keep
        self.output_folder = output_folder
        self.nts_dtype = int
        self.tfn_mode = [1, 2, 3, 4, 5, 6, 7, 8]
        self.tfn_ttype = None
        self.purpose_value = purpose_value
        print('Trip rate class is running')

    def nhb_production_weights_production(self):
        return nhb_production_weights_production(
            dfr=self.data,
            mode=self.mode,
            geo_incl=self.geo_incl,
            seg_incl=self.segments_incl,
            tfn_mode=self.tfn_mode,
            tfn_ttype=self.tfn_ttype,
            columns_to_keep=self.columns_to_keep,
            output_folder=self.output_folder,
        )

    def process_cb_data_tfn_method(self):
        return mts_process.process_cb_data_tfn_method(data=self.data,
                                                      columns_to_keep=self.columns_to_keep,
                                                      output_folder=self.output_folder,
                                                      purpose_value=self.purpose_value)


def nhb_production_weights_production(dfr, mode, geo_incl, seg_incl, tfn_mode, tfn_ttype, columns_to_keep, output_folder):
    """
    Arguments below are included to keep previous iterations code functional. See mdl.triprate:
    :param dfr: data to process
    :param mode: specify which mode to include from the classified build. [Default None]
    :param geo_incl: specify which geography to include from the classified build. [Default = tfn_at]
    :param seg_incl: specify which segmentation to include from the classified build. [Default = None]
    :param tfn_mode: hardcoded argument, linked to previous methodology. Ignore.
    :param tfn_ttype: hardcoded argument, linked to previous methodology. Ignore.



    :param columns_to_keep: which columns in the data you want to keep. Columns not included will
                            still be kept, this function ensures that regardless of any data
                            processing, the column will be kept.
    :param output_folder:   path to output folder. Where you want model outputs to go.
    :return:
    """
    dfr = pd.read_csv(dfr)
    dfr = dfr[columns_to_keep]

    # 1: Data Prep
    dfr, mode, lev_prod, lev_dest, seg_incl, col_used = prepare_data(dfr=dfr,
                                                                     mode=mode,
                                                                     geo_incl=geo_incl,
                                                                     seg_incl=seg_incl,
                                                                     tfn_ttype=tfn_ttype,
                                                                     tfn_mode=tfn_mode)

    # 2: Separate hb & nhb trips
    hbf, nhb = separate_trips(dfr=dfr)

    # 3: Merge and aggregate trips
    nhb, hbf, col_join, col_u4hb = merge_and_aggregate_trips(hbf=hbf,
                                                             nhb=nhb,
                                                             seg_incl=seg_incl)

    # 4: Filter and complete data
    nhb, df_post_processing = filter_and_complete(nhb, mode, col_join, col_u4hb, hbf, output_folder)

    # 5: Calculate Initial Gamma
    nhb = calculate_initial_gamma(nhb)

    # 6: Define at aggregation
    atx_type = define_atx_type()

    # 7: at aggregation and gamma calc
    nhb = area_type_aggregation(nhb, atx_type, col_join)

    # 8: Final processing
    nhb = final_processing_and_output(nhb, lev_dest, lev_prod, geo_incl, output_folder=output_folder)

    return nhb, df_post_processing


def custom_loss(y_true, y_pred, weight=5):
    """
    :param y_true: Recorded y values
    :param y_pred: Predicted y values
    :param weight: increase importance of ratio penalty relative to mse in the
                    full loss calc. Lower weight = model focuses more on minimising
                    mse and less on keeping target ratio (1-10 range makes sense, could
                    look to hyperparam tune this)
    :return: return final loss (mse + weighted ratio penalty term)

    This function works in a very similar way to the custom_loss_mts function. The main
    noteable difference is the fact the nhb ratio is one fixed value as opposed to the
    dictionary used in custom_loss_mts.

    """
    # calculate mean squared error between true and predict
    mse = np.mean((y_true - y_pred)**2)
    # calculate ratio of sum of pred to sum of true
    nhb_ratio = np.sum(y_pred) / np.sum(y_true)
    # penalty calc (how far nhb-ratio is from 22.5%)
    ratio_penalty = (nhb_ratio - 0.225)**2
    return mse + weight * ratio_penalty


def custom_loss_mts(y_true, y_pred, mode_period_values, weight=5, large_value_weight=2):
    """
    This function is used by Sci-Kit Learn behind the scenes. No user interaction required other than
    altering the weight values. 5 and 2 are default.

    Mode_period_grid consists of a dictionary with a grid of target ratios for mode period
    combinations. The string value that corresponds to the ratio is in the same format as the
    mode_period column in the dataframe that the modelling functions receive (see df_total in your
    output folder).

    The function works by first calculating the mean squared error between true
    and predicted values. Inside the loop, the mode period value for the iteration and corresponding
    target value is found. The ratio of that iteration is calculated. The squared difference between
    actual and target ratios is then calculated as a penalty term (added to total penalty).

    For large y values (top 20%), an additional penalty term is calculated. The average large and
    regular penalty value is then calculated to be included in a loss function. The loss combines
    the mean squared error, weighted average penalty and large penalty.

    The overall aim of the function is to guide the model towards predictions that minimise the mean
    squared error as well as maintains the target ratios for each specific mode period combination.

    :param y_true: Recorded y values
    :param y_pred: Predicted y values
    :param mode_period_values: A list of mode-period combinations ('1_1', '1_2'...) corresponding
                               to each row in y_true and y_pred.
    :param weight: increase importance of ratio penalty relative to mse in the
                    full loss calc. Lower weight = model focuses more on minimising
                    mse and less on keeping target ratio (1-10 range makes sense, could
                    look to hyperparam tune this)
    :param large_value_weight: same as weight argument but only used for large values (less accurate
                               predictions).
    :return: return final loss (mse + weighted ratio penalty term)
    """
    mode_period_grid = {
        '1_1': 0.0773, '1_2': 0.0755, '1_3': 0.0258, '1_4': 0.0179, '1_5': 0.0302, '1_6': 0.0275,
        '2_1': 0.0063, '2_2': 0.0039, '2_3': 0.0016, '2_4': 0.0021, '2_5': 0.0019, '2_6': 0.0017,
        '3_1': 0.1739, '3_2': 0.1502, '3_3': 0.0644, '3_4': 0.0596, '3_5': 0.0914, '3_6': 0.0722,
        '4_1': 0.0063, '4_2': 0.0021, '4_3': 0.0012, '4_4': 0.0028, '4_5': 0.0020, '4_6': 0.0014,
        '5_1': 0.0292, '5_2': 0.0216, '5_3': 0.0030, '5_4': 0.0042, '5_5': 0.0087, '5_6': 0.0038,
        '6_1': 0.0096, '6_2': 0.0029, '6_3': 0.0007, '6_4': 0.0030, '6_5': 0.0021, '6_6': 0.0008,
        '7_1': 0.0057, '7_2': 0.0019, '7_3': 0.0006, '7_4': 0.0012, '7_5': 0.0012, '7_6': 0.0007
    }

# todo run for all purposes, check for overfitting

    mse = np.mean((y_true - y_pred) ** 2)
    total_penalty = 0
    large_value_penalty = 0
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        mode_period = mode_period_values[i]
        target_ratio = mode_period_grid[mode_period]
        # to stop dividing by 0 error
        actual_ratio = y_p / y_t if y_t != 0 else 0
        penalty = (actual_ratio - target_ratio) ** 2
        total_penalty += penalty

        if y_t > np.percentile(y_true, 80):
            large_value_penalty += (y_t - y_p)**2 * large_value_weight

    avg_penalty = total_penalty / len(y_true)
    avg_large_penalty_value = large_value_penalty / len(y_true)
    loss = mse + weight * avg_penalty + avg_large_penalty_value

    return loss


def tune_model(X, y, custom_scorer):
    """
    :param X: x variables (explanatory)
    :param y: target variable
    :param custom_scorer: Ignore argument, this is the custom loss function that the model uses
                          for predictions
    :return: returns best model found in the search

    Fairly simple model set up function that finds the best model possible across 10 folds considering
    the hyperparamters found in param_grid. The model utilises the custom loss function, which depends
    on if mts or production weights modelling is taking place.

    """

    # hyper param. grid
    param_grid = {'n_estimators': [300, 400, 500],
                  'max_depth': [5, 7, 9],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [2, 4],
                  'learning_rate': [0.05, 0.1, 0.2],
                  'subsample': [0.8, 0.9, 1.0],
                  }

    # cross validation method
    cv = RepeatedKFold(n_splits=10, random_state=42)

    # make it faster (uses all cores available)
    n_cores = multiprocessing.cpu_count()
    print(f"Using {n_cores} CPU cores")


    print("Starting model tuning...")
    start_time = time.time()

    gb = GradientBoostingRegressor(random_state=42)
    randomised_search = RandomizedSearchCV(estimator=gb,
                                           param_distributions=param_grid,
                                           n_iter=100,
                                           cv=cv,
                                           scoring=custom_scorer,
                                           n_jobs=n_cores,
                                           verbose=2,
                                           random_state=42)

    randomised_search.fit(X, y)

    end_time = time.time()
    print(f"Model tuning completed in {end_time - start_time:.2f} seconds")
    print("Best parameters:", randomised_search.best_params_)

    return randomised_search.best_estimator_


def generate_missing_rows(purpose_value, input_data, target_column):
    """
    :param purpose_value: value from 1 to 8 to dictate which purpose is being modelling
    :param input_data: data that the missing data is being added to
    :param target_column: this is the column we are trying to predict
    :return: original data with the additional rows

    This function begins with creating all possible combinations of data from the classified
    build. This is then combined with the original data in order to ensure all possible rows
    are present in the dataframe to be modelled.

    """
    # Generate all possible combos
    all_combinations = pd.MultiIndex.from_product([
        range(1, 21),  # tfn_at
        range(1, 9),  # hh_type
        [purpose_value], # purpose
        range(1, 8),  # mode
        range(1, 7)  # period
    ], names=['tfn_at', 'hh_type', 'purpose', 'mode', 'period'])

    all_data = pd.DataFrame(index=all_combinations).reset_index()
    all_data['mode_period'] = all_data['mode'].astype(str) + '_' + all_data['period'].astype(str)

    merged_data = pd.merge(all_data, input_data,
                           on=['tfn_at', 'hh_type', 'purpose', 'mode', 'period'],
                           how='left')

    merged_data['rows_added'] = merged_data[target_column].isna()

    return merged_data


def model_to_calculate_gamma(nhb,
                             output_folder,
                             target_column,
                             numerical_features,
                             categorical_features,
                             index_columns,
                             drop_columns,
                             ignore_columns,
                             purpose_value,
                             production_weight_calculation,
                             mts_calculation):
    """
    :param nhb: data to model
    :param output_folder: path to output folder. Where you want model outputs to go.
    :param target_column: this is the column we are trying to predict
    :param numerical_features: x variables that are numerical
    :param categorical_features: x variables that are categorical
    :param index_columns: any columns to index (not relevant for modelling but relevant for data structure / analysis)
    :param drop_columns: any columns to drop
    :param ignore_columns: any columns that aren't relevant for modelling but shouldn't be indexed or dropped
    :param purpose_value: value from 1 to 8 to dictate which purpose is being modelling
    :param production_weight_calculation: None or string, if data needs to be processed for production weight modelling. Must be string if data_skip_cb_generation is None
    :param mts_calculation: None or string, if data needs to be processed for mts modelling. Must be string if data_skip_cb_generation is None
    :return: final dataframe which contains all the predicted results overlay onto the original dataframe format


    This is the main function that dictates the modelling process from start to finish. Prior to this
    point data processing is the only steps covered. Indexing, generating missing rows, and transforming
    the data all initial happen (data transformation includes scaling and encoding x variables).
    Y is then logged and left in its original scale. Training and tests splits are created based on
    this. The custom loss function is then created depending on if mts or production weight modelling is
    being done.

    The model is then initialised and predictions on both log and non log scales are made. The rest of
    the function conducts general model scoring and feature evaluation testing. All results are
    output to the output_folder.
    """
    start_time = time.time()

    # data processing
    original_data = nhb.copy()

    # make feature list to use
    if ignore_columns is None:
        features_to_use = nhb.columns.tolist()
    else:
        features_to_use = [col for col in nhb.columns if col not in ignore_columns]

    nhb_to_model = nhb[features_to_use]

    # caf.ml index columns function
    if index_columns is not None:
        nhb_to_model = pdf.index_sorter(df=nhb_to_model, index_columns=index_columns, drop_columns=drop_columns)

    if 'trips.hb' not in nhb_to_model.columns:
        nhb_to_model_final = generate_missing_rows(purpose_value=purpose_value,
                                                   input_data=nhb_to_model,
                                                   target_column=target_column)
    else:
        nhb_to_model_final = nhb_to_model
    print(nhb_to_model_final)
    # caf.ml data transformations (scale & encoding) function
    data_to_model, transformations = dpf.process_data_pipeline(df=nhb_to_model_final,
                                                               numerical_features=numerical_features,
                                                               categorical_features=categorical_features,
                                                               target_column=target_column,
                                                               output_folder=output_folder)

    data_for_training = data_to_model.dropna(subset=[target_column])
    y_log = np.log1p(data_for_training[target_column])
    y_non_log = data_for_training[target_column]
    x = data_for_training.drop(columns=[target_column])


    model_filename = os.path.join(output_folder, 'trained_model.pkl')

    train_index, test_index = train_test_split(range(len(x)), test_size=0.2, random_state=42)
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train_log, y_test_log = y_log.iloc[train_index], y_log.iloc[test_index]


    custom_scorer_loss_func = None
    if production_weight_calculation is not None:
        # make_scorer lets the loss function work with the model
        print("Using custom_loss scorer")
        custom_scorer_loss_func = make_scorer(custom_loss, greater_is_better=False)

    elif mts_calculation is not None:
        mode_period_values = nhb_to_model_final['mode_period'].tolist()
        print("Using custom_loss_mts scorer")
        custom_scorer_loss_func = make_scorer(lambda y_true, y_pred: custom_loss_mts(y_true, y_pred, mode_period_values),
        greater_is_better=False)

    # save/load model
    if os.path.exists(model_filename):
        print(f"Loading pre-trained model: {model_filename}")
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    else:
        model = tune_model(X_train, y_train_log, custom_scorer=custom_scorer_loss_func)
        model.fit(X_train, y_train_log)
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {model_filename}")


    # cross-validation for evaluation
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, scoring='r2')
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean R2 score: {cv_scores.mean()}")


    ## PREDICTION ##
    # LOG + FORCE POSITIVE
    X_all = data_to_model.drop(columns=[target_column])
    y_pred_non_log = np.exp(model.predict(X_all))
    y_pred_log = model.predict(X_all)
    y_pred_non_log_excluding_generated_rows = np.exp(model.predict(x))
    y_pred_log_excluding_generated_rows = model.predict(x)

    mode_period_values = nhb_to_model_final['mode_period'].tolist()

    if 'trips.hb' in data_to_model.columns:
        # gamma from predictions (non-log)
        gamma_pred = y_pred_non_log / original_data['trips.hb']
        final_df = nhb_to_model_final.copy()
        final_df['predicted_trips'] = y_pred_non_log
        final_df['predicted_gamma'] = gamma_pred
    else:
        final_df = nhb_to_model_final.copy()
        final_df['predicted_total_trips'] = y_pred_non_log
        final_df['predicted_total_trips_log'] = y_pred_log


    # Feature importance (for extra evaluation)
    if isinstance(model, GradientBoostingRegressor) and hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({'feature': x.columns, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(os.path.join(output_folder, 'feature_importance.csv'), index=True)

    final_df.to_csv(os.path.join(output_folder, 'final_predictions.csv'), index=True)


    metrics_log = {'mse_log': mean_squared_error(y_log,
                                                 y_pred_log_excluding_generated_rows),
                   'rmse_log': np.sqrt(mean_squared_error(y_log,
                                                          y_pred_log_excluding_generated_rows)),
                   'mae_log': mean_absolute_error(y_log,
                                                  y_pred_log_excluding_generated_rows),
                   'r2_log': r2_score(y_log,
                                      y_pred_log_excluding_generated_rows)}


    metrics_non_log = {'mse_non_log': mean_squared_error(y_non_log,
                                                         y_pred_non_log_excluding_generated_rows),
                       'rmse_non_log': np.sqrt(mean_squared_error(y_non_log,
                                                                  y_pred_non_log_excluding_generated_rows)),
                       'mae_non_log': mean_absolute_error(y_non_log,
                                                          y_pred_non_log_excluding_generated_rows),
                       'r2_non_log': r2_score(y_non_log,
                                              y_pred_non_log_excluding_generated_rows)}


    metrics_dict = {**metrics_log, **metrics_non_log}
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(output_folder, 'model_evaluation_metrics.csv'), index=True)

    print(f"Results saved to {output_folder}")
    end_time = time.time()
    print(f"Total run time: {end_time - start_time:.2f} seconds")


    if os.path.exists(model_filename):
        return final_df
    else:
        model_filename = os.path.join(output_folder, 'trained_model.joblib')
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

    return final_df


#################### functions #######################

# functions below are from Nhan's code
def prepare_data(dfr, mode, geo_incl, seg_incl, tfn_ttype, tfn_mode):
    print('preparing data')
    lev_prod, lev_dest = ["tfn_at"], ["tfn_at_d"]
    seg_incl = val_to_list(seg_incl) if seg_incl is not None else []
    col_used = ["mode", "purpose"]
    seg_incl = col_used + (lev_dest if geo_incl is not None else []) + seg_incl
    dfr = dfr[seg_incl + ["direction", "period", "tour_group", "individualid", "trips"]].copy()
    mode = tfn_mode if mode is None else mode
    return dfr, mode, lev_prod, lev_dest, seg_incl, col_used


def separate_trips(dfr):
    print('separating trips')
    hbf = dfr.loc[dfr["direction"].isin(["hb_fr"])].reset_index(drop=True)
    nhb = dfr.loc[dfr["direction"].isin(["nhb"])].reset_index(drop=True)
    return hbf, nhb


def merge_and_aggregate_trips(hbf, nhb, seg_incl):
    print('merge and aggregating trips')
    col_join = ["individualid", "tour_group"]
    nhb = pd.merge(nhb, hbf, how="left", on=col_join, suffixes=("", ".hb")).fillna(0)
    col_u4hb = [f"{col}.hb" if col in ["mode", "purpose"] else col for col in seg_incl]
    hbf = hbf.groupby(seg_incl)[["trips"]].sum().reset_index()
    hbf.rename(columns=dict(zip(seg_incl, col_u4hb)), inplace=True)
    col_join = seg_incl + [col for col in col_u4hb if col not in seg_incl]
    nhb = nhb.groupby(col_join)[["trips"]].sum().reset_index()
    return nhb, hbf, col_join, col_u4hb


def filter_and_complete(nhb, mode, col_join, col_u4hb, hbf, output_folder):
    print('filtering final dataframe')
    nhb = dfr_filter_zero(nhb, col_join)
    nhb = dfr_filter_mode(nhb, mode, "mode.hb")
    nhb = dfr_filter_mode(nhb, mode)
    nhb = dfr_complete(nhb, col_join, "mode.hb").reset_index()
    nhb = pd.merge(nhb, hbf, how="left", on=col_u4hb, suffixes=("", ".hb"))
    df_post_processing = nhb.copy()

    print('outputting df_post_processing')
    df_post_processing.to_csv(os.path.join(output_folder, 'df_post_processing.csv'), index=False)

    return nhb, df_post_processing


def calculate_initial_gamma(nhb, min_trip=500):
    print('calculating initial gamma')
    nhb['tfn_at'] = nhb['tfn_at_d']
    msk = (nhb['trips.hb'] > min_trip)
    nhb.loc[msk, "gamma"] = nhb["trips"].div(nhb["trips.hb"]).loc[msk]
    return nhb


def define_atx_type():
    atx_type = {'lv1': {1: [1, 2], 2: [3, 4, 5], 3: [6, 7, 8], 4: [9, 10, 11], 5: [12, 13, 14, 15],
                        6: [16, 17], 7: [18, 19], 8: [20]},
                'lv2': {1: [1, 2, 3], 2: [4, 5, 6, 7, 8, 20], 3: [9, 10, 11, 12, 13, 14, 15],
                        4: [16, 17, 18, 19]},
                'lv3': {1: [1, 2, 3, 4, 5, 6, 7, 8, 20], 2: [9, 10, 11, 12, 13, 14, 15],
                        3: [16, 17, 18, 19, 20]}}
    return atx_type


def area_type_aggregation(nhb, atx_type, col_join, min_trip=500):
    print('calculating gamma using aggregation where required')
    for key, itm in atx_type.items():
        nhb['tfn_at_d'] = nhb['tfn_at']
        nhb = nhb.set_index('tfn_at_d').rename(index=itm_to_key(itm)).reset_index()
        nhb['trips_agg'] = nhb.groupby(col_join)['trips'].transform('sum')
        nhb['trips.hb_agg'] = nhb.groupby(col_join)['trips.hb'].transform('sum')
        msk = (nhb['trips.hb_agg'] > min_trip) & nhb['gamma'].isna()
        nhb.loc[msk, "gamma"] = nhb["trips_agg"].div(nhb["trips.hb_agg"]).loc[msk]
    msk = nhb['gamma'].isna()
    nhb.loc[msk, 'gamma'] = nhb["trips_agg"].div(nhb["trips.hb_agg"]).loc[msk]
    nhb = nhb.drop(columns=['tfn_at_d', 'trips_agg', 'trips.hb_agg'])
    return nhb


def final_processing_and_output(nhb, lev_dest, lev_prod, geo_incl, output_folder):
    nhb = (nhb.rename(columns={key: itm for key, itm in zip(lev_dest, lev_prod)})
           if geo_incl is not None else nhb)

    if output_folder is not None:
        print('outputting nhb data')

        nhb.to_csv(os.path.join(output_folder, 'nhb_data_with_gamma.csv'), index=False)
    return nhb


############################## functions needed to run code above (also from Nhan) ################################

def val_to_list(str_text: Union[str, float, int, List]) -> List:
    return [str_text] if not isinstance(str_text, list) else str_text


def dfr_filter_zero(dfr: pd.DataFrame, col_used: Union[List, str]) -> pd.DataFrame:
    col_used = [col_used] if isinstance(col_used, str) else col_used
    return dfr.loc[(~dfr[col_used].isin([0, "0", -8, -9, -10])).all(axis=1)].reset_index(drop=True)


# filter mode values
def dfr_filter_mode(
        dfr: pd.DataFrame, inc_list: List, col_mode: str = "mode"
) -> pd.DataFrame:
    return dfr.loc[dfr[col_mode].isin(inc_list)].reset_index(drop=True)


# create a complete set of index values
def dfr_complete(
        dfr: pd.DataFrame, col_index: Union[List, str, None], col_unstk: Union[List, str]
) -> pd.DataFrame:
    col_index = [] if col_index is None else val_to_list(col_index)
    col_unstk = val_to_list(col_unstk)
    dfr = dfr.set_index(col_index) if len(col_index) > 0 else dfr
    for col in col_unstk:
        dfr = dfr.unstack(level=col, fill_value=0).stack(future_stack=True)  # future_stack implemented in pandas 2.1
    return dfr


def itm_to_key(dct: Dict, key_lower: bool = True) -> Dict:
    # swap key to value
    dct = {key: [dct[key]] if not isinstance(dct[key], list) else dct[key] for key in dct}
    return {str_lower(val) if key_lower else val: key for key in dct for val in dct[key]}


def log_stderr(*args):
    print(*args, file=sys.stderr, flush=True)


def str_lower(val: Any) -> Any:
    if isinstance(val, str):
        return val.lower()
    elif isinstance(val, (tuple, list, dict, set)):
        return tuple(itm.lower() if isinstance(itm, str) else itm for itm in val)
    else:
        return val
