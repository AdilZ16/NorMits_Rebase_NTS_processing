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
from src.nts_processing.AZ_code_MTS_model import process_cb_functions as mts_process


class TripRate:
    def __init__(self, data, mode, geo_incl, segments_incl, columns_to_keep, output_folder):
        self.data = data
        self.mode = mode
        self.geo_incl = geo_incl
        self.segments_incl = segments_incl
        self.columns_to_keep = columns_to_keep
        self.output_folder = output_folder
        self.nts_dtype = int
        self.tfn_mode = [1, 2, 3, 4, 5, 6, 7, 8]
        self.tfn_ttype = None
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
            output_folder=self.output_folder
        )

    def process_cb_data_tfn_method(self):
        return mts_process.process_cb_data_tfn_method(data=self.data,
                                                      columns_to_keep=self.columns_to_keep,
                                                      output_folder=self.output_folder)


def nhb_production_weights_production(dfr, mode, geo_incl, seg_incl, tfn_mode, tfn_ttype, columns_to_keep, output_folder):

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
    '''
    :param y_true: Recorded y values
    :param y_pred: Predicted y values
    :param weight: increase importance of ratio penalty relative to mse in the
                    full loss calc. Lower weight = model focuses more on minimising
                    mse and less on keeping target ratio (1-10 range makes sense, could
                    look to hyperparam tune this)
    :return: return final loss (mse + weighted ratio penalty term)
    '''
    # todo hyperparam optim for weight variable / change the ratio percentage based on our knowledge
    # calculate mean squared error between true and predict
    mse = np.mean((y_true - y_pred)**2)
    # calculate ratio of sum of pred to sum of true
    nhb_ratio = np.sum(y_pred) / np.sum(y_true)
    # penalty calc (how far nhb-ratio is from 22.5%)
    ratio_penalty = (nhb_ratio - 0.225)**2
    return mse + weight * ratio_penalty


def tune_model(X, y):
    '''
    :param X: x variables (all categorical other than one continuous)
    :param y: target variable (continuous)
    :return: returns best model found in the search
    '''


    # hyper param. grid
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
    }

    # cross validation method
    cv = RepeatedKFold(n_splits=5, random_state=42)
    print("Starting model tuning...")
    start_time = time.time()

    # make_scorer lets the loss function work with the model
    custom_scorer = make_scorer(custom_loss, greater_is_better=False)

    # make it faster (uses all cores available)
    n_cores = multiprocessing.cpu_count()
    print(f"Using {n_cores} CPU cores")

    # actual model generation
    gb = GradientBoostingRegressor(random_state=42)
    randomised_search = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                                           param_distributions=param_grid,
                                           n_iter=100,
                                           cv=cv,
                                           scoring=custom_scorer,
                                           n_jobs=n_cores,
                                           verbose=1,
                                           random_state=42)

    randomised_search.fit(X, y)

    end_time = time.time()
    print(f"Model tuning completed in {end_time - start_time:.2f} seconds")
    print("Best parameters:", randomised_search.best_params_)

    return randomised_search.best_estimator_


def generate_missing_rows(purpose_value, input_data, target_column):
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
                             purpose_value):
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

    # caf.ml data transformations (scale & encoding) function
    data_to_model, transformations = dpf.process_data_pipeline(df=nhb_to_model_final,
                                                               numerical_features=numerical_features,
                                                               categorical_features=categorical_features,
                                                               target_column=target_column,
                                                               output_folder=output_folder)

    ''' testing model generation with rows that are actually present in metadata'''
    data_for_training = data_to_model.dropna(subset=[target_column])
    y_log = np.log1p(data_for_training[target_column])
    y_non_log = data_for_training[target_column]
    x = data_for_training.drop(columns=[target_column])


    # log y (to ensure positive predictions when reverting log (exp.))
    # y_log = np.log1p(data_to_model[target_column]) # y_log = np.log1p(nhb[target_column])
    # y_non_log = data_to_model[target_column] # y_non_log = nhb[target_column]
    # x = data_to_model.drop(columns=[target_column])

    model_filename = os.path.join(output_folder, 'trained_model.pkl')

    train_index, test_index = train_test_split(range(len(x)), test_size=0.2, random_state=42)
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train_log, y_test_log = y_log.iloc[train_index], y_log.iloc[test_index]

    # save/load model
    if os.path.exists(model_filename):
        print(f"Loading pre-trained model: {model_filename}")
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    else:
        model = tune_model(X_train, y_train_log)
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
    ''' modified prediction based on generated rows (see other docstring)'''
    X_all = data_to_model.drop(columns=[target_column])
    y_pred_non_log = np.exp(model.predict(X_all))
    y_pred_log = model.predict(X_all)
    y_pred_non_log_excluding_generated_rows = np.exp(model.predict(x))
    y_pred_log_excluding_generated_rows = model.predict(x)


    if 'trips.hb' in data_to_model.columns:
        # gamma from predictions (non-log)
        gamma_pred = y_pred_non_log / original_data['trips.hb']
        # final_df = original_data.copy()
        final_df = nhb_to_model_final.copy()
        final_df['predicted_trips'] = y_pred_non_log
        final_df['predicted_gamma'] = gamma_pred
    else:
        # final_df = original_data.copy()
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
