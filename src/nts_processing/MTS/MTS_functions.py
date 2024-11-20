# -*- coding: utf-8 -*-
"""
Created on: 10/7/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import multiprocessing
import os
import pickle
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


class TripRate_MTS:
    """
    Used to store and call the data processing functions.
    """
    def __init__(self,
                 data,
                 columns_to_keep,
                 output_folder,
                 purpose_value,
                 index_columns
                 ):
        self.data = data
        self.columns_to_keep = columns_to_keep
        self.output_folder = output_folder
        self.purpose_value = purpose_value
        self.index_columns = index_columns
        print('Trip rate class is running')


    def process_cb_data_tfn_method(self):
        return process_cb_data_tfn_method(
            data=self.data,
            columns_to_keep=self.columns_to_keep,
            output_folder=self.output_folder,
            purpose_value=self.purpose_value,
            index_columns=self.index_columns
        )


def process_cb_data_tfn_method(data, columns_to_keep, output_folder, purpose_value, index_columns):
    """
    :param index_columns:
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

    df_total = df.groupby(['tfn_at', 'hh_type', 'purpose', 'mode', 'period']).sum()
    df_total = df_total[['trips']].reset_index()

    # df_total['split_method'] = np.where(df_total.trips >= 1000, 'observed', 'TBD')
    df_total = df_total.rename(columns={'trips': 'total_trips'})

    df_total['mode_period'] = df_total['mode'].astype(str) + '_' + df_total['period'].astype(str)
    df_total.columns = [str(col).strip() for col in df_total.columns]

    # df_total.set_index(index_columns, inplace=True)

    return df_total


def prep_processed_data(data,
                        output_folder,
                        target_column,
                        mts,
                        categorical_features,
                        index_columns,
                        drop_columns,
                        ignore_columns,
                        purpose_value):
    data = data.apply(pd.to_numeric, errors='coerce')

    # make feature list to use
    if ignore_columns is None:
        features_to_use = data.columns.tolist()
    else:
        features_to_use = [col for col in data.columns if col not in ignore_columns]

    data_ = data[features_to_use]
    # caf.ml index columns function
    if index_columns is not None:
        if not all(col in data_.index.names for col in index_columns):
            data_ = index_sorter_modified(df=data_,
                                          index_columns=index_columns,
                                          drop_columns=drop_columns)


    # create all possible mts combinations to predict
    unencoded_data = generate_missing_rows(purpose_value=purpose_value,
                                           input_data=data_,
                                           target_column=target_column)

    final_data_pre_encoding = generate_raw_mts(mts=mts,
                                               processed_df=unencoded_data,
                                               purpose_value=purpose_value,
                                               output_folder=output_folder)

    columns_to_remove = ['mode_period', 'total_trips', 'rows_added', 'trips', 'rho']
    df_removed = final_data_pre_encoding[columns_to_remove]
    df_remaining = final_data_pre_encoding.drop(columns_to_remove, axis=1)

    # encode data
    if target_column in df_remaining.columns:
        y = df_remaining[target_column]
        x = df_remaining.drop(columns=[target_column])
    else:
        y = None
        x = df_remaining

    data_encoded = pd.get_dummies(x, columns=categorical_features, drop_first=True, dtype=float)

    # if y is not None:
    #     data_encoded[target_column] = y
    # if y is not None:
    #     y = y.reindex(data_encoded.index)
    #     data_encoded[target_column] = y

    df_combined = pd.concat([df_removed, data_encoded], axis=1)
    df_combined.to_csv(os.path.join(output_folder, 'final_data.csv'), index=True)
    return df_combined, final_data_pre_encoding


def index_sorter_modified(df, index_columns, drop_columns):
    # index_columns treated as list
    if isinstance(index_columns, str):
        index_columns = [index_columns]

    # Check if  index_columns are in the DataFrame
    for col in index_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    df_indexed = df.copy()

    # Set cols as index
    df_indexed.set_index(index_columns, inplace=True, verify_integrity=False)

    if drop_columns:
        if drop_columns not in df_indexed.columns.values:
            return df_indexed
        elif drop_columns:
            df_indexed = df_indexed.drop(columns=drop_columns)

    duplicates = df.index.duplicated()
    if duplicates.any():
        print("Duplicates found in the index:")
        print(df[duplicates])
    else:
        print("No duplicates in the index.")

    return df_indexed


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
    all_combinations = pd.MultiIndex.from_product([
        range(1, 21),  # tfn_at
        range(1, 9),  # hh_type
        [purpose_value],  # purpose
        range(1, 8),  # mode
        range(1, 7)  # period
    ], names=['tfn_at', 'hh_type', 'purpose', 'mode', 'period'])

    all_data = pd.DataFrame(index=all_combinations).reset_index()
    all_data['mode_period_new'] = all_data['mode'].astype(str) + '_' + all_data['period'].astype(str)

    merged_data = pd.merge(all_data, input_data,
                           on=['tfn_at', 'hh_type', 'purpose', 'mode', 'period'],
                           how='left')

    merged_data['rows_added'] = merged_data[target_column].isna()
    merged_data = merged_data.drop(columns='mode_period')
    merged_data = merged_data.rename(columns={'mode_period_new': 'mode_period'})
    return merged_data


def custom_loss_mts(y_true,
                    y_pred,
                    mode_period_values,
                    rows_added,
                    rho,
                    weight_ratio=5,
                    weight_sample=1,
                    weight_artificial=2):

    # mode_period_grid = {
    #     '1_1': 0.0773, '1_2': 0.0755, '1_3': 0.0258, '1_4': 0.0179, '1_5': 0.0302, '1_6': 0.0275,
    #     '2_1': 0.0063, '2_2': 0.0039, '2_3': 0.0016, '2_4': 0.0021, '2_5': 0.0019, '2_6': 0.0017,
    #     '3_1': 0.1739, '3_2': 0.1502, '3_3': 0.0644, '3_4': 0.0596, '3_5': 0.0914, '3_6': 0.0722,
    #     '4_1': 0.0063, '4_2': 0.0021, '4_3': 0.0012, '4_4': 0.0028, '4_5': 0.0020, '4_6': 0.0014,
    #     '5_1': 0.0292, '5_2': 0.0216, '5_3': 0.0030, '5_4': 0.0042, '5_5': 0.0087, '5_6': 0.0038,
    #     '6_1': 0.0096, '6_2': 0.0029, '6_3': 0.0007, '6_4': 0.0030, '6_5': 0.0021, '6_6': 0.0008,
    #     '7_1': 0.0057, '7_2': 0.0019, '7_3': 0.0006, '7_4': 0.0012, '7_5': 0.0012, '7_6': 0.0007
    # }

    total_ratio_penalty = 0
    total_sample_penalty = 0
    total_artificial_penalty = 0

    total_pred = np.sum(y_pred)
    total_true = np.sum(y_true)
    # double check how rho is calculated, and artificial is only one that really uses ypred and ytrue
    # rho = total trips by m and time period / total trips without mode and time period
    # percentage for that segmentation that will use that mode and time period
    # where sample is higher, it should be closer to observed value and vice and versa 
    for y_t, y_p, mode_period, row_added, target_ratio in zip(y_true, y_pred,
                                                              mode_period_values, rows_added,
                                                              rho):
        # Ratio penalty
        actual_ratio = y_p / total_pred
        ratio_penalty = (actual_ratio - target_ratio) ** 2
        total_ratio_penalty += ratio_penalty

        # Sample size penalty (using y_true as sample size)
        sample_penalty = 1 / (y_t + 1)  # Add 1 to avoid division by zero
        total_sample_penalty += sample_penalty

        # Artificial row penalty
        if row_added:
            artificial_penalty = abs(y_p / total_pred - y_t / total_true)
            total_artificial_penalty += artificial_penalty

    avg_ratio_penalty = total_ratio_penalty / len(y_true)
    avg_sample_penalty = total_sample_penalty / len(y_true)
    avg_artificial_penalty = total_artificial_penalty / len(y_true)

    loss = (weight_ratio * avg_ratio_penalty +
            weight_sample * avg_sample_penalty +
            weight_artificial * avg_artificial_penalty)

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


def mts_predict(df,
                unencoded_df,
                output_folder,
                target_column):

    start_time = time.time()

    original_data = df.copy()

    # prep data for modelling
    non_predictive_columns = ['rho', 'mode_period', 'trips', 'rows_added']
    df_removed = df[non_predictive_columns]
    df[target_column] = df[target_column].fillna(0)

    y_log = np.log1p(df[target_column])
    y_non_log = df[target_column]
    x = df.drop(columns=[target_column] + non_predictive_columns)


    train_index, test_index = train_test_split(range(len(x)), test_size=0.2, random_state=42)
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train_log, y_test_log = y_log.iloc[train_index], y_log.iloc[test_index]


    # creating custom loss function
    custom_scorer_loss_func = None
    mode_period_values = unencoded_df['mode_period'].tolist()
    rho_values = unencoded_df['rho'].tolist()
    rows_added = unencoded_df['rows_added'].tolist()
    custom_scorer_loss_func = make_scorer(lambda y_true, y_pred: custom_loss_mts(y_true, y_pred, mode_period_values, rows_added, rho_values),
                                          greater_is_better=False)


    # save/load model
    model_filename = os.path.join(output_folder, 'trained_model.pkl')
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
    X_all = df.drop(columns=[target_column] + non_predictive_columns)
    y_pred_non_log = np.exp(model.predict(X_all))
    y_pred_non_log_excluding_generated_rows = np.exp(model.predict(x))
    y_pred_log_excluding_generated_rows = model.predict(x)


    # calculate gamma from predicted values
    gamma_pred = y_pred_non_log / original_data['trips']
    final_df = unencoded_df.copy()
    final_df['predicted_trips'] = y_pred_non_log
    final_df['predicted_gamma'] = gamma_pred


    # Feature importance (for extra evaluation)
    if isinstance(model, GradientBoostingRegressor) and hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({'feature': x.columns, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(os.path.join(output_folder, 'feature_importance.csv'), index=True)

    final_df.to_csv(os.path.join(output_folder, 'final_predictions.csv'), index=True)


    # evaluate model
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

    plots(y_non_log, y_pred_non_log_excluding_generated_rows, output_folder)

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


def generate_raw_mts(mts, purpose_value, processed_df, output_folder):
    df = pd.read_csv(mts)
    df = df[df['period'] != 0]
    df = df[~df['mode'].isin([8, 0])]
    df = df[df['purpose'] == purpose_value]
    df = df.reset_index(drop=True)
    df['mode_period'] = df['mode'].astype(str) + '_' + df['period'].astype(str)
    df.columns = [str(col).strip() for col in df.columns]

    df_final = pd.merge(processed_df, df, how='left', on=['tfn_at', 'hh_type',
                                                          'purpose', 'mode',
                                                          'period', 'mode_period'])


    df_final.to_csv(os.path.join(output_folder, 'final_data_pre_encoding.csv'), index=False)

    return df_final


def plots(y_non_log, y_pred_non_log_excluding_generated_rows, output_folder):
    residuals = y_non_log - y_pred_non_log_excluding_generated_rows
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_folder, 'residuals_distribution.png'))
    plt.close()

    sns.boxplot(x=residuals)
    plt.title('Boxplot of Residuals')
    plt.savefig(os.path.join(output_folder, 'residuals_boxplot.png'))
    plt.close()

    sns.regplot(x=y_non_log, y=y_pred_non_log_excluding_generated_rows, line_kws={"color": "red"})
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.savefig(os.path.join(output_folder, 'actual_vs_predicted.png'))
    plt.close()
