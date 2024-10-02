# -*- coding: utf-8 -*-
"""
Created on: 9/11/2024
Original author: Adil Zaheer
"""
import os

import joblib
import numpy as np
import pandas as pd
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
from caf.ml.functions import data_pipeline_functions as dpf
from caf.ml.functions import process_data_functions as pdf
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score, \
    train_test_split
from tqdm import tqdm
from src.nts_processing.inputs_nts_processing.nts_processing_inputs import ModelGrids, Models
from sklearn.linear_model import LogisticRegression


def transform_data_pre_modelling(data,
                                 output_folder,
                                 index_columns,
                                 drop_columns,
                                 numerical_features,
                                 categorical_features,
                                 target_column):
    print('Indexing, scaling and encoding functions beginning')
    if index_columns is not None:
        dat = pdf.index_sorter(data,
                               index_columns=index_columns,
                               drop_columns=drop_columns)
    else:
        dat = data


    dat, transformations = dpf.process_data_pipeline(df=dat,
                                                     numerical_features=numerical_features,
                                                     categorical_features=categorical_features,
                                                     target_column=target_column,
                                                     output_folder=output_folder)

    return dat


class train_trip_rate_model:
    def __init__(self,
                 model_type,
                 data,
                 target_column,
                 n_cores,
                 output_folder
                 ):
        self.model_type = model_type
        self.data = data
        self.target_column = target_column
        self.n_cores = n_cores
        self.output_folder = output_folder

    def initialise_model(self):
        model = self.model_type[0].get_model_instance_() if isinstance(self.model_type[0], Models) else self.model_type[0]
        print(model)
        return model

    def hyper_optim(self, model):
        print('Hyperparameter optimisation now in progress')
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        model_name = model.__class__.__name__
        param_grid = {'n_estimators': [300, 400, 500],
                      'max_depth': [5, 7, 9],
                      'min_samples_split': [2, 5],
                      'min_samples_leaf': [2, 4],
                      'learning_rate': [0.05, 0.1, 0.2],
                      'subsample': [0.8, 0.9, 1.0]}
        print(model_name)
        print(param_grid)

        cv = TimeSeriesSplit(n_splits=5)

        if model_name == 'MULTINOMIAL':
            model_instance = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            param_grid = ModelGrids[model_name].value
            search = RandomizedSearchCV(model_instance, param_grid, cv=cv, scoring='accuracy',
                                        verbose=2,
                                        n_jobs=self.n_cores, random_state=42)
            search.fit(X, y)
        else:
            scoring = make_scorer(mean_squared_error, greater_is_better=False)
            search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=2,
                                        n_jobs=self.n_cores, random_state=42)
            search.fit(X, y)

        print('Best parameters:', search.best_params_)
        print('CV results:', search.cv_results_)

        hyperparameters = search.best_params_
        model.set_params(**hyperparameters)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        return hyperparameters, model

    def save_model(self, model, hyperparameters):

        regression_method_file_path = self.output_folder / 'regression_method.pkl'
        hyperparameters_file_path = self.output_folder / 'hyperparameters.pkl'

        joblib.dump(model, regression_method_file_path)
        joblib.dump(hyperparameters, hyperparameters_file_path)

        print(f"Model saved to: {regression_method_file_path}")
        print(f"Parameters saved to: {hyperparameters_file_path}")

    def run(self):
        model_ = self.initialise_model()
        hyperparameters, model = self.hyper_optim(model_)
        self.save_model(model, hyperparameters)
        return model, hyperparameters


def prep_data_gravity(output_folder, ppm_ppk_data):
    tidy_cb_path = output_folder / 'processed_cb.csv'
    cb = pd.read_csv(tidy_cb_path)
    print(f"Processed classified build dataFrame loaded from: {tidy_cb_path}")
    ppm_ppk_data = pd.read_csv(ppm_ppk_data)
    dat = pd.merge(cb, ppm_ppk_data, on=['mode', 'purpose'], how='outer')
    dat = dat.dropna()
    dat = dat.set_index(['tfn_at', 'tt', 'mode', 'purpose', 'period', 'tfn_at_o', 'tfn_at_d', 'dist_bands'])
    dat = dat[['triptravtime', 'trav_dist', 'ppm', 'ppk']]
    return dat


# for each band, mode, period, tt, at should be trips value. where there isnt, make model predict
def cost_func(data):
    # gen_cost = (data['ppm'] * data['triptravtime']) + (data['ppk'] * data['tripdisincsw'])
    gen_cost_in_minutes = (data['triptravtime']) + ((data['ppk'] / data['ppm']) * data['trav_dist'])
    # + monetary_costs
    data['gen_cost'] = gen_cost_in_minutes
    return data
# at = 1-20, mode 1-7, period 1-7, 16 bands
# mode , purpose, period, tt,
# predict trip rates based on diff distance travelled. grav model to have cost related func
#

def grav_model(gen_cost_data, x_pred, target_column, model, hyperparameters):
    # make predictions
    x_pred = x_pred.drop(columns=[target_column])
    model.fit(x_pred, x_pred[target_column])
    model.set_params(**hyperparameters)
    trip_rates = model.predict(x_pred)

    gen_cost_data['predicted_trips'] = trip_rates
    pop_o = 1
    emp_d = 1


    # creating distance bands
    bins = [0, 1, 2, 5, 9, 14, 20, 30, 45, 70, 100, 140, 200, 300, 450, 700, 999]
    labels = [f'{bins[i]} up to {bins[i + 1]}' for i in range(len(bins) - 1)]
    gen_cost_data['dist_bands'] = pd.cut(gen_cost_data['trav_dist'], bins=bins, labels=labels,
                              include_lowest=True, right=False)
    gen_cost_data.loc[gen_cost_data['trav_dist'] == bins[-1], 'dist_bands'] = f'{bins[-2]} up to {bins[-1]}'

    # grav model
    gen_cost_data['grav_model_result'] = gen_cost_data['predicted_trips'] * np.exp(-0.1 * gen_cost_data['gen_cost']) * pop_o * emp_d

    return gen_cost_data






























def probability_of_trip(model, best_params, data):
    model.set_params(**best_params)
    utility = model.predict(data)[0]
    total_trips = data['trips'].sum()
    probability = 1 / (1 + np.exp(-utility))
    trips_calculated = probability * total_trips
    return trips_calculated


def create_base_utility_function(model, feature_names, best_params):
    def utility(**kwargs):
        input_data = pd.DataFrame([kwargs])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        model.set_params(**best_params)
        return model.predict(input_data)[0]



    return utility


def generate_utility_functions(model, data, best_params):
    feature_names = data.columns.tolist()
    utility_function = create_base_utility_function(model=model,
                                                    feature_names=feature_names,
                                                    best_params=best_params)

    return utility_function, feature_names


def calculate_trips(utility_function, inputs):
    utility = utility_function(**inputs)
    # Convert utility to probability (you might want to adjust this based on your specific needs)
    probability = 1 / (1 + np.exp(-utility))
    # Assume total trips is 1000 (adjust as needed)
    total_trips = 1000
    trips = probability * total_trips
    return trips


def load_model(output_folder):
    regression_method_file_path = output_folder / 'regression_method.pkl'
    hyperparameters_file_path = output_folder / 'hyperparameters.pkl'
    dataframe_file_path = output_folder / 'final_data_ready_to_model.csv'

    regression_method = joblib.load(regression_method_file_path)
    parameters = joblib.load(hyperparameters_file_path)
    df_final = pd.read_csv(dataframe_file_path)

    print(f"Model loaded from: {regression_method_file_path}")
    print(f"Parameters loaded from: {hyperparameters_file_path}")
    print(f"DataFrame loaded from: {dataframe_file_path}")

    return regression_method, parameters, df_final





















#############################################################################
# optional function not yet included
def feature_selection(data, target_column, model_type, output_folder, n_cores):

    x = data.drop(columns=[target_column])
    y = data[target_column]

    print('Selecting best model')
    print("Models to test:", [type(model) for model in model_type])
    model = simple_model_selection(x=x, y=y, models_to_test=model_type, n_jobs=n_cores) if len(
        model_type) > 1 else model_type[0].value()

    best_features = {'rfe': [], 'mi': []}
    best_scores = {'rfe': float('-inf'), 'mi': float('-inf')}

    cv = TimeSeriesSplit(n_splits=5)

    for train_index, test_index in tqdm(cv.split(x), desc='Feature selection in progress'):
        X_train, y_train = x.iloc[train_index], y.iloc[train_index]

        # RFE
        selector_rfe = RFE(estimator=model, n_features_to_select=5, step=1)
        selector_rfe.fit(X_train, y_train)
        score_rfe = selector_rfe.score(X_train, y_train)
        if score_rfe > best_scores['rfe']:
            best_scores['rfe'] = score_rfe
            best_features['rfe'] = X_train.columns[selector_rfe.support_].tolist()

        # Mutual Information
        selector_mi = SelectKBest(mutual_info_classif, k='all')
        selector_mi.fit(X_train, y_train)
        score_mi = selector_mi.score(X_train, y_train)
        if score_mi > best_scores['mi']:
            best_scores['mi'] = score_mi
            best_features['mi'] = X_train.columns[selector_mi.get_support()].tolist()

    print(f"Best RFE score: {best_scores['rfe']}")
    print(f"Best MI score: {best_scores['mi']}")

    selected_features = list(set(best_features['rfe'] + best_features['mi']))
    final_data = data[selected_features + [target_column]]
    final_data.to_csv(output_folder, 'data_post_feature_selection.csv', index=True)
    print(f"Feature selected data exported to: {output_folder, 'data_post_feature_selection.csv'}")

    return final_data, model


def sample_data(x, y, sample_size=10000, random_state=42):

    if len(x) > sample_size:
        np.random.seed(random_state)
        sampled_indices = np.random.choice(len(x), size=sample_size, replace=False)
        return x.iloc[sampled_indices], y.iloc[sampled_indices]
    return x, y


def simple_model_selection(x, y, models_to_test, n_jobs):
    cv = 2
    best_model = None
    best_score = float('-inf')


    print("Testing models:")
    for model_enum in models_to_test:
        model_instance = model_enum.value()
        model_name = model_enum.name

        scores = cross_val_score(model_instance, x, y, cv=cv, scoring='r2', n_jobs=n_jobs)
        mean_score = scores.mean()

        print(f"{model_name}: Average R2 score = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model_instance

    print(f"\nBest model: {type(best_model).__name__}")
    print(f"Best R2 score: {best_score:.4f}")

    return best_model


def filter_data(original_data,
                output_folder,
                target_column,
                best_features_f_regressor=None,
                best_features_rfe=None):


    if best_features_f_regressor is None:
        best_features_f_regressor = []
    if best_features_rfe is None:
        best_features_rfe = []


    all_selected_features = best_features_f_regressor + best_features_rfe
    feature_counts = pd.Series(all_selected_features).value_counts()
    selected_columns = feature_counts[feature_counts >= 2].index.tolist()
    selected_columns.append(target_column)
    filtered_data = original_data[selected_columns]

    output_filename = 'data_post_feature_selection.csv'
    output_path = os.path.join(output_folder, output_filename)
    filtered_data.to_csv(output_path, index=True)
    print('-------------------------------------------------------------')
    print(f"Feature selected data exported to: {output_path}")

    return filtered_data
