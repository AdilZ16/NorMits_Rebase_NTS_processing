# -*- coding: utf-8 -*-
"""
Created on: 6/26/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import warnings

import numpy as np

from src.nts_processing.AZ_code_MTS_model.process_cb_functions import (process_cb_data_atkins_method,
                                                                       process_data_for_hierarchical_bayesian_model,
                                                                       generate_priors,
                                                                       predict_model_hierarchical_bayesian_model,
                                                                       load_model_and_parameters,
                                                                       save_model_and_parameters,
                                                                       evaluate_model,
                                                                       process_cb_data_tfn_method)

warnings.filterwarnings("ignore")



def main(params):
    if params.atkins_data_derivation_method is not None:
        df, audit_df = process_cb_data_atkins_method(data=params.data_cb,
                                       columns_to_keep=params.columns_to_keep,
                                       output_folder=params.output_folder)

    else:

        df = process_cb_data_tfn_method(data=params.data_cb,
                                        columns_to_keep=params.columns_to_keep,
                                        output_folder=params.output_folder)


    df = process_data_for_hierarchical_bayesian_model(df=df,
                                                      columns_to_keep=params.columns_to_keep,
                                                      reduce_data_size=params.reduce_data_size,
                                                      target_column=params.target_column,
                                                      output_folder=params.output_folder)

    model, trace = load_model_and_parameters(output_folder=params.output_folder)

    if model is None or trace is None:
        model, trace = generate_priors(data=df,
                                       target_column=params.target_column)
        save_model_and_parameters(model=model,
                                  trace=trace,
                                  output_folder=params.output_folder)

    else:
        print("Using pre-existing model and trace.")
        model, trace = load_model_and_parameters(output_folder=params.output_folder)

    predictions = predict_model_hierarchical_bayesian_model(data_to_predict=df,
                                                            trip_model=model,
                                                            trace=trace,
                                                            output_folder=params.output_folder,
                                                            target_column=params.target_column)

    metrics_df = evaluate_model(y_true=predictions[params.target_column],
                                y_pred_log=predictions[f'predicted_log_{params.target_column}'],
                                output_folder=params.output_folder)
    return


warnings.filterwarnings("default")
