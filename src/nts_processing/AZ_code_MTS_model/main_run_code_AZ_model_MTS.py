# -*- coding: utf-8 -*-
"""
Created on: 6/26/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import warnings
from src.nts_processing.AZ_code_MTS_model.process_cb_functions import (process_cb_data,
                                                                       process_data_for_hierarchical_bayesian_model,
                                                                       generate_priors,
                                                                       predict_model_hierarchical_bayesian_model,
                                                                       load_model_and_parameters,
                                                                       save_model_and_parameters)

warnings.filterwarnings("ignore")



def main(params):
    df, audit_df = process_cb_data(data=params.data_cb,
                                   columns_to_keep=params.columns_to_keep,
                                   output_folder=params.output_folder
                                   )

    df, predict_data, training_data = process_data_for_hierarchical_bayesian_model(df=df,
                                                                                   columns_to_keep=params.columns_to_keep,
                                                                                   reduce_data_size=params.reduce_data_size,
                                                                                   target_column=params.target_column,
                                                                                   output_folder=params.output_folder
                                                                                   )

    model, trace = load_model_and_parameters(output_folder=params.output_folder)

    if model is None or trace is None:
        model, trace = generate_priors(data=training_data)
        save_model_and_parameters(model=model,
                                  trace=trace,
                                  output_folder=params.output_folder)

    else:
        print("Using pre-existing model and trace.")
        model, trace = load_model_and_parameters(output_folder=params.output_folder)

    predictions, credible_intervals = predict_model_hierarchical_bayesian_model(data_to_predict=predict_data,
                                                                                trip_model=model,
                                                                                trace=trace,
                                                                                output_folder=params.output_folder,
                                                                                target_column=params.target_column)
    return


warnings.filterwarnings("default")
