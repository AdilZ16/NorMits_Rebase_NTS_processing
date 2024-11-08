# -*- coding: utf-8 -*-
"""
Created on: 10/7/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position


import pandas as pd
from src.nts_processing.MTS.MTS_functions import TripRate_MTS, mts_predict, prep_processed_data


def main(params):
    df = None
    if params.data_skip_cb_generation is not None:

        df = pd.read_csv(params.data_skip_cb_generation)

        final_predictions = mts_predict(nhb=df,
                                        output_folder=params.output_folder,
                                        target_column=params.target_column,
                                        numerical_features=params.numerical_features,
                                        categorical_features=params.categorical_features,
                                        index_columns=params.index_columns,
                                        drop_columns=params.drop_columns,
                                        ignore_columns=params.ignore_columns,
                                        purpose_value=params.purpose_value)
        return final_predictions

    else:
        trip_rate_object = TripRate_MTS(data=params.data,
                                        columns_to_keep=params.columns_to_keep,
                                        output_folder=params.output_folder,
                                        purpose_value=params.purpose_value)

        processed_df = trip_rate_object.process_cb_data_tfn_method()

        df_to_model, unencoded_data = prep_processed_data(data=processed_df,
                                                          output_folder=params.output_folder,
                                                          target_column=params.target_column,
                                                          numerical_features=params.numerical_features,
                                                          categorical_features=params.categorical_features,
                                                          index_columns=params.index_columns,
                                                          drop_columns=params.drop_columns,
                                                          ignore_columns=params.ignore_columns,
                                                          purpose_value=params.purpose_value)

        final_predictions = mts_predict(df=df_to_model,
                                        unencoded_df=unencoded_data,
                                        output_folder=params.output_folder,
                                        target_column=params.target_column)

    return final_predictions
