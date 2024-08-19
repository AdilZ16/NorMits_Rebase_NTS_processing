# -*- coding: utf-8 -*-
"""
Created on: 7/10/2024
Original author: Adil Zaheer
"""
import pandas as pd

# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position

from src.nts_processing.production_weights.production_weight_functions import TripRate, model_to_calculate_gamma


def main(params):
    # global statement
    df = None

    if params.data_skip_cb_generation is not None:

        df = pd.read_csv(params.data_skip_cb_generation)

        final_predictions = model_to_calculate_gamma(nhb=df,
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
        trip_rate_object = TripRate(data=params.data,
                                    mode=params.mode,
                                    geo_incl=params.geo_incl,
                                    segments_incl=params.segments_incl,
                                    columns_to_keep=params.columns_to_keep,
                                    output_folder=params.output_folder)

        if params.production_weight_calculation is not None:
            df, df_post_processing = trip_rate_object.nhb_production_weights_production()
            print(df)

        if params.mts_calculation is not None:
            df = trip_rate_object.process_cb_data_tfn_method()


        final_predictions = model_to_calculate_gamma(nhb=df,
                                                     output_folder=params.output_folder,
                                                     target_column=params.target_column,
                                                     numerical_features=params.numerical_features,
                                                     categorical_features=params.categorical_features,
                                                     index_columns=params.index_columns,
                                                     drop_columns=params.drop_columns,
                                                     ignore_columns=params.ignore_columns,
                                                     purpose_value=params.purpose_value)

    return final_predictions
