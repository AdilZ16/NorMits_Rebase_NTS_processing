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

    if params.data_skip_cb_generation is not None:

        nhb = pd.read_csv(params.data_skip_cb_generation)

        final_predictions = model_to_calculate_gamma(nhb=nhb,
                                                     output_folder=params.output_folder,
                                                     target_column=params.target_column,
                                                     numerical_features=params.numerical_features,
                                                     categorical_features=params.categorical_features,
                                                     index_columns=params.index_columns,
                                                     drop_columns=params.drop_columns,
                                                     ignore_columns=params.ignore_columns)


    else:
        trip_rate_object = TripRate(data=params.data,
                                    mode=params.mode,
                                    geo_incl=params.geo_incl,
                                    segments_incl=params.segments_incl,
                                    columns_to_keep=params.columns_to_keep,
                                    output_folder=params.output_folder)

        nhb, df_post_processing = trip_rate_object.nhb_production_weights_production()
        print(nhb)

        final_predictions = model_to_calculate_gamma(nhb=nhb,
                                                     output_folder=params.output_folder,
                                                     target_column=params.target_column,
                                                     numerical_features=params.numerical_features,
                                                     categorical_features=params.categorical_features,
                                                     index_columns=params.index_columns,
                                                     drop_columns=params.drop_columns,
                                                     ignore_columns=params.ignore_columns)

    return
