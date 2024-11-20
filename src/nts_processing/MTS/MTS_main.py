# -*- coding: utf-8 -*-
"""
Created on: 10/7/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import pandas as pd
from src.nts_processing.MTS.MTS_functions import TripRate_MTS, mts_predict, prep_processed_data
# import sys
# sys.path.append(r"C:\Users\Liberty\Documents\GitHub\NTS-Processing\python")
# from mdloutput import _hb_mts


def main(params):
    trip_rate_object = TripRate_MTS(data=params.data,
                                    columns_to_keep=params.columns_to_keep,
                                    output_folder=params.output_folder,
                                    purpose_value=params.purpose_value,
                                    index_columns=params.index_columns)

    processed_df = trip_rate_object.process_cb_data_tfn_method()


    df_to_model, final_data_pre_encoding = prep_processed_data(data=processed_df,
                                                               output_folder=params.output_folder,
                                                               target_column=params.target_column,
                                                               mts=params.mts,
                                                               categorical_features=params.categorical_features,
                                                               index_columns=params.index_columns,
                                                               drop_columns=params.drop_columns,
                                                               ignore_columns=params.ignore_columns,
                                                               purpose_value=params.purpose_value)


    final_predictions = mts_predict(df=df_to_model,
                                    unencoded_df=final_data_pre_encoding,
                                    output_folder=params.output_folder,
                                    target_column=params.target_column)

    return final_predictions
