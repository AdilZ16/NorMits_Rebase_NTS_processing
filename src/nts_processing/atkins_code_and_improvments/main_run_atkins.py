# -*- coding: utf-8 -*-
"""
Created on: 6/21/2024
Original author: Adil Zaheer
"""
import warnings

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position
import pandas as pd

from src.nts_processing.atkins_code_and_improvments.hb_code_atkins import process_data_hb, \
    model_unedited_hb, model_evaluation_atkins_method_tfnat_hb, \
    model_evaluation_atkins_method_hhtype_hb
from src.nts_processing.atkins_code_and_improvments.nhb_code_atkins import process_data_nhb, \
    model_unedited_nhb, model_evaluation_atkins_method_tfnat_nhb

warnings.filterwarnings("ignore")


def main(params):

    ######## HB ########
    df_trip_join, audit_df = process_data_hb(data=params.data_hb,
                                             output_folder=params.output_folder_hb)

    (resultant_df_final,
     audit_df,
     df_trip_join,
     all_modelled_at,
     all_modelled_hh) = model_unedited_hb(df_trip_join,
                                          audit_df,
                                          output_folder=params.output_folder_hb)

    model_scoring_doc_at = model_evaluation_atkins_method_tfnat_hb(df_trip_join,
                                                                   all_modelled_at,
                                                                   output_folder=params.output_folder_hb)

    model_scoring_doc_hh = model_evaluation_atkins_method_hhtype_hb(df_trip_join,
                                                                    all_modelled_hh,
                                                                    output_folder=params.output_folder_hb)

    ######## NHB ########
    df_trip_join_nhb_pre_modelling = process_data_nhb(data=params.data_nhb,
                                                      output_folder=params.output_folder_nhb)


    (resultant_df_final,
     audit_df,
     df_trip_join,
     all_modelled_at) = model_unedited_nhb(df_trip_join_nhb_pre_modelling,
                                           output_folder=params.output_folder_nhb)

    model_scoring_doc_at = model_evaluation_atkins_method_tfnat_nhb(df_trip_join,
                                                                    all_modelled_at,
                                                                    output_folder=params.output_folder_nhb)

    return


warnings.filterwarnings("default")
