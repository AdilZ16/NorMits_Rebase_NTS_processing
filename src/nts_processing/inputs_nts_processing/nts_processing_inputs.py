# -*- coding: utf-8 -*-
"""
Created on: 6/21/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

from pathlib import Path
from typing import Optional, List
from caf.toolkit import BaseConfig


class NTS_mts_inputs(BaseConfig):
    data_hb: Optional[Path] = None
    data_nhb: Optional[Path] = None
    output_folder_hb: Optional[Path] = None
    output_folder_nhb: Optional[Path] = None


class NTS_mts_inputs_AZ_model(BaseConfig):
    data_cb: Optional[Path] = None
    columns_to_keep: Optional[List[str]] = None
    output_folder: Optional[Path] = None
    reduce_data_size: Optional[str] = None
    target_column: Optional[str] = None
    atkins_data_derivation_method: Optional[str] = None


class NTS_production_weights_inputs(BaseConfig):
    data: Optional[Path] = None
    data_skip_cb_generation: Optional[Path] = None
    mode: Optional[List[str]] = None
    geo_incl: Optional[str] = None
    segments_incl: Optional[List[str]] = None
    columns_to_keep: Optional[List[str]] = None
    output_folder: Optional[Path] = None
    target_column: Optional[str] = None
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    index_columns: Optional[List[str]] = None
    drop_columns: Optional[List[str]] = None
    ignore_columns: Optional[List[str]] = None
    production_weight_calculation: Optional[str] = None
    mts_calculation: Optional[str] = None
