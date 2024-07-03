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
