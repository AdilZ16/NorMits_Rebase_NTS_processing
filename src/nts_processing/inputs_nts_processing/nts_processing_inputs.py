# -*- coding: utf-8 -*-
"""
Created on: 6/21/2024
Original author: Adil Zaheer
"""
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

from pathlib import Path
from typing import Optional, List, Any

import numpy as np
from caf.toolkit import BaseConfig
import enum
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor


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
    purpose_value: Optional[int] = None


class destination_choice_inputs(BaseConfig):
    data: Optional[Path] = None
    output_folder: Optional[Path] = None
    index_columns: Optional[List[str]] = None
    drop_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    columns_to_model: Optional[List[str]] = None
    model_type: List[Any]
    saved_model: Optional[str] = None
    purpose_value: Optional[int] = None
    ppm_ppk_data: Optional[Path] = None


class Models(enum.Enum):
    MULTINOMIAL = (LogisticRegression, {'multi_class': 'multinomial', 'solver': 'lbfgs'})
    RANDOM_FOREST = RandomForestRegressor
    EXTRA_TREES = ExtraTreesRegressor
    GRADIENT_BOOSTING = GradientBoostingRegressor
    DECISION_TREE = DecisionTreeRegressor

    def get_model_instance_(self):
        if isinstance(self.value, tuple):
            model_class, params = self.value
            return model_class(**params)
        else:
            return self.value()



'''    def get_model_instance_(self):
        model_class, params = self.value
        return model_class(**params) if model_class == LogisticRegression else model_class
'''

Models_List_ = [Models.RANDOM_FOREST,
                Models.EXTRA_TREES,
                Models.GRADIENT_BOOSTING,
                Models.DECISION_TREE]


class ModelGrids(enum.Enum):
    RANDOM_FOREST = {
        "n_estimators": [10, 50, 100, 200, 300],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False]
    }

    EXTRA_TREES = {
        "n_estimators": [50, 100, 200, 300, 400],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False]
    }

    GRADIENT_BOOSTING = {'n_estimators': [300, 400, 500],
     'max_depth': [5, 7, 9],
     'min_samples_split': [2, 5],
     'min_samples_leaf': [2, 4],
     'learning_rate': [0.05, 0.1, 0.2],
     'subsample': [0.8, 0.9, 1.0],
     }

    DECISION_TREE = {
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_features": ["auto", "sqrt", "log2"],
        "criterion": ["gini", "entropy"]
    }

    MULTINOMIAL = {"C": np.arange(0.1, 10, 0.1).tolist(), "solver": ["lbfgs", "newton-cg", "sag", "saga"]}
