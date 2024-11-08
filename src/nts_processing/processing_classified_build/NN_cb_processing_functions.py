# -*- coding: utf-8 -*-
"""
Created on: 10/7/2024
Original author: Adil Zaheer (these are Nhans functions)
"""
import os
import sys
from typing import Any, Dict, Union, List

import pandas as pd


# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position

def prepare_data(dfr, mode, geo_incl, seg_incl, tfn_ttype, tfn_mode):
    print('preparing data')
    lev_prod, lev_dest = ["tfn_at"], ["tfn_at_d"]
    seg_incl = val_to_list(seg_incl) if seg_incl is not None else []
    col_used = ["mode", "purpose"]
    seg_incl = col_used + (lev_dest if geo_incl is not None else []) + seg_incl
    dfr = dfr[seg_incl + ["direction", "period", "tour_group", "individualid", "trips"]].copy()
    mode = tfn_mode if mode is None else mode
    return dfr, mode, lev_prod, lev_dest, seg_incl, col_used


def separate_trips(dfr):
    print('separating trips')
    hbf = dfr.loc[dfr["direction"].isin(["hb_fr"])].reset_index(drop=True)
    nhb = dfr.loc[dfr["direction"].isin(["nhb"])].reset_index(drop=True)
    return hbf, nhb


def merge_and_aggregate_trips(hbf, nhb, seg_incl):
    print('merge and aggregating trips')
    col_join = ["individualid", "tour_group"]
    nhb = pd.merge(nhb, hbf, how="left", on=col_join, suffixes=("", ".hb")).fillna(0)
    col_u4hb = [f"{col}.hb" if col in ["mode", "purpose"] else col for col in seg_incl]
    hbf = hbf.groupby(seg_incl)[["trips"]].sum().reset_index()
    hbf.rename(columns=dict(zip(seg_incl, col_u4hb)), inplace=True)
    col_join = seg_incl + [col for col in col_u4hb if col not in seg_incl]
    nhb = nhb.groupby(col_join)[["trips"]].sum().reset_index()
    return nhb, hbf, col_join, col_u4hb


def filter_and_complete(nhb, mode, col_join, col_u4hb, hbf, output_folder):
    print('filtering final dataframe')
    nhb = dfr_filter_zero(nhb, col_join)
    nhb = dfr_filter_mode(nhb, mode, "mode.hb")
    nhb = dfr_filter_mode(nhb, mode)
    nhb = dfr_complete(nhb, col_join, "mode.hb").reset_index()
    nhb = pd.merge(nhb, hbf, how="left", on=col_u4hb, suffixes=("", ".hb"))
    df_post_processing = nhb.copy()

    print('outputting df_post_processing')
    df_post_processing.to_csv(os.path.join(output_folder, '1_df_post_processing.csv'), index=False)

    return nhb, df_post_processing


def calculate_initial_gamma(nhb, min_trip=500):
    print('calculating initial gamma')
    nhb['tfn_at'] = nhb['tfn_at_d']
    msk = (nhb['trips.hb'] > min_trip)
    nhb.loc[msk, "gamma"] = nhb["trips"].div(nhb["trips.hb"]).loc[msk]
    return nhb


def define_atx_type():
    atx_type = {'lv1': {1: [1, 2], 2: [3, 4, 5], 3: [6, 7, 8], 4: [9, 10, 11], 5: [12, 13, 14, 15],
                        6: [16, 17], 7: [18, 19], 8: [20]},
                'lv2': {1: [1, 2, 3], 2: [4, 5, 6, 7, 8, 20], 3: [9, 10, 11, 12, 13, 14, 15],
                        4: [16, 17, 18, 19]},
                'lv3': {1: [1, 2, 3, 4, 5, 6, 7, 8, 20], 2: [9, 10, 11, 12, 13, 14, 15],
                        3: [16, 17, 18, 19, 20]}}
    return atx_type


def area_type_aggregation(nhb, atx_type, col_join, min_trip=500):
    print('calculating gamma using aggregation where required')
    for key, itm in atx_type.items():
        nhb['tfn_at_d'] = nhb['tfn_at']
        nhb = nhb.set_index('tfn_at_d').rename(index=itm_to_key(itm)).reset_index()
        nhb['trips_agg'] = nhb.groupby(col_join)['trips'].transform('sum')
        nhb['trips.hb_agg'] = nhb.groupby(col_join)['trips.hb'].transform('sum')
        msk = (nhb['trips.hb_agg'] > min_trip) & nhb['gamma'].isna()
        nhb.loc[msk, "gamma"] = nhb["trips_agg"].div(nhb["trips.hb_agg"]).loc[msk]
    msk = nhb['gamma'].isna()
    nhb.loc[msk, 'gamma'] = nhb["trips_agg"].div(nhb["trips.hb_agg"]).loc[msk]
    nhb = nhb.drop(columns=['tfn_at_d', 'trips_agg', 'trips.hb_agg'])
    return nhb


def final_processing_and_output(nhb, lev_dest, lev_prod, geo_incl, output_folder):
    nhb = (nhb.rename(columns={key: itm for key, itm in zip(lev_dest, lev_prod)})
           if geo_incl is not None else nhb)

    if output_folder is not None:
        print('outputting nhb data')

        nhb.to_csv(os.path.join(output_folder, '2_nhb_data_with_gamma.csv'), index=False)
    return nhb


############################## functions needed to run code above (also from Nhan) ################################

def val_to_list(str_text: Union[str, float, int, List]) -> List:
    return [str_text] if not isinstance(str_text, list) else str_text


def dfr_filter_zero(dfr: pd.DataFrame, col_used: Union[List, str]) -> pd.DataFrame:
    col_used = [col_used] if isinstance(col_used, str) else col_used
    return dfr.loc[(~dfr[col_used].isin([0, "0", -8, -9, -10])).all(axis=1)].reset_index(drop=True)


# filter mode values
def dfr_filter_mode(
        dfr: pd.DataFrame, inc_list: List, col_mode: str = "mode"
) -> pd.DataFrame:
    return dfr.loc[dfr[col_mode].isin(inc_list)].reset_index(drop=True)


# create a complete set of index values
def dfr_complete(
        dfr: pd.DataFrame, col_index: Union[List, str, None], col_unstk: Union[List, str]
) -> pd.DataFrame:
    col_index = [] if col_index is None else val_to_list(col_index)
    col_unstk = val_to_list(col_unstk)
    dfr = dfr.set_index(col_index) if len(col_index) > 0 else dfr
    for col in col_unstk:
        dfr = dfr.unstack(level=col, fill_value=0).stack(future_stack=True)  # future_stack implemented in pandas 2.1
    return dfr


def itm_to_key(dct: Dict, key_lower: bool = True) -> Dict:
    # swap key to value
    dct = {key: [dct[key]] if not isinstance(dct[key], list) else dct[key] for key in dct}
    return {str_lower(val) if key_lower else val: key for key in dct for val in dct[key]}


def log_stderr(*args):
    print(*args, file=sys.stderr, flush=True)


def str_lower(val: Any) -> Any:
    if isinstance(val, str):
        return val.lower()
    elif isinstance(val, (tuple, list, dict, set)):
        return tuple(itm.lower() if isinstance(itm, str) else itm for itm in val)
    else:
        return val
