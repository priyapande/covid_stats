from typing import Dict, Any, List
import pandas as pd
import numpy as np
import math


def get_state_data(filename: str, states: List[str], cols: List[str], location_col_name='state',
                   remove_nan: bool = True):
    """
    Select data corresponding to each state
    Additionally, remove missing (nan) values

    :param filename: data path for full data
    :param states: states for which we want data
    :param cols: columns of interest
    :param location_col_name: column name of location
    :param remove_nan: flag to check if nan values need to be removed

    :return: List[dataframe]: data corresponding to each state
    """
    data_df = pd.read_csv(filename)
    states_data = []
    for state in states:
        state_data = data_df.loc[data_df[location_col_name] == state]
        state_data_cols = state_data[cols]
        total_rows = state_data_cols.shape[0]
        if remove_nan:
            state_data_cols = state_data_cols.dropna()
            total_rows_after_nan_removal = state_data_cols.shape[0]
            print("State: {} rows with missing values: {}".format(state, total_rows - total_rows_after_nan_removal))
        states_data.append(state_data_cols)
    return states_data


def get_daily_cases_data(data, location_col_name='state', date_col_name='submission_date',
                         non_cumulative_cols=[], set_zero_for_negatives=True):
    """
    Daily data extracted from cumulative data

    :param data: dataframe of raw data for a given state
    :param location_col_name: location column name
    :param date_col_name: date column name
    :param non_cumulative_cols: columns not to be factored in for daily computation
    :param set_zero_for_negatives: flag to check if negative values need to be made 0
    :return: Dataframe with daily data
    """
    daily_data_df = pd.DataFrame()

    # sort dataset by date (this is needed for daily data computation from cumulative)
    data[date_col_name] = pd.to_datetime(data[date_col_name])
    data = data.sort_values(by=date_col_name).reset_index(drop=True)

    for col in data:
        if col == date_col_name or col == location_col_name or col in non_cumulative_cols:
            daily_data_df[col] = data[col]
            continue
        # computing the daily values by subtracting prev row values
        daily_data_df[col] = data[col].diff().fillna(data.iloc[0][col])
        if set_zero_for_negatives:
            # some values were negative (e.g. when number of cases is corrected)
            # replacing those with zero
            negative_values = len(daily_data_df[daily_data_df[col] < 0])
            daily_data_df[col][daily_data_df[col] < 0] = 0
            print("{} negative values in daily data for col {}".format(negative_values, col))

    return daily_data_df


def compute_tukey_parameters(data_col):
    alpha = 1.5

    sorted_col_data = np.array(data_col.sort_values())
    n = len(sorted_col_data)

    Q1_idx = math.ceil(n * 0.25)
    Q3_idx = math.ceil(n * 0.75)

    IQR = sorted_col_data[Q3_idx] - sorted_col_data[Q1_idx]

    lower_threshold = sorted_col_data[Q1_idx] - alpha * IQR
    upper_threshold = sorted_col_data[Q3_idx] + alpha * IQR

    return lower_threshold, upper_threshold, IQR


def remove_outliers(data, cols_to_consider=['tot_cases', 'tot_death'], keep_zeros=True, keep_outliers=False):
    """
    Remove non-zero outliers from the data
    :param data: Main dataframe
    :param cols_to_consider: Columns to consider while removing outliers
    :param keep_zeros: Flag to check if zero valued outliers should be kept
    :param keep_outliers: Flag to check if outliers should not be removed

    :return: Dataframe with clean data
    """
    outlier_idx = set()
    clean_df = pd.DataFrame()

    for col in data:
        if col not in cols_to_consider:
            clean_df[col] = data[col]
            continue

        clean_df[col] = data[col]

        # remove outliers based on tukey's rule
        lower_threshold, upper_threshold, IQR = compute_tukey_parameters(data[col])
        column_outliers = clean_df[col][((clean_df[col] < lower_threshold) | (clean_df[col] > upper_threshold))].index

        # we want to remove non-zero outliers
        if keep_zeros:
            zero_outliers = clean_df[col][clean_df[col] == 0].index
            column_outliers = column_outliers.difference(zero_outliers)

        outlier_idx = outlier_idx.union(column_outliers)
        print("Col: {}, Outlier count: {}, lower_threshold: {}, upper_threshold: {}, IQR: {}".format(
            col, len(column_outliers), lower_threshold, upper_threshold, IQR))

    if not keep_outliers:
        print("Total outlier rows removed: {}".format(len(outlier_idx)))
        clean_df.drop(outlier_idx, inplace=True)

    return clean_df
