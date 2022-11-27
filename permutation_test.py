import random

import numpy as np

random.seed(42)


def compute_sample_mean(data):
    sum_val = 0
    for val in data:
        sum_val += val
    return sum_val / len(data)


def compute_statistic_for_permutations(num_permutations, data1, data2):
    d1 = data1.to_list()
    d2 = data2.to_list()

    d1_size = len(d1)

    # combine both datasets
    combined_data = d1 + d2

    # computed statistic values
    T_i_list = []

    for permutation in range(num_permutations):
        permuted_data = np.random.permutation(combined_data)

        # divide the permuted list into two parts of size proportionate to d1 and d2
        d1_new_mean = compute_sample_mean(permuted_data[:d1_size])
        d2_new_mean = compute_sample_mean(permuted_data[d1_size:])

        T_i = abs(d1_new_mean - d2_new_mean)
        T_i_list.append(T_i)
    return T_i_list


def permutation_test(state1_data, state2_data, col_name):
    # compute the mean for both sample sets
    state1_mean = compute_sample_mean(state1_data[col_name])
    state2_mean = compute_sample_mean(state2_data[col_name])

    # initial statistic value
    T_obs = abs(state1_mean - state2_mean)

    # compute the statistic for permutations of data
    T_i_list = compute_statistic_for_permutations(1000, state1_data[col_name], state2_data[col_name])

    num_extreme_vals = 0
    for T_i in T_i_list:
        if T_i > T_obs:
            num_extreme_vals += 1

    p_val = num_extreme_vals / 1000
    critical_value = 0.05
    if p_val <= critical_value:
        print(
            "Permutation test for col: {} rejects the null hypothesis.\nT-obs is {} and p-value is {}, which is less than the critical-value: {}".format(
                col_name, T_obs, p_val, critical_value
            ))
    else:
        print(
            "Permutation test for col: {} accepts the null hypothesis.\nT-obs is {} and p-value is {}, which is more than the critical-value: {}".format(
                col_name, T_obs, p_val, critical_value
            ))
