from permutation_test import compute_sample_mean
from two_ks_test import get_ecdf_for_point, generate_ecdf
from scipy.stats import poisson
from scipy.stats import geom
from scipy.stats import binom
import pandas as pd


def compute_sample_variance(data):
    mean = compute_sample_mean(data)
    sum_val = 0
    for val in data:
        sum_val += (val - mean) ** 2
    return sum_val / len(data)


def get_poisson_mme(data):
    # lambda_mme = mean
    mean = compute_sample_mean(data)
    print("Poisson: \nMME value for lambda: {}".format(mean))
    return mean


def get_geometric_mme(data):
    # p_mme = 1/mean
    p = 1 / compute_sample_mean(data)
    print("Geometric: \nMME value for p: {}".format(p))
    return p


def get_binomial_mme(data):
    # n_mme = mean^2/(mean-variance)
    # p_mme = mean/n_mme
    mean = compute_sample_mean(data)
    var = compute_sample_variance(data)

    n_mme = mean ** 2 / (mean - var)
    p_mme = mean / n_mme
    print("Binomial: \nMME value for n: {}, p: {}".format(n_mme, p_mme))
    return [n_mme, p_mme]


def estimate_parameters(data, dist):
    if dist == 'Poisson':
        return get_poisson_mme(data)
    elif dist == 'Geometric':
        return get_geometric_mme(data)
    elif dist == 'Binomial':
        return get_binomial_mme(data)
    return None


def compute_one_ks_pvalue(x_vals, params, dist_name, data, col_name):
    ks_values = []

    # compute table column values for each x
    for x in x_vals:
        # ecdf to the left of x for state 2's distribution
        F_left = get_ecdf_for_point(data, col_name, x, 'eCDF', left=True)
        # ecdf to the right of x for state 1's distribution
        F_right = get_ecdf_for_point(data, col_name, x, 'eCDF')

        # Fx is the cdf of distribution at x
        if dist_name == 'Poisson':
            lambda_mme = params
            Fx = poisson.cdf(x, lambda_mme)

        elif dist_name == 'Geometric':
            p_mme = params
            Fx = geom.cdf(x, p_mme)

        elif dist_name == 'Binomial':
            n_mme, p_mme = params
            if n_mme < 0 or p_mme < 0:
                print("Invalid MME param for distribution: {}, n_mme: {}, p_mme: {}".format(dist_name, n_mme, p_mme))
                return
            Fx = binom.cdf(x, n_mme, p_mme)
        else:
            print("Unsupported distributed for one sample KS test: {}".format(dist_name))
            return
        # difference between left ecdf and Fx values
        abs_diff_left = round(abs(F_left - Fx), 4)
        # difference between right ecdf and Fx values
        abs_diff_right = round(abs(F_right - Fx), 4)

        ks_values.append(
            {'x': x,
             'F_left': F_left,
             'F_right': F_right,
             'Fx': Fx,
             'abs_diff_left': abs_diff_left,
             'abs_diff_right': abs_diff_right
             })

    # table for all KS values
    table_ks = pd.DataFrame(ks_values, columns=['x', 'F_left', 'F_right', 'Fx',
                                                'abs_diff_left', 'abs_diff_right'])

    # Calculate KS statistic
    # max of right ecdf values
    max_d_right = table_ks['abs_diff_right'].max()
    # max of left ecdf values
    max_d_left = table_ks['abs_diff_left'].max()
    # max of both values
    d = max(max_d_right, max_d_left)
    critical_value = 0.05
    if d > critical_value:
        print(
            "One sample KS Test for col: {} rejects the null hypothesis.\nThe statistic d is {}, which is more than the critical-value: {}".format(
                col_name, d, critical_value
            ))
        print()
    else:
        print(
            "One sample KS Test for col: {} accepts the null hypothesis.\nThe statistic d is {}, which is less than the critical-value: {}".format(
                col_name, d, critical_value
            ))
        print()


def one_sample_KS_test(state1_data, state2_data, col_name, distributions=['Poisson', 'Geometric', 'Binomial']):
    for dist in distributions:
        # use state1_data for the column to estimate distribution parameter (MME)
        params = estimate_parameters(state1_data[col_name], dist)

        # Take data from state 2 for KS test
        # sort the data by column on which we are performing the KS Test
        state2_data_sorted = state2_data.sort_values(col_name)
        # compute ecdf for the distribution
        state2_data_sorted['eCDF'] = generate_ecdf(state2_data_sorted.shape[0])
        # we keep last value for each x point since that is the one with final ecdf value for that point
        state2_data_sorted_distinct = state2_data_sorted.drop_duplicates(subset=col_name, keep="last").reset_index(
            drop=True)

        x_vals = state2_data_sorted_distinct[col_name].to_numpy()
        compute_one_ks_pvalue(x_vals, params, dist, state2_data_sorted_distinct, col_name)
