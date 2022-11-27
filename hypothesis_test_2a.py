import math
import pandas as pd
from scipy.stats import norm, t


def filter_time(df, month):
    if month == 3:
        f1 = df['submission_date'] >= pd.Timestamp(2021, 3, 1)
        f2 = df['submission_date'] <= pd.Timestamp(2021, 3, 31)
        return f1 & f2
    elif month == 2:
        f1 = df['submission_date'] >= pd.Timestamp(2021, 2, 1)
        f2 = df['submission_date'] <= pd.Timestamp(2021, 2, 28)
        return f1 & f2
    return []


def get_mean(df, col):
    return df[col].sum() / df[col].shape[0]


def walds_one_sample_test(true_df, predicted_df, column, state, alpha=0.05):
    """
        Statistic for Wald's 1 sample test
        W = theta_cap - theta_knot / standard_error(theta_cap)
        :param true_df: Series of data which is hypothesized as the true mean
        :param predicted_df:  Series of data against which to test the hypothesis
   """
    print("**** WALD'S ONE SAMPLE TEST ****")
    print(
        "Testing null hypothesis that the mean of daily {0}s in Feb 2021 is same as March 2021 for state - {1}".format(
            column, state))
    theta_knot = get_mean(true_df, column)
    theta_cap = get_mean(predicted_df, column)
    # given that data is a poisson distribution, MLE is the same as sample mean which is our parameter lambda
    lambda_est_mle = theta_cap
    # variance of poisson distribution is lambda
    # from class, after simplifying for estimator std_err = root(variance/n)
    standard_error = math.sqrt(lambda_est_mle / predicted_df.shape[0])
    W = abs(round((theta_cap - theta_knot) / standard_error, 2))
    p_val = round(norm.ppf(1 - (alpha / 2)), 2)
    print("True Mean = {0:.2f}, Sample Mean = {1:.2f}, Standard Error = {2:.2f}".format(theta_knot, theta_cap,
                                                                                        standard_error))
    if W > p_val:
        print(
            "Rejected null hypothesis as the Walds Statistic {0} is greater than the critical value {1} specified.".format(
                W, p_val))
    else:
        print(
            "Accepted null hypothesis as the Walds Statistic {0} is less than or equal to the critical value {1} specified.".format(
                W, p_val))
    print()


def walds_two_sample_test(true_df, predicted_df, column, state, alpha=0.05):
    """
        Statistic for Wald's two sample test
        W = delta_cap/ standard_error(delta_cap), where delta = sample_mean1 - sample_mean2
   """
    print("**** WALD'S TWO SAMPLE TEST ****")
    print(
        "Testing null hypothesis that the difference in mean of daily {0}s in Feb 2021 and March 2021 is zero for state - {1}".format(
            column, state))
    delta_0 = get_mean(true_df, column)
    delta_1 = get_mean(predicted_df, column)
    # similar to one sample Wald's test we get lambda_MLE as the sample mean which is also the variance for poisson distribution
    # from class, we know that std_err = root(var1/n + var2/m) after simplification
    standard_error = math.sqrt((delta_0 / true_df.shape[0]) + (delta_1 / predicted_df.shape[0]))
    W = abs(round((delta_1 - delta_0) / standard_error, 2))
    p_val = round(norm.ppf(1 - (alpha / 2)), 2)
    print("Mean X = {0:.2f}, Mean Y = {1:.2f}, Standard Error = {2:.2f}".format(delta_0, delta_1, standard_error))
    if W > p_val:
        print(
            "Rejected null hypothesis as the Wald's Statistic {0} is greater than the critical value {1} specified.".format(
                W, p_val))
    else:
        print(
            "Accepted null hypothesis as the Wald's Statistic {0} is less than or equal to the critical value {1} specified.".format(
                W, p_val))
    print()


def t_test(true_df, predicted_df, column, state, alpha=0.05):
    """
        Statistic for T-Test two sample test
        T = Sample Mean - True Mean/ std_dev_corr/sqrt(n),
        where std_dev_corr = sqrt(sum(X - sample_mean)^2 / n - 1)
    """
    print("**** T TEST ****")
    print(
        "Testing null hypothesis that the mean of daily {0}s in Feb 2021 is same as March 2021 for state - {1}".format(
            column, state))
    true_mean = get_mean(true_df, column)
    sample_mean = get_mean(predicted_df, column)
    std_dev = math.sqrt(((predicted_df[column] - sample_mean) ** 2).sum() / predicted_df.shape[0] - 1)
    T = abs(round((sample_mean - true_mean) / (std_dev / math.sqrt(predicted_df.shape[0])), 3))
    # Looking up in T table and keeping degree of freedom as n-1
    p_val = round(t.ppf(1 - (alpha / 2), df=predicted_df.shape[0] - 1), 2)
    print("True Mean = {0:.2f}, Sample Mean = {1:.2f}, Standard Deviation = {2:.2f}".format(true_mean, sample_mean,
                                                                                            std_dev))
    if T > p_val:
        print(
            "Rejected null hypothesis as the T Statistic {0} is greater than the critical value {1} specified.".format(
                T, p_val))
    else:
        print(
            "Accepted null hypothesis as the T Statistic {0} is less than or equal to the critical value {1} specified.".format(
                T, p_val))
    print()


def t_test_unpaired(X, Y, column, state, alpha=0.05):
    """
          Statistic for T-Test two sample test
          T = X_mean - Y_mean/ sqrt(std_dev1^2/n + std_dev2^2/m),
          where std_dev = sqrt(sum(X - sample_mean)^2 / n - 1)
    """
    print("**** T TEST UNPAIRED ****")
    print("Testing null hypothesis that the difference in mean of daily {0}s in Feb 2021 and March 2021 is zero for "
          "state - {1}".format(column, state))
    X_mean = get_mean(X, column)
    Y_mean = get_mean(Y, column)
    std_dev1_2 = ((X[column] - X_mean) ** 2).sum() / X.shape[0] - 1
    std_dev2_2 = ((Y[column] - Y_mean) ** 2).sum() / Y.shape[0] - 1
    pool_stddev = math.sqrt(std_dev1_2 / X.shape[0] + std_dev2_2 / Y.shape[0])
    T = abs(round((X_mean - Y_mean) / pool_stddev, 2))
    # Looking up in T table and keeping degree of freedom as m-1 + n-1 = m+n-2
    p_val = round(t.ppf(1 - (alpha / 2), df=X.shape[0] + Y.shape[0] - 2), 2)
    print("Mean X = {0:.2f}, Mean Y = {1:.2f}, Standard Deviation = {2:.2f}".format(X_mean, Y_mean, pool_stddev))
    if T > p_val:
        print(
            "Rejected null hypothesis as the T Statistic {0} is greater than the critical value {1} specified.".format(
                T, p_val))
    else:
        print(
            "Accepted null hypothesis as the T Statistic {0} is less than or equal to the critical value {1} specified.".format(
                T, p_val))
    print()


def z_test(true_df, predicted_df, df, column, state, alpha=0.05):
    """
          Statistic for Z-Test One Sample
          T = X_mean - mu_knot/ std_dev/sqrt(n),
          where std_dev is true standard deviation of the distribution
    """
    print("**** Z-TEST ****")
    print(
        "Testing null hypothesis that the mean of daily {0}s in Feb 2021 is same as March 2021 for state - {1}".format(
            column, state))
    mu_knot = get_mean(true_df, column)
    X_mean = get_mean(predicted_df, column)
    data_mean = get_mean(df, column)
    # the true uncorrected standard deviation for our dataset is calculated on
    # the entire dataset for that state without date filter
    std_dev = math.sqrt(((df[column] - data_mean) ** 2).sum() / df.shape[0])
    Z = abs(round((X_mean - mu_knot) / (std_dev / math.sqrt(predicted_df.shape[0])), 2))
    p_val = round(norm.ppf(1 - (alpha / 2)), 2)
    print("Mean X = {0:.2f}, Mean Y = {1:.2f}, Standard Deviation = {2:.2f}".format(X_mean, mu_knot, std_dev))
    if Z > p_val:
        print(
            "Rejected null hypothesis as the Z Statistic {0} is greater than the critical value {1} specified.".format(
                Z, p_val))
    else:
        print(
            "Accepted null hypothesis as the Z Statistic {0} is less than or equal to the critical value {1} specified.".format(
                Z, p_val))
    print()


def run_hyp_tests():
    # Full Dataset of each state
    data_ct = pd.read_csv("processed/clean_ct_cases.csv")
    data_fl = pd.read_csv("processed/clean_fl_cases.csv")
    data_ct['submission_date'] = pd.to_datetime(data_ct.submission_date)
    data_fl['submission_date'] = pd.to_datetime(data_fl.submission_date)

    # filtered on month feb and march
    feb_df_CT = data_ct[filter_time(data_ct, 2)]
    mar_df_CT = data_ct[filter_time(data_ct, 3)]

    feb_df_FL = data_fl[filter_time(data_fl, 2)]
    mar_df_FL = data_fl[filter_time(data_fl, 3)]

    print("---- Running Hypothesis Test ----")
    print()
    walds_one_sample_test(feb_df_FL, mar_df_FL, 'new_case', 'Florida')
    walds_one_sample_test(feb_df_FL, mar_df_FL, 'new_death', 'Florida')
    walds_one_sample_test(feb_df_CT, mar_df_CT, 'new_case', 'Connecticut')
    walds_one_sample_test(feb_df_CT, mar_df_CT, 'new_death', 'Connecticut')

    t_test(feb_df_FL, mar_df_FL, 'new_case', 'Florida')
    t_test(feb_df_FL, mar_df_FL, 'new_death', 'Florida')
    t_test(feb_df_CT, mar_df_CT, 'new_case', 'Connecticut')
    t_test(feb_df_CT, mar_df_CT, 'new_death', 'Connecticut')

    z_test(feb_df_FL, mar_df_FL, data_fl, 'new_case', 'Florida')
    z_test(feb_df_FL, mar_df_FL, data_fl, 'new_death', 'Florida')
    z_test(feb_df_CT, mar_df_CT, data_ct, 'new_case', 'Connecticut')
    z_test(feb_df_CT, mar_df_CT, data_ct, 'new_death', 'Connecticut')

    walds_two_sample_test(feb_df_FL, mar_df_FL, 'new_case', 'Florida')
    walds_two_sample_test(feb_df_FL, mar_df_FL, 'new_death', 'Florida')
    walds_two_sample_test(feb_df_CT, mar_df_CT, 'new_case', 'Connecticut')
    walds_two_sample_test(feb_df_CT, mar_df_CT, 'new_death', 'Connecticut')

    t_test_unpaired(feb_df_FL, mar_df_FL, 'new_case', 'Florida')
    t_test_unpaired(feb_df_FL, mar_df_FL, 'new_death', 'Florida')
    t_test_unpaired(feb_df_CT, mar_df_CT, 'new_case', 'Connecticut')
    t_test_unpaired(feb_df_CT, mar_df_CT, 'new_death', 'Connecticut')
