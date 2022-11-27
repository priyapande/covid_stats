import pandas as pd


def generate_ecdf(n):
    ecdf = []
    ecdf.append(0)
    for i in range(n - 1):
        ecdf.append(ecdf[-1] + (1 / n))
    return ecdf


def get_ecdf_for_point(data, col_name, x, ecdf_col_name, left=False):
    # point lies to the right of entire distribution
    if data[col_name].max() < x:
        return 1
    # point lies to the left of entire distribution
    elif data[col_name].min() > x:
        return 0
    else:
        if left:
            # find ecdf for point that is just before x
            ecdf = data.loc[data[col_name] < x, ecdf_col_name]
            val = 0.0 if ecdf.empty else ecdf.max()
        else:
            # find ecdf for point that is just after x
            ecdf = data.loc[data[col_name] >= x, ecdf_col_name]
            val = 0.0 if ecdf.empty else ecdf.min()
        return val


def two_sample_KS_test(state1_data, state2_data, col_name):
    # sort the data by column on which we are performing the KS Test
    state1_data_sorted = state1_data.sort_values(col_name)
    state2_data_sorted = state2_data.sort_values(col_name)

    # compute ecdf for each distribution
    state1_data_sorted['eCDF'] = generate_ecdf(state1_data_sorted.shape[0])
    state2_data_sorted['eCDF'] = generate_ecdf(state2_data_sorted.shape[0])

    # we keep last value for each x point since that is the one with final ecdf value for that point
    state1_data_sorted_distinct = state1_data_sorted.drop_duplicates(subset=col_name, keep="last").reset_index(
        drop=True)
    state2_data_sorted_distinct = state2_data_sorted.drop_duplicates(subset=col_name, keep="last").reset_index(
        drop=True)

    # Take x points from state 2
    x_vals = state2_data_sorted_distinct[col_name].to_numpy()

    ks_values = []

    # compute table column values for each x
    for x in x_vals:
        # ecdf to the left of x for state 1's distribution
        F1_left = get_ecdf_for_point(state1_data_sorted_distinct, col_name, x, 'eCDF', left=True)
        # ecdf to the right of x for state 1's distribution
        F1_right = get_ecdf_for_point(state1_data_sorted_distinct, col_name, x, 'eCDF')
        # ecdf to the left of x for state 2's distribution
        F2_left = get_ecdf_for_point(state2_data_sorted_distinct, col_name, x, 'eCDF', left=True)
        # ecdf to the right of x for state 2's distribution
        F2_right = get_ecdf_for_point(state2_data_sorted_distinct, col_name, x, 'eCDF')
        # difference between corresponding left ecdf values
        abs_diff_left = round(abs(F1_left - F2_left), 4)
        # difference between corresponding right ecdf values
        abs_diff_right = round(abs(F1_right - F2_right), 4)

        ks_values.append(
            {'x': x,
             'F1_left': F1_left,
             'F1_right': F1_right,
             'F2_left': F2_left,
             'F2_right': F2_right,
             'abs_diff_left': abs_diff_left,
             'abs_diff_right': abs_diff_right
             })

    # table for all KS values
    table_ks = pd.DataFrame(ks_values, columns=['x', 'F1_left', 'F1_right', 'F2_left', 'F2_right',
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
            "KS Test for col: {} rejects the null hypothesis.\nThe statistic d is {}, which is more than the critical-value: {}".format(
                col_name, d, critical_value
            ))
        print()
    else:
        print(
            "KS Test for col: {} accepts the null hypothesis.\nThe statistic d is {}, which is less than the critical-value: {}".format(
                col_name, d, critical_value
            ))
        print()
