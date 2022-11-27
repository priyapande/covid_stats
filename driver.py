from preprocessing import get_state_data, get_daily_cases_data, remove_outliers
from two_ks_test import two_sample_KS_test
from permutation_test import permutation_test
from one_ks_test import one_sample_KS_test
from hypothesis_test_2a import run_hyp_tests
from bayesian import analyze_ct, analyze_fl
from ar_ewma_pairedTtest import part_e_d

# States allocates: Connecticut (CT) and Florida (FL)


def process_cases_data():
    covid_cases_data_path = './dataset/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv'

    ct_state_cases_data, fl_state_cases_data = get_state_data(filename=covid_cases_data_path,
                                                              states=['CT', 'FL'],
                                                              location_col_name='state',
                                                              cols=['submission_date', 'state',
                                                                    'tot_cases', 'tot_death',
                                                                    'new_case', 'new_death'])

    ct_daily_cases_data = get_daily_cases_data(ct_state_cases_data,
                                               location_col_name='state',
                                               date_col_name='submission_date',
                                               non_cumulative_cols=['new_case', 'new_death'],
                                               set_zero_for_negatives=True)
    fl_daily_cases_data = get_daily_cases_data(fl_state_cases_data,
                                               location_col_name='state',
                                               date_col_name='submission_date',
                                               non_cumulative_cols=['new_case', 'new_death'],
                                               set_zero_for_negatives=True)

    ct_daily_cleaned_data = remove_outliers(ct_daily_cases_data)
    fl_daily_cleaned_data = remove_outliers(fl_daily_cases_data)
    return ct_daily_cleaned_data, fl_daily_cleaned_data


def process_vax_data():
    covid_vax_data_path = './dataset/COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv'

    ct_state_vax_data, fl_state_vax_data = get_state_data(filename=covid_vax_data_path,
                                                          states=['CT', 'FL'],
                                                          location_col_name='Location',
                                                          cols=['Date', 'Location',
                                                                'Administered'])

    ct_daily_vax_data = get_daily_cases_data(ct_state_vax_data,
                                             location_col_name='Location',
                                             date_col_name='Date',
                                             non_cumulative_cols=[],
                                             set_zero_for_negatives=True)
    fl_daily_vax_data = get_daily_cases_data(fl_state_vax_data,
                                             location_col_name='Location',
                                             date_col_name='Date',
                                             non_cumulative_cols=[],
                                             set_zero_for_negatives=True)

    ct_daily_cleaned_vax_data = remove_outliers(ct_daily_vax_data,
                                                cols_to_consider=['Administered'])
    fl_daily_cleaned_vax_data = remove_outliers(fl_daily_vax_data,
                                                cols_to_consider=['Administered'])
    return ct_daily_cleaned_vax_data, fl_daily_cleaned_vax_data


def get_data_for_date_range(data, start_date, end_date, date_col_name):
    return data[(data[date_col_name] >= start_date) & (data[date_col_name] <= end_date)].reset_index(drop=True)


if __name__ == "__main__":
    print("-----------Part 1--------------")
    print("-----------Data cleaning for cases and death statistics--------------")
    # Mandatory Task 1: To clean the given dataset for cases
    ct_daily_cleaned_data, fl_daily_cleaned_data = process_cases_data()

    ct_daily_cleaned_data.to_csv('./processed/clean_ct_cases.csv')
    fl_daily_cleaned_data.to_csv('./processed/clean_fl_cases.csv')

    print("-----------Data cleaning for vaccination statistics--------------")
    ct_daily_cleaned_vax_data, fl_daily_cleaned_vax_data = process_vax_data()

    ct_daily_cleaned_vax_data.to_csv('./processed/clean_ct_vax.csv')
    fl_daily_cleaned_vax_data.to_csv('./processed/clean_fl_vax.csv')

    print("\n\n-----------Part 2a--------------")
    run_hyp_tests()

    print("\n\n-----------Part 2b--------------")
    # Mandatory Task 2b: To infer equality of distributions
    start_date = '2021-10-01'
    end_date = '2021-12-31'
    ct_last_quarter_cases = get_data_for_date_range(ct_daily_cleaned_data, start_date, end_date, 'submission_date')
    fl_last_quarter_cases = get_data_for_date_range(fl_daily_cleaned_data, start_date, end_date, 'submission_date')

    print("\n\n----------- Two-sample KS test--------------")
    print("\n----------- Cases --------------")
    two_sample_KS_test(ct_last_quarter_cases, fl_last_quarter_cases, 'tot_cases')
    print("\n----------- Deaths --------------")
    two_sample_KS_test(ct_last_quarter_cases, fl_last_quarter_cases, 'tot_death')

    print("\n\n----------- Permutation test--------------")
    print("\n----------- Cases --------------")
    permutation_test(ct_last_quarter_cases, fl_last_quarter_cases, 'tot_cases')
    print("\n----------- Deaths --------------")
    permutation_test(ct_last_quarter_cases, fl_last_quarter_cases, 'tot_death')

    print("\n\n----------- One-sample KS test--------------")
    print("\n----------- Cases --------------")
    one_sample_KS_test(ct_last_quarter_cases, fl_last_quarter_cases, 'tot_cases')
    print("\n----------- Deaths --------------")
    one_sample_KS_test(ct_last_quarter_cases, fl_last_quarter_cases, 'tot_death')

    print("\n\n-----------Part 2c--------------")
    # Mandatory Task 2c: Bayesian inference
    print("\n----------- Connecticut stats --------------")
    analyze_ct(ct_daily_cleaned_data)
    print("\n----------- Florida stats --------------")
    analyze_fl(fl_daily_cleaned_data)

    print("\n\n-----------Part 2d and 2e--------------")
    part_e_d()

