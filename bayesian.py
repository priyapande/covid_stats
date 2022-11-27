import scipy.special
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


def analyze_ct(ct_data):
    # Week 1-4
    ct_data['daily_stats'] = ct_data['tot_cases'] + ct_data['tot_death']
    four_week_data = ct_data[ct_data['submission_date'] > '2020-05-31']
    four_week_data = four_week_data[four_week_data['submission_date'] < '2020-06-29']
    beta = 1 / four_week_data['daily_stats'].mean()
    x = np.linspace(0, 250, 250)
    y1 = stats.gamma.pdf(x, a=1, scale=1 / beta)

    # Week 5
    fifth_week_data = ct_data[ct_data['submission_date'] > '2020-06-28']
    fifth_week_data = fifth_week_data[fifth_week_data['submission_date'] < '2020-07-06']
    alpha_1 = 1 + fifth_week_data['daily_stats'].sum()
    beta_1 = beta + len(fifth_week_data)
    y2 = stats.gamma.pdf(x, a=alpha_1, scale=1 / beta_1)

    # Week 6
    sixth_week_data = ct_data[ct_data['submission_date'] > '2020-07-06']
    sixth_week_data = sixth_week_data[sixth_week_data['submission_date'] < '2020-07-14']
    alpha_2 = alpha_1 + sixth_week_data['daily_stats'].sum()
    beta_2 = beta_1 + len(sixth_week_data)
    y3 = stats.gamma.pdf(x, a=alpha_2, scale=1 / beta_2)

    # Week 7
    seventh_week_data = ct_data[ct_data['submission_date'] > '2020-07-14']
    seventh_week_data = seventh_week_data[seventh_week_data['submission_date'] < '2020-07-22']
    alpha_3 = alpha_2 + seventh_week_data['daily_stats'].sum()
    beta_3 = beta_2 + len(seventh_week_data)
    y4 = stats.gamma.pdf(x, a=alpha_3, scale=1 / beta_3)

    # Week 8
    eightht_week_data = ct_data[ct_data['submission_date'] > '2020-07-22']
    eightht_week_data = eightht_week_data[eightht_week_data['submission_date'] < '2020-07-30']
    alpha_4 = alpha_3 + eightht_week_data['daily_stats'].sum()
    beta_4 = beta_3 + len(eightht_week_data)
    y5 = stats.gamma.pdf(x, a=alpha_4, scale=1 / beta_4)

    figure(figsize=(10, 5), dpi=100)
    plt.plot(x, y1, label='Prior - Exponiential distribution')
    plt.plot(x, y2, label='Posterior after week-5 (Gamma distribution)')
    plt.plot(x, y3, label='Posterior after week-6 (Gamma distribution)')
    plt.plot(x, y4, label='Posterior after week-7 (Gamma distribution)')
    plt.plot(x, y5, label='Posterior after week-8 (Gamma distribution)')
    plt.xlabel('Covid Cases')
    plt.ylabel('Pmf')

    # displaying the title
    plt.title("Posterior distributions for Connecticut covid cases")
    plt.legend()
    plt.savefig('./plots/CT_stats_posterior.png')
    y2 = y2.tolist()
    max_index = y2.index(max(y2))
    print("MAP for posterior after week 5:", max_index)

    y3 = y3.tolist()
    max_index = y3.index(max(y3))
    print("MAP for posterior after week 6:", max_index)

    y4 = y4.tolist()
    max_index = y4.index(max(y4))
    print("MAP for posterior after week 7:", max_index)

    y5 = y5.tolist()
    max_index = y5.index(max(y5))
    print("MAP for posterior after week 8:", max_index)


def analyze_fl(fl_data):
    # Week 1-4
    fl_data['daily_stats'] = fl_data['tot_cases'] + fl_data['tot_death']
    four_week_data = fl_data[fl_data['submission_date'] > '2020-05-31']
    four_week_data = four_week_data[four_week_data['submission_date'] < '2020-06-29']
    beta = 1 / four_week_data['daily_stats'].mean()
    x = np.linspace(8000, 11000, 3000)
    y1 = stats.gamma.pdf(x, a=1, scale=1 / beta)

    # Week 5
    fifth_week_data = fl_data[fl_data['submission_date'] > '2020-06-28']
    fifth_week_data = fifth_week_data[fifth_week_data['submission_date'] < '2020-07-06']
    alpha_1 = 1 + fifth_week_data['daily_stats'].sum()
    beta_1 = beta + len(fifth_week_data)
    y2 = stats.gamma.pdf(x, a=alpha_1, scale=1 / beta_1)

    # Week 6
    sixth_week_data = fl_data[fl_data['submission_date'] > '2020-07-06']
    sixth_week_data = sixth_week_data[sixth_week_data['submission_date'] < '2020-07-14']
    alpha_2 = alpha_1 + sixth_week_data['daily_stats'].sum()
    beta_2 = beta_1 + len(sixth_week_data)
    y3 = stats.gamma.pdf(x, a=alpha_2, scale=1 / beta_2)

    # Week 7
    seventh_week_data = fl_data[fl_data['submission_date'] > '2020-07-14']
    seventh_week_data = seventh_week_data[seventh_week_data['submission_date'] < '2020-07-22']
    alpha_3 = alpha_2 + seventh_week_data['daily_stats'].sum()
    beta_3 = beta_2 + len(seventh_week_data)
    y4 = stats.gamma.pdf(x, a=alpha_3, scale=1 / beta_3)

    # Week 8
    eightht_week_data = fl_data[fl_data['submission_date'] > '2020-07-22']
    eightht_week_data = eightht_week_data[eightht_week_data['submission_date'] < '2020-07-30']
    alpha_4 = alpha_3 + eightht_week_data['daily_stats'].sum()
    beta_4 = beta_3 + len(eightht_week_data)
    y5 = stats.gamma.pdf(x, a=alpha_4, scale=1 / beta_4)

    figure(figsize=(10, 5), dpi=100)
    plt.plot(x, y1, label='Prior - Exponiential distribution')
    plt.plot(x, y2, label='Posterior after week-5 (Gamma distribution)')
    plt.plot(x, y3, label='Posterior after week-6 (Gamma distribution)')
    plt.plot(x, y4, label='Posterior after week-7 (Gamma distribution)')
    plt.plot(x, y5, label='Posterior after week-8 (Gamma distribution)')
    plt.xlabel('Covid Cases')
    plt.ylabel('Pmf')

    # displaying the title
    plt.title("Posterior distributions for Florida covid cases")
    plt.legend()
    plt.savefig('./plots/FL_stats_posterior.png')
    y2 = y2.tolist()
    max_index = y2.index(max(y2))
    print("MAP for posterior after week 5:", max_index + 8000)

    y3 = y3.tolist()
    max_index = y3.index(max(y3))
    print("MAP for posterior after week 6:", max_index + 8000)

    y4 = y4.tolist()
    max_index = y4.index(max(y4))
    print("MAP for posterior after week 7:", max_index + 8000)

    y5 = y5.tolist()
    max_index = y5.index(max(y5))
    print("MAP for posterior after week 8:", max_index + 8000)