import csv
import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
from numpy import dtype
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def part_e_d():
    #read the cleaned vaccination dataset for florida
    df_fl = pd.read_csv('./processed/clean_fl_vax.csv')

    #read the cleaned vaccination dataset for Connecticut
    df_ct = pd.read_csv('./processed/clean_ct_vax.csv')
    df_fl.Date = pd.to_datetime(df_fl.Date)

    df_fl = df_fl.sort_values(by='Date')
    df_ct.Date = pd.to_datetime(df_ct.Date)
    df_ct = df_ct.sort_values(by='Date')

    #get number of covid vaccinations administered for month of may
    may_df_FL = df_fl[(df_fl['Date'] >= '2021-05-01') & (df_fl['Date'] <= '2021-05-31')]
    may_df_CT = df_ct[(df_ct['Date'] >= '2021-05-01') & (df_ct['Date'] <= '2021-05-31')]
    may_FL = may_df_FL[['Date','Administered']]
    may_CT = may_df_CT[['Date','Administered']]

    #get number of vaccines administered for month of september
    sept_df_FL = df_fl[(df_fl['Date'] >= '2021-09-01') & (df_fl['Date'] <= '2021-09-30') ]
    sept_df_CT = df_ct[(df_ct['Date'] >= '2021-09-01') & (df_ct['Date'] <= '2021-09-30') & (df_ct['Date'] != '2021-09-08')]
    sept_FL = sept_df_FL[['Administered']]
    sept_CT = sept_df_CT[['Administered']]

    #get number of vaccines administered for month of november
    nov_df_FL = df_fl[(df_fl['Date'] >= '2021-11-01') & (df_fl['Date'] <= '2021-11-30') & (df_fl['Date'] != '2021-11-29')]
    nov_df_CT = df_ct[(df_ct['Date'] >= '2021-11-01') & (df_ct['Date'] <= '2021-11-30')]
    nov_FL = nov_df_FL[['Administered']]
    nov_CT = nov_df_CT[['Administered']]

    #calculating T value for paired T-test
    def paired_T_test(data1, data2):
        D = data1.to_numpy() - data2.to_numpy()
        D = list(D)
        mean = np.mean(D)
        n = len(D)
        sd = math.sqrt(np.var(D))
        T = mean/(sd/math.sqrt(n))
        return T
    print('T value for Paired T-test for means of number of vaccines administered between Florida state and Connecticut State for september: ',paired_T_test(sept_FL,sept_CT))
    print('T value for Paired T-test for means of number of vaccines administered between Florida state and Connecticut State for november: ',paired_T_test(nov_FL,nov_CT))

    #function to calculate MSE
    def MSE(Y_act, Y_pred):
        SSE = 0
        Y_sse = Y_act-Y_pred
        for i in range(0,len(Y_pred)):
            SSE = SSE + Y_sse[i]*Y_sse[i]
        return SSE/len(Y_act)

    #function to calculate MAPE
    def MAPE(Y_act,Y_pred):
        err = (abs(Y_act-Y_pred))/Y_act
        MAPE = 0
        for i in range(0, len(Y_act)):
            MAPE = MAPE + err[i]
        return 100*MAPE/len(Y_act)

    #function to calculate Beta
    def Beta(x,y):
        return np.matmul(np.matmul(inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y)

    #function to calculate auto regression
    def AR(data, p):
        n_data = data.to_numpy()
        x_train = []
        y_train = []
        y_pred = []
        n_3 = n_data[:-7]
        n_1 = n_data[-7:]
        y_act = n_1

        for i in range(len(n_3)-p):
            x_train = x_train + [n_3[i:i+p]]
            y_train = y_train + [n_3[i+p]]
        one_app = np.ones((len(x_train),),dtype=int)
        one_app = np.reshape(one_app,(len(one_app),1))
        x_train = np.append(one_app , x_train, axis = 1)
        beta = Beta(x_train,y_train)
        x = []
        for i in range(len(n_data)-7,len(n_data)):
            x = [n_data[i-p:i]]
            x = np.append([1],x)
            y = np.matmul(x,beta)
            y_pred = y_pred + [y]
        y_pred = np.reshape(y_pred,(len(y_act),))
        return y_pred,y_act


    #AR results for florida state
    y_pred_Fl_AR_a, y_act_Fl_AR_a = AR(may_FL['Administered'],5)
    print("FLorida state : MSE of AR with p = 5 is " ,MSE(y_act_Fl_AR_a, y_pred_Fl_AR_a))
    print("FLorida state : MAPE of AR with p = 5 is" ,MAPE(y_act_Fl_AR_a, y_pred_Fl_AR_a))

    y_pred_Fl_AR_b, y_act_Fl_AR_b = AR(may_FL['Administered'],3)
    print("FLorida state : MSE of AR with p = 3 is " ,MSE(y_act_Fl_AR_b, y_pred_Fl_AR_b))
    print("FLorida state : MAPE of AR with p = 3 is" ,MAPE(y_act_Fl_AR_b, y_pred_Fl_AR_b))

    #plot for florida
    X = [i for i in range(1,8)]
    plt.figure('AR_f')
    plt.plot(X, y_act_Fl_AR_a ,label='Actual')
    plt.plot(X, y_pred_Fl_AR_b ,label='AR p = 3')
    plt.plot(X, y_pred_Fl_AR_a ,label='AR p = 5')
    plt.xlabel('Days in the last week of MAY')
    plt.ylabel('No. of vaccines administered')
    plt.title('AR prediction on Administered Vaccines in FLORIDA')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('./plots/AR_prediction_florida.png')

    #AR results for Connecticut state
    y_pred_CT_AR_a, y_act_CT_AR_a = AR(may_CT['Administered'],5)
    print("Connecticut state : MSE of AR with p = 5 is " ,MSE(y_act_CT_AR_a, y_pred_CT_AR_a))
    print("Connecticut state : MAPE of AR with p = 5 is" ,MAPE(y_act_CT_AR_a, y_pred_CT_AR_a))

    y_pred_CT_AR_b, y_act_CT_AR_b = AR(may_CT['Administered'],3)
    print("Connecticut state : MSE of AR with p = 3 is " ,MSE(y_act_CT_AR_b, y_pred_CT_AR_b))
    print("Connecticut state : MAPE of AR with p = 3 is" ,MAPE(y_act_CT_AR_b, y_pred_CT_AR_b))

    #plot for Connecticut
    X = [i for i in range(1,8)]
    plt.figure('AR_c')
    plt.plot(X, y_act_CT_AR_a ,label='Actual')
    plt.plot(X, y_pred_CT_AR_b ,label='AR p = 3')
    plt.plot(X, y_pred_CT_AR_a ,label='AR p = 5')
    plt.xlabel('Days in the last week of MAY')
    plt.ylabel('No. of vaccines administered')
    plt.title('AR prediction on Administered Vaccines in Connecticut')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('./plots/AR_prediction_Connecticut.png')

    #function to calculate EWMA
    def EWMA(data, alpha):
        n_data = data.to_numpy()
        Y_pred = [n_data[0]]
        for i in range(1,len(n_data)):
            Y_pred = np.append(Y_pred , [(alpha*n_data[i-1] +(1-alpha)*Y_pred[i-1])])
        Y_pred,Y_act = np.array(Y_pred[-7:]),np.array(n_data[-7:])
        return Y_pred,Y_act

    #EWMA results for florida with alpha = 0.5
    y_pred_Fl_EWMA_a, y_act_Fl_EWMA_a = EWMA(may_FL['Administered'],0.5)
    print("FLorida state : MSE of EWMA with alpha = 0.5 is " ,MSE(y_act_Fl_EWMA_a, y_pred_Fl_EWMA_a))
    print("FLorida state : MAPE of EWMA with alpha = 0.5 is" ,MAPE(y_act_Fl_EWMA_a, y_pred_Fl_EWMA_a))

    #EWMA results for florida with alpha = 0.8
    y_pred_Fl_EWMA_b, y_act_Fl_EWMA_b = EWMA(may_FL['Administered'],0.8)
    print("FLorida state : MSE of EWMA with alpha = 0.8 is " ,MSE(y_act_Fl_EWMA_b, y_pred_Fl_EWMA_b))
    print("FLorida state : MAPE of EWMA with alpha = 0.8 is" ,MAPE(y_act_Fl_EWMA_b, y_pred_Fl_EWMA_b))

    #plot for EWMA for Florida state
    X = [i for i in range(1,8)]
    plt.figure('EWMA_f')
    plt.plot(X, y_act_Fl_EWMA_a ,label='Actual')
    plt.plot(X, y_pred_Fl_EWMA_b ,label='EWMA 0.8 pred')
    plt.plot(X, y_pred_Fl_EWMA_a ,label='EWMA 0.5 pred')
    plt.xlabel('Days in the last week of MAY')
    plt.ylabel('No. of vaccines administered')
    plt.title('EWMA prediction on Administered Vaccines in FLORIDA')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('./plots/EWMA_prediction_florida.png')

    #EWMA results for co nneticut with alpha = 0.5
    y_pred_CT_EWMA_a, y_act_CT_EWMA_a = EWMA(may_CT['Administered'],0.5)
    print("Connecticut state : MSE of EWMA with alpha = 0.5 is " ,MSE(y_act_CT_EWMA_a, y_pred_CT_EWMA_a))
    print("Connecticut state : MAPE of EWMA with alpha = 0.5 is" ,MAPE(y_act_CT_EWMA_a, y_pred_CT_EWMA_a))

    #EWMA results for Connecticut with alpha = 0.8
    y_pred_CT_EWMA_b, y_act_CT_EWMA_b = EWMA(may_CT['Administered'],0.8)
    print("Connecticut state : MSE of EWMA with alpha = 0.8 is " ,MSE(y_act_CT_EWMA_b, y_pred_CT_EWMA_b))
    print("Connecticut state : MAPE of EWMA with alpha = 0.8 is" ,MAPE(y_act_CT_EWMA_b, y_pred_CT_EWMA_b))

    #plot for EWMA for Connecticut state
    X = [i for i in range(1,8)]
    plt.figure('EWMA_c')
    plt.plot(X, y_act_CT_EWMA_a ,label='Actual')
    plt.plot(X, y_pred_CT_EWMA_b ,label='EWMA 0.8 pred')
    plt.plot(X, y_pred_CT_EWMA_a ,label='EWMA 0.5 pred')
    plt.xlabel('Days in the last week of MAY')
    plt.ylabel('No. of vaccines administered')
    plt.title('EWMA prediction on Administered Vaccines in Connecticut')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('./plots/EWMA_prediction_Connecticut.png')

