# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:13:24 2018

@author: johndaniels
"""

# import modules
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from dateutil import parser
import csv
from bisect import bisect_left
import math
from scipy import interpolate
from scipy.signal import medfilt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import scipy.io as sio

import time
import datetime


from time import mktime
from datetime import date


#################
out_dim = 1
n_prev  = 24

SAMPLES_IN_DAY = 288

# ---------------------------------------------------------
# NORMALIZING DATA FROM ABC4D
# --

def to_adverse_event(norm_reference, num_classes):
    sHypo_x   = 54  # 0.039 #3 clinically relevant hypoglycaemia
    Hypo_x    = 70  # 0.083 #3.89 typically suggested level for hypoglycaemia
    Hyper_x   = 180 # 0.389 #10 typically suggested level for hyperglycaemia
    sHyper_x  = 240 # 0.556 #13.33 hyperglycaemia level that may result in ketones in blood

    lb = -2     # lower bound for time of event classification (~10 mins)
    ub = +3     # upper bound for time of event classification (~10 mins)
    window = 5  # time period for event classification

    reference = norm_reference*18
    event_ref = []

    for index in range(window, len(reference)):

        snippet = reference[index-window:index]

        if(num_classes == 5):

            if(np.sum(np.logical_and(snippet >= Hypo_x, snippet < Hyper_x)) >= 3):
                event_ref.append(2)
            elif(np.sum(np.logical_and(snippet >= sHypo_x, snippet < Hypo_x)) >= 3):
                event_ref.append(1)
            elif(np.sum(np.logical_and(snippet >= 0, snippet < sHypo_x)) >= 3):
                event_ref.append(0)
            elif(np.sum(np.logical_and(snippet >= Hyper_x, snippet < sHyper_x)) >= 3):
                event_ref.append(3)
            elif(np.sum(np.logical_and(snippet >= sHyper_x, snippet < 400)) >= 3):
                event_ref.append(4)
            else:
                event_ref.append(event_ref[-1])


        else:

            if(np.sum(np.logical_and(snippet >= Hypo_x, snippet < Hyper_x)) >= 3):
                event_ref.append(1)
            elif(np.sum(np.logical_and(snippet < Hypo_x, snippet < Hypo_x)) >= 3):
                event_ref.append(0)
            elif(np.sum(np.logical_and(snippet > Hyper_x, snippet > Hyper_x)) >= 3):
                event_ref.append(2)
            else:
                event_ref.append(event_ref[-1])

    event_ref = np.array(event_ref)

    return event_ref

def to_adverse_hyper_event(reference, num_classes):

    adverse_events = to_adverse_event(reference, 5)

    if(num_classes == 2):
        np.place(adverse_events, adverse_events <= 2, [0])
        np.place(adverse_events, adverse_events >  2, [1])
        hyper_event = adverse_events
    else:
        np.place(adverse_events, adverse_events <= 2, [0])
        np.place(adverse_events, adverse_events == 3, [1])
        np.place(adverse_events, adverse_events == 4, [2])
        hyper_event = adverse_events

    return hyper_event

def to_adverse_hypo_event(reference, num_classes):

    adverse_events = to_adverse_event(reference, 5)

    if(num_classes == 2):
        np.place(adverse_events, adverse_events >= 2, [0])
        np.place(adverse_events, adverse_events <  2, [1])
        hypo_event = adverse_events
    else:
        hypo_event = adverse_events
        np.place(adverse_events, adverse_events >= 2, [2])
        np.place(adverse_events, adverse_events == 1, [1])
        np.place(adverse_events, adverse_events == 0, [0])
        hypo_event = adverse_events

    return hypo_event


def get_dataset(dataset, pID, is_training = True):

    if is_training == True:
        type = "train"
    else:
        type = "test"

    path = 'C:/Users/kaise/Projects/ARISES/DeepCGM/dataset/'+dataset+'/'+type+'/'
    df = pd.read_csv(path + pID + '.csv')
    df.fillna(0)

    impute = pd.read_csv(path +'impute_' +pID+ '.csv', header=None)
    ip     = np.squeeze(impute.values)

    return df, ip


def load_estimation_samples(n_prev, input, output, isrealBG_array):
    docX, docY, docIP = [], [], []

    for i in range(len(input) - (n_prev - 1)):
        docX.append(input[i:i+n_prev])
        docY.append(output[i + n_prev - 1])
        docIP.append(isrealBG_array[i + n_prev - 1])

    alsX = np.array(docX)
    alsY = np.array(docY)
    alsIP = np.array(docIP)

    return alsX, alsY, alsIP

def load_prediction_samples(n_prev, input, output, isrealBG_array, pred_horizon = 6):
    docX, docY, docIP = [], [], []

    for i in range(len(input) - n_prev - pred_horizon +1):
        docX.append(input[i : i + n_prev])
        docY.append(output[i + n_prev - 1 : i + n_prev + pred_horizon])
        docIP.append(isrealBG_array[i + n_prev - 1 : i + n_prev + pred_horizon])

        # docY.append(output[i + n_prev + pred_horizon])
        # docIP.append(isrealBG_array[i + n_prev + pred_horizon])

    alsX  = np.array(docX)
    alsY  = np.array(docY)
    alsY  = alsY[:,:,0]
    alsIP = np.array(docIP)

    return alsX, alsY, alsIP

def _load_prediction_samples(timesteps, input, output, isrealBG_array, pred_horizon = 6):
    docX, docY, docIP = [], [], []

    for i in range(timesteps, len(input)):
        docX.append(input[i - timesteps: i])
        docY.append(output[i - pred_horizon - 1 : i])
        docIP.append(isrealBG_array[i - pred_horizon - 1 : i])

        # docY.append(output[i + n_prev + pred_horizon])
        # docIP.append(isrealBG_array[i + n_prev + pred_horizon])

    alsX  = np.array(docX)
    alsY  = np.array(docY)
    alsY  = alsY[:,:,0]
    alsIP = np.array(docIP)

    return alsX, alsY, alsIP


def train_test_split(df, pred_window, test_size=0.5, task = 'estimation'):
    """
    This just split data to training and testing parts
    """
    ntrn = np.int(round(len(df) * (1 - test_size)))

    training = 1
    if(task == 'estimation'):
        X_train, y_train = load_estimation_samples(df.iloc[0:ntrn], n_prev, pred_window, out_dim, training)
    else:
        X_train, y_train = load_prediction_samples(df.iloc[0:ntrn], n_prev, pred_window, out_dim, training)
    X_test, y_test   = [], []

    if(test_size != 0):
        training = 0
        if(task == 'estimation'):
            X_test, y_test = load_estimation_samples(df.iloc[ntrn:], n_prev, pred_window, out_dim, training)
        else:
            X_test, y_test = load_prediction_samples(df.iloc[ntrn:], n_prev, pred_window, out_dim, training)

    return (X_train, y_train), (X_test, y_test)

def vital_subsample(timestep, data, threshold, duration_reset):
    Y_, X_ = [], []
    count = 0

    Y_.append(data[0])
    X_.append(timestep[0])
    for i in range(1, len(timestep)):
        if(abs(data[i-1] - data[i]) > threshold):
            count = 0
            Y_.append(data[i])
            X_.append(timestep[i])
        elif(count > duration_reset):
            count = 0
            Y_.append(data[i])
            X_.append(timestep[i])
        else:
            count += 1

    Y = np.array(Y_)
    X = np.array(X_)

    return X, Y


# def get_time_emb(df, dataset):
#     #get time vectors
#     pi = math.pi
#     time_vec = []
#     if(dataset == 'OhioT1DM'):
#         ToD = np.copy(df['ToD'].values)
#         ToD = ToD/3600
#         for i in ToD:
#             time_vec.append(math.cos((2*pi*i)/24))
#     elif(dataset == 'ARISES'):
#         ToD = np.copy(df['ToD'].values)
#         for i in ToD:
#             t = datetime.fromtimestamp(i)
#             t_inp = t.hour + (t.minute/60)
#             time_vec.append(math.cos((2*pi*t_inp)/24))
#     else:
#         print('Unknown Dataset')
#
#     time_vec = np.array(time_vec)
#
#     return time_vec

def getGPinputs(df, task_setting, scaler_list, enable_scaling = True):

    X_ = np.arange(len(df))/SAMPLES_IN_DAY
    # Y_ = np.copy(df[['CAL', 'ACC', 'GSR', 'ST']].values)
    Y_ = np.copy(df[['CAL', 'HR', 'GSR', 'ST']].values)
    # Y_ = np.copy(df[['CAL', 'RMSSD', 'SCR', 'Entropy']].values)

    scaler_list = []

    X0 = np.delete(X_, np.where(Y_[:,0]==0))
    Y0 = np.delete(Y_[:,0], np.where(Y_[:,0]==0))

    X1_ = np.delete(X_, np.where(Y_[:,1]==0))
    X2_ = np.delete(X_, np.where(Y_[:,2]==0))
    X3_ = np.delete(X_, np.where(Y_[:,3]==0))

    Y1_ = np.delete(Y_[:,1], np.where(Y_[:,1]==0))
    Y2_ = np.delete(Y_[:,2], np.where(Y_[:,2]==0))
    Y3_ = np.delete(Y_[:,3], np.where(Y_[:,3]==0))

    X1, Y1 = vital_subsample(X1_, Y1_, 30, 12)#10
    X2, Y2 = vital_subsample(X2_, Y2_, 0.5, 36)#0.5
    X3, Y3 = vital_subsample(X3_, Y3_, 2, 12)

    X0widx = np.c_[X0,np.ones(X0.shape[0])*0]
    X1widx = np.c_[X1,np.ones(X1.shape[0])*1]
    X2widx = np.c_[X2,np.ones(X2.shape[0])*2]
    X3widx = np.c_[X3,np.ones(X3.shape[0])*3]

    # plt.plot(X3_,Y3_)
    # plt.scatter(X3,Y3, c ="k", marker="x")
    # plt.show()

    if(enable_scaling):

        scalerBG  = MinMaxScaler(feature_range=(-1, 1))
        scalerHR  = MinMaxScaler(feature_range=(-1, 1))#0,1
        scalerGSR = MinMaxScaler(feature_range=(-1, 1))#0,1
        scalerST  = MinMaxScaler(feature_range=(-1, 1))#0,1

        # scalerBG  = StandardScaler()
        # scalerHR  = StandardScaler()#0,1
        # scalerGSR = StandardScaler()#0,1
        # scalerST  = StandardScaler()#0,1

        scalerBG.fit(np.atleast_2d(Y0).T)
        scalerHR.fit(np.atleast_2d(Y1).T)
        scalerGSR.fit(np.atleast_2d(Y2).T)
        scalerST.fit(np.atleast_2d(Y3).T)

        scaler_list = [scalerBG, scalerHR, scalerGSR, scalerST]

    Y0 = scaler_list[0].transform(np.atleast_2d(Y0).T)
    Y1 = scaler_list[1].transform(np.atleast_2d(Y1).T)
    Y2 = scaler_list[2].transform(np.atleast_2d(Y2).T)
    Y3 = scaler_list[3].transform(np.atleast_2d(Y3).T)

    print(len(Y1))
    print(len(Y0))
    # plt.plot(X1,Y1)
    # plt.plot(X0,Y0)
    # plt.scatter(X1,Y1, c ="k", marker="x")
    # plt.show()


    if(task_setting == 'STGP'):
        X = np.atleast_2d(X0).T
        Y = Y0
    else:
        # X = np.r_[X0widx,X1widx,X2widx,X3widx]
        # Y = np.r_[Y0,Y1,Y2,Y3]
        X = np.r_[X0widx,X1widx]
        Y = np.r_[Y0,Y1]

    return X, Y, scaler_list


def getdataFormats(df, dataset, scaler):

    #get time vectors
    pi = math.pi
    time_vec = []
    if(dataset == 'OhioT1DM'):
        ToD = np.copy(df['ToD'].values)
        ToD = ToD/3600
        for i in ToD:
            time_vec.append(math.cos((2*pi*i)/24))
    elif(dataset == 'ARISES'):
        ToD = np.copy(df['ToD'].values)
        for i in ToD:
            t = datetime.datetime.fromtimestamp(i)
            t_inp = t.hour + (t.minute/60)
            time_vec.append(math.cos((2*pi*t_inp)/24))
    else:
        print('Unknown Dataset')

    time_vec = np.array(time_vec)

    #get insulin
    bolus = df['I'].values
    basal = df['BAS'].values

    if(dataset == 'ARISES'):
        bolus = bolus/100
        basal = basal/100

    insulin  = bolus + basal
    insulin  = insulin/100

    #get meals
    carb = df['M'].values
    carb = carb/200
    #get
    exercise = df['exercise'].values
    exercise[exercise>0] = 1

    if(dataset == 'ARISES'):
        input  = np.c_[time_vec, carb, bolus, basal, exercise]
    else:
        input  = np.c_[time_vec, carb, insulin, exercise]

    output_ = df['G'].values
    output  = scaler.transform(np.atleast_2d(output_).T)
    # output  = np.atleast_2d(output_).T/120

    return input, output


def load_data_batch(posterior_dist, aux_input, output, ip_array, task = 'prediction', timestep = 24, horizon = 6):#24

    num_posterior_samples = posterior_dist.shape[1]


    loop_init = True
    for i in range(0, num_posterior_samples):

        posterior_sample = np.atleast_2d(posterior_dist[:,i]).T
        input = np.c_[posterior_sample, aux_input]

        delete_array = []

        if(task == 'estimation'):
            X_Train_, Y_Train_, sample_check = load_estimation_samples(timestep, input, output, ip_array)
            for i in range(len(Y_Train_)):
                if(np.count_nonzero(sample_check[i]) == 0):
                    delete_array.append(i)

        else:
            X_Train_, Y_Train_, sample_check = load_prediction_samples(timestep, input, output, ip_array, horizon)
            for i in range(len(Y_Train_)):
                if(np.count_nonzero(sample_check[i]) < 4):
                    delete_array.append(i)

        X_Train_ind = np.delete(X_Train_, np.array(delete_array), axis=0)
        Y_Train_ind = np.delete(Y_Train_, np.array(delete_array), axis=0)

        if(loop_init == True):
            loop_init = False
            X_Train = X_Train_ind
            Y_Train = Y_Train_ind
        else:
            X_Train = np.vstack((X_Train, X_Train_ind))
            Y_Train = np.vstack((Y_Train, Y_Train_ind))

    return X_Train, Y_Train

def _load_data_batch(posterior_dist, aux_input, output, ip_array, task = 'prediction', timestep = 24, horizon = 6):#24

    num_posterior_samples = posterior_dist.shape[1]
    print(num_posterior_samples)
    loop_init = True
    for i in range(0, num_posterior_samples):

        posterior_sample = np.atleast_2d(posterior_dist[:,i]).T
        # posterior_sample = np.atleast_2d(posterior_dist[:,i]).T/120
        input = np.c_[posterior_sample, aux_input]

        delete_array = []

        if(task == 'estimation'):
            X_, Y_, sample_check = _load_estimation_samples(timestep, input, output, ip_array)
            # X_Train_, Y_Train_, sample_check = load_estimation_samples(timestep, input, output, ip_array)
            # for i in range(len(Y_Train_)):
            #     if(np.count_nonzero(sample_check[i]) == 0):
            #         delete_array.append(i)

        else:
            X_, Y_, sample_check = _load_prediction_samples(timestep+horizon, input, output, ip_array, horizon)
            # for i in range(len(Y_Train_)):
            #     if(np.count_nonzero(sample_check[i]) < 4):
            #         delete_array.append(i)

        # X_Train_ind = np.delete(X_Train_, np.array(delete_array), axis=0)
        # Y_Train_ind = np.delete(Y_Train_, np.array(delete_array), axis=0)

        if(loop_init == True):
            loop_init = False
            X = X_
            Y = Y_

            active_entries = sample_check
        else:
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))

            active_entries = np.vstack((active_entries, sample_check))

    if(num_posterior_samples > 1):
        print(X.shape)
        print(Y.shape)
        m = X_.shape[0]
        X_final_, Y_final_, sample_check_ = [], [], []
        for j in range(0, X_.shape[0]):
            for k in range(0, posterior_dist.shape[1]):
                X_final_.append(X[m*k + j,:,:])
                Y_final_.append(Y[m*k + j])
                sample_check_.append(active_entries[m*k + j])

        X_final = np.array(X_final_)
        Y_final = np.array(Y_final_)
        sample_check = np.array(sample_check_)
    else:
        X_final  = X
        Y_final  = Y
        sample_check = active_entries

    # return X, Y, active_entries
    return X_final, Y_final, sample_check


def format_predictions(predictions, _target_scaler):
  """Reverts any normalisation to give predictions in original scale.

  Args:
    predictions: Dataframe of model predictions.

  Returns:
    Data frame of unnormalised predictions.
  """
  output = predictions.copy()

  column_names = predictions.columns
  # if col not in {'forecast_time', 'identifier'}:
  for col in column_names:
    output[col] = _target_scaler.inverse_transform(np.atleast_2d(predictions[col]).T)

  return output

def get_scores(output_map, y_test, ip_test, output_scale, timestep = 24, pred_horizon = 6):

    lag = timestep + 6

    pred_index = 't+{}'.format(pred_horizon)

    if(pred_horizon == 0):
        del_index  = np.where(ip_test[timestep: -pred_horizon] == 0)[0]
        y_test = y_test[timestep : -pred_horizon]
    else:
        del_index = np.where(ip_test[lag:] == 0)[0]
        y_test = y_test[lag: ]

    y_test[del_index] = np.nan

    if output_scale is None:
        lower_q = output_map["p2.5"]
        median_forecast = output_map["p50.0"]
        upper_q = output_map["p97.5"]
    else:
        lower_q = format_predictions(output_map["p2.5"], output_scale)
        median_forecast = format_predictions(output_map["p50.0"], output_scale)
        upper_q = format_predictions(output_map["p97.5"], output_scale)

    median_forecast.loc[del_index,pred_index] = np.nan
    lower_q.loc[del_index,pred_index] = np.nan
    upper_q.loc[del_index,pred_index] = np.nan


    plt.figure(pred_horizon)
    x____ = np.arange(len(median_forecast[pred_index]))
    plt.plot(median_forecast[pred_index])
    plt.plot(y_test, 'r')
    plt.fill_between(x____, lower_q[pred_index], upper_q[pred_index], alpha=0.2, color='b')

    y_pred = median_forecast[pred_index].dropna().values
    y_test = np.delete(y_test, del_index)

    RMSE = np.sqrt(np.mean((y_pred - y_test)**2))
    MAPE = np.mean(np.abs(100*(y_pred - y_test)/y_test))
    MAE  = np.mean(np.abs(y_pred - y_test))

    return RMSE, MAE, plt


################
