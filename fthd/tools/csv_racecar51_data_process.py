#!/usr/bin/env python3

import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import os

def write_dataset(horizon, save=True):
    global TOTAL_TIME
    dirname = os.path.dirname(__file__)
    log6 = os.path.join(dirname,'data/synced_data_log6.csv')
    log7 = os.path.join(dirname,'data/synced_data_log7.csv')
    log8 = os.path.join(dirname,'data/synced_data_log8.csv')
    
    savefile = os.path.join(dirname,'data/racecar51.csv')
    
    data6 = np.genfromtxt(log6, delimiter=',', skip_header=1)
    data7 = np.genfromtxt(log7, delimiter=',', skip_header=1)
    data8 = np.genfromtxt(log8, delimiter=',', skip_header=1)
    
    vx6 = data6[:,3] # vx
    vy6 = data6[:,4] # vy
    omega6 = data6[:,5] # omega
    steering6 = data6[:,6] # steering
    duty_cycle6 = data6[:,7] # duty_cycle
    len6 = len(vx6)
    
    vx7 = data7[:,3] # vx
    vy7 = data7[:,4] # vy
    omega7 = data7[:,5] # omega
    steering7 = data7[:,6] # steering
    duty_cycle7 = data7[:,7] # duty_cycle
    len7 = len(vx7)
    
    vx8 = data8[:,3] # vx
    vy8 = data8[:,4] # vy
    omega8 = data8[:,5] # omega
    steering8 = data8[:,6] # steering
    duty_cycle8 = data8[:,7] # duty_cycle
    len8 = len(vx8)

    print("len6 :",len6)
    print("len7 :",len7)
    print("len8 :",len8)
    
    total_len = len6+len7+len8
    
    times = np.zeros((total_len,1),dtype=np.double)
    
    for i in range(total_len):
        times[i,0] = TOTAL_TIME
        TOTAL_TIME += SAMPLING_TIME
    
    # plt.plot(times[0:len6],vx6)
    # plt.plot(times[0:len6],vy6)
    # plt.plot(times[0:len6],omega6)
    
    # plt.show()
    
    duty_cmd6 = duty_cycle6 - [0.0, *duty_cycle6[:-1]]
    steering_cmd6 = steering6 - [0.0, *steering6[:-1]]
    
    duty_cmd7 = duty_cycle7 - [0.0, *duty_cycle7[:-1]]
    steering_cmd7 = steering7 - [0.0, *steering7[:-1]]    
    
    duty_cmd8 = duty_cycle8 - [0.0, *duty_cycle8[:-1]]
    steering_cmd8 = steering8 - [0.0, *steering8[:-1]]
    
    features = np.zeros((total_len - 3*horizon,  horizon, 7), dtype=np.double)
    labels = np.zeros((total_len - 3*horizon, 3), dtype=np.double)
    time_features = np.zeros((total_len - 3*horizon,1),dtype=np.double)
    timelines = np.zeros((total_len - 3*horizon,1),dtype=np.double)
    
    for i in tqdm(range(len6 - horizon), desc="Compiling dataset 6"):

        features[i] = np.array([vx6[i:i+horizon], vy6[i:i+horizon], omega6[i:i+horizon], duty_cycle6[i:i+horizon], steering6[i:i+horizon], duty_cmd6[i:i+horizon], steering_cmd6[i:i+horizon]]).T

        labels[i] = np.array([vx6[i+horizon], vy6[i+horizon], omega6[i+horizon]]).T

        timelines[i] = times[i+horizon,-1]
        time_features[i] = times[i+horizon-1,-1]
        
    for i in tqdm(range(len6 - horizon, len6 + len7 - 2*horizon), desc="Compiling dataset 7"):

        j = i - (len6 - horizon)
        features[i] = np.array([vx7[j:j+horizon], vy7[j:j+horizon], omega7[j:j+horizon], duty_cycle7[j:j+horizon], steering7[j:j+horizon], duty_cmd7[j:j+horizon], steering_cmd7[j:j+horizon]]).T

        labels[i] = np.array([vx7[j+horizon], vy7[j+horizon], omega7[j+horizon]]).T

        timelines[i] = times[j+horizon,-1]
        time_features[i] = times[j+horizon-1,-1]
        
    for i in tqdm(range(len6 + len7 - 2*horizon, total_len - 3*horizon), desc="Compiling dataset 8"):

        j = i - (len6 + len7 - 2*horizon)
        features[i] = np.array([vx8[j:j+horizon], vy8[j:j+horizon], omega8[j:j+horizon], duty_cycle8[j:j+horizon], steering8[j:j+horizon], duty_cmd8[j:j+horizon], steering_cmd8[j:j+horizon]]).T

        labels[i] = np.array([vx8[j+horizon], vy8[j+horizon], omega8[j+horizon]]).T

        timelines[i] = times[j+horizon,-1]
        time_features[i] = times[j+horizon-1,-1]
    
    print(features[-1])
    print("last time:",times[-1])

    if save:
        np.savez(savefile[:savefile.find(".csv")] + "_" + str(horizon) + "RNN_val.npz", features=features, 
                 labels=labels,times_features=time_features, times = timelines)
    

if __name__ == "__main__":
    TOTAL_TIME = 0.00
    SAMPLING_TIME = 0.05
    # horizon_ = 4
    for k in range(1,21):
        write_dataset(k)