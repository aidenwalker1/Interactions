import json
import random

import numpy as np
import pickle
import math

from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

num_features = 2

#get battery data
#get movement data

def read_sensor_data(filename, seconds) -> np.array:
    file = open(filename, 'r')

    file.readline()

    i = 0
    st = ''
    fields = []

    data = []

    (xr, yr, zr) = (0,0,0)
    (xa, ya, za) = (0,0,0)

    current_second = 0

    while current_second < seconds :
        st = file.readline()
        fields = json.loads(st[:st.find(',\n')])
        
        if fields['message_type'] != 'device_motion' :
            continue
        sensor = fields["sensors"] 
        xr += sensor['rotation_rate_x']
        yr += sensor['rotation_rate_y']
        zr += sensor['rotation_rate_z']

        xa += sensor['user_acceleration_x']
        ya += sensor['user_acceleration_y']
        za += sensor['user_acceleration_z']
        
        if (i % 10 == 0) :
            rotation = math.pow(xr**2 + yr**2 + zr**2, 1/2)
            acceleration = math.pow(xa**2 + ya**2 + za**2, 1/2)

            data.append(fill_features(rotation, acceleration))

            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)

            current_second += 1

        i+=1

    return np.array(data)
        
def fill_features(rotation, acceleration) -> list:
    attribute_list = [None] * (num_features)

    attribute_list[0] = rotation
    attribute_list[1] = acceleration

    return attribute_list


def plot_data(data, lower, upper) :
    y = [data[i][1] for i in range(lower,upper)]
    y2 = [data[i][0] for i in range(lower,upper)]
    x = [i for i in range(lower,upper)]
    fig, ax = plt.subplots()
    ax.plot(x, y, "red")
    ax.plot(x, y2, "b--")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration/Rotation")
    plt.title("Acceleration/Rotation over time")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 120))
    plt.show()

def alert_user(data) -> bool:

    total = 0
    for i in range(len(data)) :
        if data[i][0] + data[i][1] < 1 :
            total += 1
    
    print("total: " + str(total))
    if total > 3420 :
        print("YOU NEED TO WEAR THE WATCH")
        return True
    return False

def main() :
     # file read data
    path = "C:\\Users\\Aiden\\Downloads\\sensordata\\sensor_data.json"
    read_range = 6000

    data = read_sensor_data(path, read_range)

    alert_user(data[0:3600,:])
    plot_data(data, 0,6000)

main()