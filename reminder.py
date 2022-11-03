import json

import numpy as np
import math

import matplotlib.pyplot as plt

num_features = 2

#reads sensor data from sensor.json
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

    #reads through 
    while current_second < seconds:
        st = file.readline()

        #if reach end of json
        if ']' in st :
            break

        fields = json.loads(st[:st.find(',\n')])
        
        #makes sure current line is watch device motion
        if fields['message_type'] != 'device_motion' :
            continue

        sensor = fields["sensors"] 
        
        #gets current rotation
        xr += sensor['rotation_rate_x']
        yr += sensor['rotation_rate_y']
        zr += sensor['rotation_rate_z']

        #gets current rotation
        xa += sensor['user_acceleration_x']
        ya += sensor['user_acceleration_y']
        za += sensor['user_acceleration_z']
        
        #sensor runs at 10 hz so save the data every second
        if (i != 0 and i % 10 == 0) :
            #gets size of vectors
            rotation = math.pow(xr**2 + yr**2 + zr**2, 1/2)
            acceleration = math.pow(xa**2 + ya**2 + za**2, 1/2)

            #appends last second of data
            data.append(fill_features(rotation, acceleration))

            #resets vectors
            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)

            current_second += 1

        i+=1

    return np.array(data)

#creates list of rotation and acceleration
def fill_features(rotation, acceleration) -> list:
    attribute_list = [None] * (num_features)

    attribute_list[0] = rotation
    attribute_list[1] = acceleration

    return attribute_list

#plots acceleration + rotation data
def plot_data(data, lower, upper) :
    #gets acceleration and rotation
    y1 = [data[i][1] for i in range(lower,upper)]
    y2 = [data[i][0] for i in range(lower,upper)]
    x = [i for i in range(lower,upper)]

    fig, ax = plt.subplots()

    #plots data
    ax.plot(x, y1, "red")
    ax.plot(x, y2, "b--")

    #labels
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration/Rotation")
    plt.title("Acceleration/Rotation over time")

    #x ticks
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 120))

    #shows data
    plt.show()

#alerts the user if total seconds not moving above threshold
def alert_user(data) -> bool:
    total = 0

    #combines acceleration + rotation
    for i in range(len(data)) :
        if data[i][0] + data[i][1] < 1 :
            total += 1
    
    print("total: " + str(total))

    #needs less than 3 minutes of movement in the hour to notify
    if total > 3420 :
        print("YOU NEED TO WEAR THE WATCH")
        return True
    return False

def run_program() :
    path = "C:\\Users\\Aiden\\Downloads\\sensordata\\sensor_data.json"
    read_range = 6000 # in seconds

    data = read_sensor_data(path, read_range)

    alert_user(data[0:3600,:])
    plot_data(data, 0,6000)

if __name__ == '__main__' :
    run_program()