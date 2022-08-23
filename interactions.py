import json
import datetime
from random import Random
import random
from tokenize import Double
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import classification_report,confusion_matrix

num_features = 6


# Import necessary modules
from sklearn.model_selection import train_test_split

#reads watch json data and converts it into clean and usable data
def read_watch_data(bio_file, times, read_range = 10000) :
    rssi_current = 0

    bio_file.readline()

    #output list
    bio_test = []
    
    #data over given seconds
    rssi_avg_minute = [0] * 60
    rate_of_change = [0] * 3
    deviations = [0] * 5

    attribute_list = [0] * (num_features + 1)

    #calculated data
    roc = 0 #rate of change
    rssi_avg = 0
    stddev = 0
    
    #current time in rssi
    (time_60, time_70, time_80) = (0,0,0)

    st = ''
    fields = None

    interaction = 0

    #goes through file
    for i in range(0,read_range) :
        #reads data, fills dictionary with json value
        st = bio_file.readline()
        fields = json.loads(st[:st.find(',\n')])

        rssi_current = fields["peer_bt_rssi"] 

        #fills data arrays with current rssi
        rssi_avg_minute[i % 60] = rssi_current
        deviations[i % 5] = rssi_current
        rate_of_change[i % 3] = rssi_current

        #update times based on current rssi
        (time_60, time_70, time_80) = update_times(rssi_current, (time_60, time_70, time_80))

        #calculates average, rate of change, standard deviation
        rssi_avg = calculate_avg(rssi_avg_minute)
        
        roc = calculate_roc(rate_of_change)

        stddev = calculate_stddev(deviations, rssi_avg)

        #checks if interaction is happening in current time
        interaction = interaction_happened(times, i)
        
        #fills attribute list with fields
        fill_attributes(attribute_list, rssi_current, stddev, time_60, time_70, time_80, roc, interaction)

        #saves current attribute list
        bio_test.append(attribute_list.copy())
    
    return bio_test

#updates the times given the current rssi
def update_times(rssi_current, times) :
    time_60, time_70, time_80 = times
    if rssi_current > -60 :
        time_60 = time_60 + 1
        time_70 = time_70 + 1
    elif rssi_current > -70 :
        time_70 = time_70 + 1
        time_80 = time_80 + 1
    elif rssi_current > -80 :
        time_80 = time_80 + 1
        time_60 = time_60 - 0.5 if time_60 > 0 else 0
    else :
        if rssi_current > -85 :
            time_80 = time_80 - 1 if time_80 > 0 else 0
            time_70 = time_70 - 0.5 if time_70 > 0 else 0
            time_60 = 0
        else :               
            time_80 = 0
            time_70 = 0
    
    return (time_60, time_70, time_80)

#fills list with given attributes
def fill_attributes(attribute_list, rssi_current, stddev, time_60, time_70, time_80, roc, interaction) :
    attribute_list[0] = rssi_current
    attribute_list[1] = stddev
    attribute_list[2] = time_60
    attribute_list[3] = time_70
    attribute_list[4] = time_80
    attribute_list[5] = roc
    attribute_list[num_features] = interaction

#calculates rssi average over a minute
def calculate_avg(rssi_avg_minute) :
    rssi_avg = 0
    for j in range(0,60) :
        rssi_avg = rssi_avg + rssi_avg_minute[j]
    rssi_avg = rssi_avg / 60

    return rssi_avg

#calculates rate of change over 3 seconds
def calculate_roc(rate_of_change) :
    roc = 0

    for j in range(0,2) :
        roc = roc + (rate_of_change[j + 1] - rate_of_change[j])
    roc = roc / 2
        
    return roc

#calculates standard deviation over 5 seconds
def calculate_stddev(deviations, rssi_avg):
    stddev = 0
    for j in range(0, 5) :
        stddev = stddev + pow(deviations[j]-rssi_avg, 2)
       
    return pow(stddev, 0.5)

#detects if current time is in given interaction times
def interaction_happened(times, time) :
    for cur_time in times: 
        if cur_time[0] <= time <= cur_time[1] :
            return 1

    return 0 

#creates model
def train_data(bio_train) :
    arr = numpy.array(bio_train)
    X = arr[:,0:num_features]
    y = arr[:,num_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.003, random_state=40)

    mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), activation='logistic', solver='adam', max_iter=20000,batch_size=200,tol=1e-5)
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print(confusion_matrix(y_train,predict_train))
    print(classification_report(y_train,predict_train))

    print(mlp.score(X_test,y_test))

    return mlp

def test_model(model, test_arr) :
    time_since_interaction = 0
    can_prompt = False

    #goes through test data
    for i in range(0,int(len(test_arr) / 8)) :
        #gets 8 seconds of data and predicts if interaction occured
        X = test_arr[i*8:(i*8)+8,0:num_features]
        pred = model.predict(X)

        total = 0

        #goes through and sums interaction values
        for elem in pred:  
            total = total + elem

        #if any second had an interaction
        if total >= 1 :
            time_since_interaction = 0
            can_prompt = True
        else:
            time_since_interaction = time_since_interaction + 8

            #prompts user after 16 seconds
            if can_prompt and time_since_interaction >= 16 :
                print('interaction at ' + str(i*8))
                can_prompt = False

def main() :
    random.seed(None)

    #how far to read into the rssi data file to train model
    read_range = 8000

    path = "C:\\Users\\Aiden\\Downloads\\dyadTest03A\\dyadTest03A\\rssi_data.json"

    bio_file = open(path, "r")

    #times where interactions occured - seconds since the start
    times = [
                (13,47),
                (113,127),
                (145,159),
                (621,637),
                (1521,1597),
                (2830,2867),
                (2909,2919),
                (6127,6183),
                (6391,6417)
            ]
    
    #creates list of training and test data
    data = read_watch_data(bio_file, times, read_range)

    #creates neural network
    mlp = train_data(data)

    bio_file.close()
    #opens new file to test data
    path = "C:\\Users\\Aiden\\Downloads\\dyadTest02A-w\\dyadTest02A-w\\rssi_data.json"

    read_range = 20000
    new_times = [
        (10,32),
        (438,469),
        (532,554),
        (786,851),
        (984,1006),
        (1096,1121),
        (1390,1410)
    ]
    bio_file = open(path, "r")

    t = read_watch_data(bio_file, new_times, read_range)
    test_arr = numpy.array(t)

    test_model(mlp, test_arr)

main()