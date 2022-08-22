import json
import datetime
from random import Random
import random
from tokenize import Double
import numpy
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix


# Import necessary modules
from sklearn.model_selection import train_test_split

def read_watch_data(bio_file, times, read_range = 10000) :
    rssi_current = 0

    bio_file.readline()

    bio_test = []
    
    rssi_avg_minute = [None] * 60
    rate_of_change = [None] * 3
    deviations = [None] * 5

    attribute_list = [None] * 7

    roc = 0 #rate of change
    rssi_avg = 0
    stddev = 0

    rate_index = 0
    rssi_index = 0
    dev_index = 0
    
    time_80 = 0
    time_70 = 0
    time_60 = 0

    dev_full = False
    avg_full = False
    rate_full = False

    st = ''
    fields = None

    interaction = 0

    for i in range(0,read_range) :
        #reads data, fills dictionary with json value
        st = bio_file.readline()
        fields = json.loads(st[:st.find(',\n')])

        rssi_current = fields["peer_bt_rssi"] 

        #fills data arrays with current rssi
        rssi_avg_minute[rssi_index] = rssi_current
        deviations[dev_index] = rssi_current
        rate_of_change[rate_index] = rssi_current

        #updates indices for rssi, deviations, roc
        rate_index = rate_index + 1
        rssi_index = rssi_index + 1
        dev_index = dev_index + 1

        #update times based on current rssi
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

        #calculates average, rate of change, std deviation
        rssi_avg = calculate_avg(avg_full, rssi_avg_minute, rssi_index)
        
        roc = calculate_roc(rate_full, rate_of_change, rate_index)

        stddev = calculate_stddev(dev_full, deviations, dev_index, rssi_avg)

        #checks if interaction is happening in current time
        interaction = interaction_happened(times, i)
        
        #checks if array indices out of bounds
        if dev_index > 4 :
            dev_full = True
            dev_index = 0

        if rssi_index > 59 :
            avg_full = True
            rssi_index = 0

        if rate_index > 2 :
            rate_full = True
            rate_index = 0
        
        #fills attribute list with fields
        fill_attributes(attribute_list, rssi_current, stddev, time_60, time_70, time_80, roc, interaction)

        #saves current attribute list
        bio_test.append(attribute_list.copy())
    
    return bio_test

#fills list with given attributes
def fill_attributes(attribute_list, rssi_current, stddev, time_60, time_70, time_80, roc, interaction) :
    attribute_list[0] = rssi_current
    attribute_list[1] = stddev
    attribute_list[2] = time_60
    attribute_list[3] = time_70
    attribute_list[4] = time_80
    attribute_list[5] = roc
    attribute_list[6] = interaction

#calculates rssi average over a minute
def calculate_avg(avg_full, rssi_avg_minute, rssi_index) :
    rssi_avg = 0
    if not avg_full :
        for j in range(0,rssi_index) :
            rssi_avg = rssi_avg + rssi_avg_minute[j]
        rssi_avg = rssi_avg / rssi_index
    else :
        for j in range(0,60) :
            rssi_avg = rssi_avg + rssi_avg_minute[j]
        rssi_avg = rssi_avg / 60
    return rssi_avg

def calculate_roc(rate_full, rate_of_change, rate_index) :
    roc = 0

    if not rate_full :
        for j in range(0,rate_index - 1) :
            roc = roc + (rate_of_change[j + 1] - rate_of_change[j])
        roc = roc / rate_index
    else :
        for j in range(0,2) :
            roc = roc + (rate_of_change[j + 1] - rate_of_change[j])
        roc = roc / 3
    return roc

def calculate_stddev(dev_full, deviations, dev_index, rssi_avg):
    stddev = 0
    if not dev_full:
        for j in range(0, dev_index) :
            stddev = stddev + pow(deviations[j]-rssi_avg, 2)
    else :
        for j in range(0, 5) :
            stddev = stddev + pow(deviations[j]-rssi_avg, 2)
    return pow(stddev, 0.5)

#detects if current time is in given interaction times
#params: times: list of tuples of ints
#        time: int
def interaction_happened(times, time) :
    for cur_time in times: 
        if cur_time[0] <= time <= cur_time[1] :
            return 1

    return 0 

def train_data(bio_train) :
    arr = numpy.array(bio_train)
    X = arr[:,0:6]
    y = arr[:,6]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.003, random_state=40)

    mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), activation='logistic', solver='adam', max_iter=10000,batch_size=200)
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
    for i in range(0,int(len(test_arr) / 8)) :
        X = test_arr[i*8:(i*8)+8,0:6]
        pred = model.predict(X)
        total = 0

        for elem in pred:  
            total = total + elem

        if total >= 1 :
            time_since_interaction = 0
            can_prompt = True
        else:
            time_since_interaction = time_since_interaction + 5
            if can_prompt and time_since_interaction >= 20 :
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