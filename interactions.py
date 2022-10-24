from cgi import test
import json
import random
import pandas as pd
from tokenize import Double
from turtle import pos
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn import cluster
num_features = 5
import matplotlib.pyplot as plt

# Import necessary modules
from sklearn.model_selection import train_test_split

def read_sensor_data(bio_file, times, read_range = 10000) :
    rssi_current = 0

    bio_file.readline()

    #output list
    bio_test = []
    
    #data over given seconds - fills with rssi value of -85
    rssi_avg_minute = [-85] * 20
    rate_of_change = [-85] * 3
    deviations = [-85] * 5

    attribute_list = [0] * (num_features + 1)

    below_85 = [0] * 15
    total_below_85 = 0

    #calculated data
    roc = 0 #rate of change
    rssi_avg = 0
    stddev = 0
    
    #current time in rssi
    (time_60, time_70, time_80) = (0,0,0)

    st = ''
    fields = None

    interaction = 0
    i = 0
    #goes through file
    while i < read_range:
        #reads data, fills dictionary with json value
        st = bio_file.readline()
        fields = json.loads(st[:st.find(',\n')])
        
        if fields['message_type'] != 'bluetooth_proximity' :
            continue
        sensor = fields["sensors"] 
        rssi_current = sensor['peer_bt_rssi']

        #fills data arrays with current rssi
        rssi_avg_minute[i % 20] = rssi_current
        deviations[i % 5] = rssi_current
        rate_of_change[i % 3] = rssi_current
        below_85[i % 10] = 1 if rssi_current < -85 else 0
        total_below_85 = below_85.count(1)

        #calculates average, rate of change, standard deviation
        rssi_avg = calculate_avg(rssi_avg_minute)

        #update times based on current rssi
        (time_60, time_70, time_80) = update_times(rssi_avg, (time_60, time_70, time_80))
        
        roc = calculate_roc(rate_of_change)

        stddev = calculate_stddev(deviations, rssi_avg)

        #checks if interaction is happening in current time
        interaction = interaction_happened(times, i)
        
        #fills attribute list with fields
        fill_attributes(attribute_list, rssi_current,rssi_avg, stddev, time_60, time_70, time_80, roc, interaction)

        #saves current attribute list
        bio_test.append(attribute_list.copy())
        i += 1
    return bio_test

#reads watch json data and converts it into clean and usable data
def read_watch_data(bio_file, times, read_range = 10000) :
    rssi_current = 0

    bio_file.readline()

    #output list
    bio_test = []
    
    #data over given seconds - fills with rssi value of -85
    rssi_avg_minute = [-85] * 20
    rate_of_change = [-85] * 3
    deviations = [-85] * 5

    attribute_list = [0] * (num_features + 1)

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
        rssi_avg_minute[i % 20] = rssi_current
        deviations[i % 5] = rssi_current
        rate_of_change[i % 3] = rssi_current

        #calculates average, rate of change, standard deviation
        rssi_avg = calculate_avg(rssi_avg_minute)

        #update times based on current rssi
        (time_60, time_70, time_80) = update_times(rssi_avg, (time_60, time_70, time_80))

        stddev = calculate_stddev(deviations, rssi_avg)

        #checks if interaction is happening in current time
        interaction = interaction_happened(times, i)
        
        #fills attribute list with fields
        fill_attributes(attribute_list,rssi_avg, stddev, time_60, time_70, time_80, interaction)

        #saves current attribute list
        bio_test.append(attribute_list.copy())
    
    return bio_test

#updates the times given the current rssi
def update_times(rssi_current, times) :
    time_60, time_70, time_80 = times
    if rssi_current >= -60 :
        time_60 = time_60 + 1
        time_70 = time_70 + 1
    elif rssi_current >= -70 :
        time_70 = time_70 + 1
        time_80 = time_80 + 1
    elif rssi_current >= -80 :
        time_80 = time_80 + 1
        time_60 = time_60 - 1 if time_60 > 0 else 0
    else :
        if rssi_current >= -85 :
            time_80 = time_80 - 1 if time_80 > 0 else 0
            time_70 = time_70 - 0.5 if time_70 > 0 else 0
            time_60 = 0
        else :            
            time_60 = 0   
            time_80 = 0
            time_70 = 0
    
    return (time_60, time_70, time_80)

#fills list with given attributes
def fill_attributes(attribute_list,rssi_avg, stddev, time_60, time_70, time_80, interaction) :
    attribute_list[0] = rssi_avg
    attribute_list[1] = stddev
    attribute_list[2] = time_60
    attribute_list[3] = time_70
    attribute_list[4] = time_80

    attribute_list[num_features] = interaction

#calculates rssi average over a minute
def calculate_avg(rssi_avg_minute) :
    rssi_avg = 0
    for j in range(0,20) :
        rssi_avg = rssi_avg + rssi_avg_minute[j]
    rssi_avg = rssi_avg / 20

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
    arr = np.array(bio_train)

    sm = SMOTE(sampling_strategy='minority')
    
    X = arr[:,0:num_features]
    y = arr[:,num_features]
    X, y = sm.fit_resample(X,y)

    clf = ensemble.AdaBoostClassifier()
    parameters = {
              'n_estimators':[10,25,50,100, 200],
              'learning_rate':[0.0001, 0.005, 0.01,0.1]
              }
    
    grid = GridSearchCV(clf, parameters, refit = True, verbose = 3,n_jobs=-1,scoring="accuracy") 
    grid.fit(X, y)
    
    return grid

def test_accuracy(model, X, y) :
    predictions = model.predict(X)
    correct_outputs = y
    tn, fp, fn, tp = metrics.confusion_matrix(correct_outputs, predictions).ravel()
    print("Accuracy = %f" % (metrics.accuracy_score(correct_outputs, predictions)))
    print("TN = %d FP = %d FN = %d TP = %d" % (tn, fp, fn, tp))
    print(metrics.classification_report(correct_outputs, predictions, target_names = ["No interaction", "Interaction"]))

def form_initial_model(path, times, read_range) :
    bio_file = open(path, "r")
    
    #creates list of training and test data
    data = read_sensor_data(bio_file, times, read_range)

    #creates neural network
    model = train_data(data)

    bio_file.close()

    pickle.dump(model, open('model.sav', 'wb'))

def form_mlp(data) :
    X = np.delete(data, num_features, 1)
    y = data[:,num_features]

    mlp = MLPClassifier(hidden_layer_sizes=(16,8,4),activation='tanh',batch_size=64,solver='adam', max_iter=30000,tol=1e-8)
    
    mlp.fit(X,y)
    pickle.dump(mlp, open('mlpmodel.sav', 'wb'))

def read_data(path, times, read_range, model) :
    bio_file = open(path, "r")
    data = read_sensor_data(bio_file, times, read_range)
    bio_file.close()

    arr = np.array(data)
    preds = model.predict(arr[:,0:num_features])
    newarr = []

    for i in range(0, len(preds)) :
        out = preds[i]
        newarr.append([out])

    arr = np.append(arr, newarr, axis=1)

    return arr

def form_models(path, read_range, times, already_trained) :
    success = True
    base_model = None
    mlp = None

    try :
        base_model = pickle.load(open('model.sav', 'rb'))
    except :
        success = False

    if not already_trained or not success:
        form_initial_model(path, times, read_range)
        base_model = pickle.load(open('model.sav', 'rb'))

    data = read_data(path, times, read_range, base_model)
    data = np.array(data)
    success = True

    try :
        mlp = pickle.load(open('mlpmodel.sav', 'rb'))
    except :
        success = False

    if not already_trained or not success:
        form_mlp(data)
        mlp = pickle.load(open('mlpmodel.sav', 'rb'))
    
    return base_model, mlp

def main() :
    random.seed(None)
    
    already_trained = True

    times = [
        (150,515),
        (1285,1575),
        (1987,2010),
        (2315,2385),
        (2930,3210),
        (3500,3905),
        (4340,4600),
        (5090,5287)
    ]

    path = "C:\\Users\\Aiden\\Downloads\\sensordata\\sensor_data.json"
    read_range = 3000
    base_model, mlp = form_models(path, read_range, times, already_trained)
    
    read_range = 6000
    test_arr = read_data(path, times, read_range, base_model)
    X = np.delete(test_arr, num_features, 1)
    y = test_arr[:,num_features]

    test_accuracy(mlp, X[3000:6000], y[3000:6000])
    plot_model(X,y, mlp)
    plot_data(test_arr, 0, 6000)

def plot_data(data, lower, upper) :
    y = [data[i][0] for i in range(lower,upper)]
    baseline = [-80 for i in range(lower,upper)]

    x = [i for i in range(lower,upper)]
    fig, ax = plt.subplots()
    ax.plot(x, y, "red")
    ax.plot(x, baseline, "b--")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Bluetooth RSSI distance")
    plt.title("Distance of 2 watch users")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 120))
    plt.show()

def plot_model(X,y, model) :
    x = [i for i in range(len(y))]
    predictions = model.predict(X) 
    correct_outputs = [y[i] for i in range (len(y))]

    fig, ax = plt.subplots()
    ax.plot(x, predictions, "r--")
    ax.plot(x, correct_outputs, "b")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Interaction occured")
    plt.title("ML Model vs Actual Data")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 120))
    plt.show()

main()