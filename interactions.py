from cgi import test
from code import interact
import json
import datetime
from logging import root
from random import Random
import random
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import classification_report,confusion_matrix


# Import necessary modules
from sklearn.model_selection import train_test_split
from math import sqrt

t = []

def read_watch_data(bio_file, read_range = 10000) :
    rssi_avg = -100
    rssi_total = 0
    rssi_current = 0
    data_split = read_range / 2

    bio_file.readline()
    bio_test = []
    bio_interactions = []

    time_80 = 0
    time_70 = 0
    time_60 = 0

    time_start = 0
    time_last = 0
    cur_time = 0

    interaction_possible = False
    interaction_confirmed = False
    index = 0
    x = 0
    y = 0
    z = 0
    for i in range(0,read_range) :
        st = bio_file.readline()
        fields = json.loads(st[:st.find(',\n')])

        if fields['message_type'] == 'bluetooth_proximity' :
            rssi_current = fields['sensors']["peer_bt_rssi"]
            rssi_total = rssi_total + rssi_current
            index = index + 1
            rssi_avg = rssi_total / index
            if index % 5 == 0 :
                cur_time = fields["stamp"]
                cur_time = cur_time[:cur_time.find("+")]
                cur_time = datetime.datetime.strptime(cur_time, '%Y-%m-%dT%H:%M:%S.%f').timestamp()
                
                if rssi_avg < -90 : 
                    interaction_possible = False
                    interaction_confirmed = False
                    time_60 = 0
                    time_70 = 0
                    time_80 = 0
                if rssi_avg > -80 :
                    if not interaction_possible:
                        time_start = cur_time
                        time_last = cur_time
                        interaction_possible = True
                    elif interaction_possible :
                        time_last = cur_time

                    if rssi_avg > -45 or (interaction_possible and cur_time - time_start > 60) :
                        interaction_confirmed = True
                elif interaction_possible and cur_time - time_last > 10:
                    interaction_possible = False
                    interaction_confirmed = False
                    time_60 = 0
                    time_70 = 0
                    time_80 = 0

                if rssi_avg > -60 :
                    time_60 = time_60 + 5
                elif rssi_avg > -70 :
                    time_70 = time_70 + 2
                elif rssi_avg > -80 :
                    time_80 = time_80 + 1


                index = 0
                rssi_total = 0

            attribute_list = []
            attribute_list.append(rssi_avg)
            attribute_list.append(time_60)
            attribute_list.append(time_70)
            attribute_list.append(time_80)

            if interaction_confirmed : 
                attribute_list.append(1)
            else:
                attribute_list.append(0)

            if i > 1000000:    
                t.append(attribute_list)
                continue
            if i < data_split : 
                bio_interactions.append(attribute_list)
            else :
                bio_test.append(attribute_list)
        

    return (bio_interactions, bio_test)

def train_data(bio_train, bio_test) :
    arr = numpy.array(bio_train)
    test_arr = numpy.array(bio_test)
    X = arr[:,0:4]
    y = arr[:,4]

    testX = test_arr[:,0:4]
    testy = test_arr[:,4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=40)

    mlp = MLPClassifier(hidden_layer_sizes=(12,8,4), activation='logistic', solver='adam', max_iter=5000)
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)
    print(confusion_matrix(y_train,predict_train))
    print(classification_report(y_train,predict_train))

    print(mlp.score(testX,testy))

    return mlp

def test_model(model) :
    test_arr = numpy.array(t)
    preds = []

    time_since_interaction = 0
    can_prompt = False
    saved = [None,None,None,None]
    for i in range(0,int(len(t) / 30)) :
        if i > 5:
            saved[0] =  test_arr[(i-1)*30:((i-1)*30)+30,0:4]
            saved[1] =  test_arr[(i-2)*30:((i-2)*30)+30,0:4]
            saved[2] =  test_arr[(i-3)*30:((i-3)*30)+30,0:4]
            saved[3] =  test_arr[(i-4)*30:((i-4)*30)+30,0:4]
        X = test_arr[i*30:(i*30)+30,0:4]
        pred = model.predict(X)
        length = len(pred)
        total = 0
        for elem in pred:  
            total = total + elem
        avg = total / length
        if avg > 0.3:
            time_since_interaction = 0
            can_prompt = True
        else:
            time_since_interaction = time_since_interaction + 30
            if can_prompt and time_since_interaction >= 60 :
                print('did interaction happen?')
                can_prompt = False

        preds.append(pred)

    
    return preds

def main() :
    random.seed(None)

    read_range = 1500000



    bio_file = open("C:\\Users\\Aiden\\Downloads\\dyadTest03A\\dyadTest03A\\sensor_data.json", "r")

    bi, bt = read_watch_data(bio_file, read_range)

    mlp = train_data(bi, bt)

    test_arr = numpy.array(t)
    tx = test_arr[:,0:4]
    ty = test_arr[:,4]

    print(mlp.score(tx,ty))

    test_model(mlp)

main()