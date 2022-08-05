import json
import functools
import time
import datetime
import sklearn
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import classification_report,confusion_matrix


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

bio_file = open("C:\\Users\\Aiden\\Downloads\\dyadTest03A\\dyadTest03A\\sensor_data.json", "r")
bt_file = open("C:\\Users\\Aiden\\Downloads\\dyadTest03A\\dyadTest03A\\rssi_data.json", "r")

rssi_avg = 0
rssi_list = json.load(bt_file)

bio_file.readline()
bio_test = []
bio_interactions = []

time_start = 0
time_last = 0
cur_time = ''

interaction_possible = False
interaction_confirmed = False
m = []
index = 0
n = 0
for elem in rssi_list : 
    rssi_avg = rssi_avg + elem["peer_bt_rssi"]
    index = index + 1
    if index == 5:
        rssi_avg = rssi_avg / 5
        
        cur_time = elem["stamp"]
        cur_time = cur_time[:cur_time.find("+")]

        cur_time = datetime.datetime.strptime(cur_time, '%Y-%m-%dT%H:%M:%S.%f').timestamp()

        if rssi_avg < -80 and not interaction_possible: #if new rssi in range
            time_start = cur_time
            time_last = cur_time
            interaction_possible = True
        elif rssi_avg < -80 and interaction_possible :
            time_last = cur_time
        elif interaction_possible and cur_time - time_last > 30:
            interaction_possible = False
            interaction_confirmed = False

        if interaction_possible and cur_time - time_start > 300 :
            interaction_confirmed = True
                

        rssi_avg = 0
        index = 0
    motion = []
    acceleration = []
    for i in range(0,6) : #13 to read whole thing
        
        st = bio_file.readline()

        j = json.loads(st[:st.find(',\n')])

        if j["message_type"] != 'device_motion' : 
            continue

        j.pop('stamp')
        l = []
        for key, value in j["sensors"].items() :
            if 'mag' not in key and 'gravity' not in key and key != 'heading':
                l.append(value)

        
        if interaction_confirmed : 
            l.append(1)
            
        else :
            l.append(0)

        if n < 20000 : 
            bio_interactions.append(l)
        else :
            bio_test.append(l)
    n = n + 1

    if n > 30000 : break


arr = numpy.array(bio_interactions)
test_arr = numpy.array(bio_test)
X = arr[:,0:9]
y = arr[:,9]

testX = test_arr[:,0:9]
testy = test_arr[:,9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=40)

mlp = MLPClassifier(hidden_layer_sizes=(50,25,10), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

print(mlp.score(testX,testy))


print(10)
#classify data into activities, train on that
