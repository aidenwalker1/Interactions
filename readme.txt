How to use:

1. Go to main at bottom
2. Change test_path to file where data is on your pc
3. Change test_read_range to length to read from file
4. Change time to intervals where interactions occured in the data - if want to test model accuracy over that range
5. Run and verify 

The test_model function goes over data from test file, and console prints where it thinks interactions occured (16 seconds after last detection).
The number printed corresponds to the line in the rssi json file where an interaction occured previously. 
