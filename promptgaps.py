from datetime import datetime
#search for:
#proximity ended
#timestart - timeend -> save day as last day to reset when day changes
#duration = x

#check duration > 300
#check: timeend in trigger time
#check time after timeout

#if true, get next line and verify prompted, get timeout

class hourtime :
    def __init__(self, hour, minute) -> None:
        self.hour = hour
        self.minute = minute

def create_interval() :
    arr = []
    d1 = hourtime(hour=8,minute=0)
    d2 = hourtime(hour=10,minute=15)
    arr.append([d1,d2, 0])

    d3 = hourtime(hour=10,minute=30)
    d4 = hourtime(hour=12,minute=45)
    arr.append([d3,d4, 0])

    d5 = hourtime(hour=13,minute=0)
    d6 = hourtime(hour=15,minute=15)
    arr.append([d5,d6, 0])

    d7 = hourtime(hour=15,minute=30)
    d8 = hourtime(hour=17,minute=45)
    arr.append([d7,d8, 0])

    d9 = hourtime(hour=18,minute=0)
    d10 = hourtime(hour=20,minute=15)
    arr.append([d9,d10, 0])

    return arr

intervals = create_interval()

def read_data(path) :
    file = open(path, 'r')

    st = ''

    st = file.readline()
    current_day = None
    timeout = None
    schedule_time = None
    i = 1

    while st != '' :
        if 'Proximity interaction ended:' in st:
            time_range = file.readline()
            duration = file.readline()
            duration = float(duration[duration.index('=') + 2:duration.index('s')])
            i += 2
            if duration > 300 :
                time = time_range[time_range.index('- ') + 2:-1]
                dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

                if current_day == None:
                    current_day = dt
                elif current_day.day != dt.day :
                    global intervals
                    intervals = create_interval()
                    current_day = dt
                    timeout = None
                
                if in_intervals(dt) :
                    if timeout == None or (dt.hour > timeout.hour or (dt.hour >= timeout.hour and dt.minute >= timeout.minute)):
                        st = file.readline()
                        i += 1

                        if 'scheduled prompt' not in st or schedule_time != None:
                            print("ERROR 1 at line " + str(i))
                            reset_interval(dt)
                            
                        else :
                            st = st[:st.index('|') - 1]
                            schedule_time = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")

        elif 'prompting user' in st :
            if schedule_time == None :
                print("ERROR 2 at line " + str(i))
            else :
                cur_time = st[:st.index('|') - 1]
                cur_time = datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S")

                if (not (58 < (cur_time - schedule_time).total_seconds() < 62)) :
                    print("ERROR 3 at line " + str(i))
            timeout = st[st.index('time:')+6:-1]
            timeout = datetime.strptime(timeout, "%Y-%m-%d %H:%M:%S")
            schedule_time = None
            
        st = file.readline()
        i +=1
    if schedule_time == None :
        print("ERROR 4")
    file.close()

def in_intervals(time: datetime) :
    for current_interval in intervals :
        if time.hour == current_interval[0].hour :
            if time.minute >= current_interval[0].minute :
                if current_interval[2] == 0 :
                    current_interval[2] = 1
                    return True
                else :
                    return False
        elif time.hour == current_interval[1].hour :
            if time.minute <= current_interval[1].minute :
                if current_interval[2] == 0 :
                    current_interval[2] = 1
                    return True
                else :
                    return False
        elif time.hour > current_interval[0].hour and time.hour < current_interval[1].hour :
            if current_interval[2] == 0 :
                current_interval[2] = 1
                return True
            else :
                return False
    return False

def reset_interval(time : datetime) :
    for current_interval in intervals :
        if time.hour == current_interval[0].hour :
            if time.minute >= current_interval[0].minute :
                current_interval[2] = 0
        elif time.hour == current_interval[1].hour :
            if time.minute <= current_interval[1].minute :
                current_interval[2] = 0
        elif time.hour > current_interval[0].hour and time.hour < current_interval[1].hour :
            current_interval[2] = 0
def main() :
    path = "C:\\Users\\Aiden\\Downloads\\H01\\H01\\dyadH02A2w.system_logs.log"

    read_data(path)

if __name__ == '__main__' :
    main()