from datetime import datetime
import json

#time class, takes in hours and minute to compare interval
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

def read_data(path, prompts) :
    file = open(path, 'r')

    st = ''

    st = file.readline()
    current_day = None
    timeout = None
    schedule_time = None
    interaction_confirmed = False
    random_time = None
    prompt_group = None
    i = 1
    global intervals

    while st != '' :
        if 'Proximity interaction ended:' in st:
            time_range = file.readline()
            duration = file.readline()
            duration = float(duration[duration.index('=') + 2:duration.index('s')])
            i += 2
            if duration > 330 or interaction_confirmed :
                interaction_confirmed = False
                time = time_range[time_range.index('- ') + 2:-1]
                dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

                if current_day == None:
                    current_day = dt
                elif current_day.day != dt.day :
                    intervals = create_interval()
                    current_day = dt
                    timeout = None
                
                if in_intervals(dt) :
                    if timeout == None or (dt.hour > timeout.hour or (dt.hour >= timeout.hour and dt.minute >= timeout.minute)):
                        st = file.readline()
                        i += 1

                        if 'scheduled prompt' not in st or schedule_time != None:
                            print("ERROR 1: prompt should have occured at line " + str(i))
                            reset_interval(dt, 0)
                            
                        else :
                            st = st[:st.index('|') - 1]
                            schedule_time = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")

        elif 'prompting user' in st :
            if schedule_time == None :    
                t = st[:st.index('|') - 1]
                t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

                if current_day == None:
                    current_day = t
                elif current_day.day != t.day :
                    intervals = create_interval()
                    current_day = t
                
                if in_intervals(t) :
                    if did_respond(prompt_group, prompts):
                        reset_interval(t, 1)
                    else :
                        reset_interval(t, 0)
                else :
                    if did_respond(prompt_group, prompts):
                        if search_interval(t) == 1 :
                            print("ERROR 2: prompt has already been shown in interval at line " + str(i))
            else :
                cur_time = st[:st.index('|') - 1]
                cur_time = datetime.strptime(cur_time, "%Y-%m-%d %H:%M:%S")

                if not (58 < (cur_time - schedule_time).total_seconds() < 62) :
                    if random_time != None :
                        if not (-2 < (cur_time - random_time).total_seconds() < 2) :
                            print("ERROR 3: prompt not shown at right time at line " + str(i))
                        random_time = None
                    else :
                        print("ERROR 3: prompt not shown at right time at line " + str(i))
            timeout = st[st.index('time:')+6:-1]
            timeout = datetime.strptime(timeout, "%Y-%m-%d %H:%M:%S")
            schedule_time = None
        elif 'proximity interaction start' in st :
            interaction_confirmed = True
        elif ' Scheduled random' in st:
            random_time = st[st.index('for ') + 4:-1]
            random_time = datetime.strptime(random_time, "%Y-%m-%d %H:%M:%S")
        elif 'Showing Prompt Group' in st:
            prompt_group = st[st.index('Group') + 6:-1]
        st = file.readline()
        i +=1

    if schedule_time != None :
        print("ERROR 4: prompt should have been shown but data collection ended")
    file.close()

def search_interval(time: datetime) :
    for current_interval in intervals :
        if time.hour == current_interval[0].hour :
            if time.minute >= current_interval[0].minute :
                return current_interval[2]
        elif time.hour == current_interval[1].hour :
            if time.minute <= current_interval[1].minute :
                return current_interval[2]
        elif time.hour > current_interval[0].hour and time.hour < current_interval[1].hour :
            return current_interval[2]
    return -1

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

def read_prompts(path) :
    f = open(path, 'r')
    data = json.load(f)
    f.close()
    return data

def did_respond(id, data) :
    for prompt in data :
        if prompt['identifier'] == id :
            return prompt['prompts'][0]['chosen_response'] != None
    return False

def reset_interval(time : datetime, val) :
    for current_interval in intervals :
        if time.hour == current_interval[0].hour :
            if time.minute >= current_interval[0].minute :
                current_interval[2] = val
        elif time.hour == current_interval[1].hour :
            if time.minute <= current_interval[1].minute :
                current_interval[2] = val
        elif time.hour > current_interval[0].hour and time.hour < current_interval[1].hour :
            current_interval[2] = val
def main() :
    path = "C:\\Users\\Aiden\\Downloads\\H01\\H01\\dyadH04A2w.system_logs.log"
    prompt_path = "C:\\Users\\Aiden\\Downloads\\H01\\H01\\dyadH04A2w.prompt_groups.json"
    prompts = read_prompts(prompt_path)
    read_data(path, prompts)

if __name__ == '__main__' :
    main()