import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def gen_synthetic(n=500):
    rows=[]
    train_types = ['Express','Local','Freight']
    weathers = ['Clear','Rain','Fog']

    for i in range(n):
        train_id = 1000 + i
        t_type = np.random.choice(train_types, p=[0.4,0.3,0.3])
        sched_time = datetime(2025,9,1,6,0) + timedelta(minutes=5*i)  # spread trains
        tod = sched_time.hour + sched_time.minute/60
        dow = sched_time.weekday()
        weather = np.random.choice(weathers, p=[0.8,0.15,0.05])
        congestion = np.clip(np.random.normal(0.5,0.2),0,1)

        # base delay factors
        base = {'Express':1, 'Local':2, 'Freight':4}[t_type]
        w_extra = 2 if weather!='Clear' else 0
        delay = max(0, np.random.normal(base + congestion*6 + w_extra, 2))

        rows.append({
            'train_id': train_id,
            'train_type': t_type,
            'scheduled_time': sched_time,
            'time_of_day': tod,
            'day_of_week': dow,
            'weather': weather,
            'congestion': congestion,
            'historical_delay': delay + np.random.normal(0,1),
            'actual_delay': delay
        })
    return pd.DataFrame(rows)

if __name__=="__main__":
    df = gen_synthetic(600)
    df.to_csv("synthetic_trains.csv", index=False)
    print("Saved synthetic_trains.csv with", len(df), "rows")
