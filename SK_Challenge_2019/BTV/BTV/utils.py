import numpy as np, pandas as pd
from datetime import datetime
from keras.callbacks import Callback
import gc

check_k = [1,2,3,5]
top_movies = [5439, 5440, 9872, 9297, 997, 4977, 6708, 9134, 1662, 10730, 7927]

def ready_data(df_train_x, train_label, mlb): 
    np_train = np.array(df_train_x, dtype=np.uint32)
    train_movie = np_train[ :, :, 1]
    train_dur = np_train[:, :, 2]
    train_date = np_train[:, :, 3]
    train_seq = np_train[:, :, 4]
    train_yea, train_mon, train_day, train_wee = date_breaker(train_date)
    train_y= mlb.transform(train_label)

    return train_movie, train_dur, train_seq, train_yea, train_mon, train_day, train_wee, train_y

def date_breaker(date_series):
    ys, ms, ds ,ws = [], [], [], []
    for i, dates in enumerate(date_series):
        if i % 5000 == 0:
            print("\r {}/{}".format(i, len(date_series)), end='')
        y, m, d, w =[], [], [], []
        for date in dates:
            s_datetime = datetime.strptime(str(date), '%Y%m%d')
            y.append(s_datetime.year)
            m.append(s_datetime.month)
            d.append(s_datetime.day)
            w.append(s_datetime.weekday())
        ys.append(y)
        ms.append(m)
        ds.append(d)
        ws.append(w)
    return np.array([ys, ms, ds, ws])

def map_score(y_true, y_pred, topK):
    pred_sort = np.argsort(y_pred)
    
    for k in topK:
        eval_score = 0
        for y, y_ in zip(y_true, pred_sort):
            eval_score += (sum(np.isin(y_[-k:], y))/k)
        eval_score /= len(pred_sort)

        print("MAP - top %d - score: %.6f" % (k, eval_score))

    return eval_score

class Custom_Eval_MAP(Callback):
    def __init__(self, validation_data=(), check_k=[1,2,3,5,10,20,30], interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = map_score(self.y_val, y_pred, check_k)
            gc.collect()
            