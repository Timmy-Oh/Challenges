import os
import datetime
import pandas as pd
import numpy as np

X1 = ['X00', 'X07', 'X28', 'X31', 'X32']
X2 = ['X01', 'X06', 'X22', 'X27', 'X29']
X3 = ['X02', 'X03', 'X18', 'X24', 'X26']
X4 = ['X04', 'X10', 'X21', 'X36', 'X39']
X5 = ['X05', 'X08', 'X09', 'X23', 'X33']
X6 = ['X11', 'X14', 'X16', 'X19', 'X34']
X7 = ['X12', 'X20', 'X30', 'X37', 'X38']
X8 = ['X13', 'X15', 'X17', 'X25', 'X35']

Xs = np.concatenate([X1, X2, X3, X4, X5, X6, X7, X8])

Ys = ['Y{}'.format(str(i).zfill(2)) for i in range(19)]


# 대회 평가
def mse_AIFrenz(y_true, y_pred):
    '''
    y_true: 실제 값
    y_pred: 예측 값
    '''
    diff = abs(y_true - y_pred)
    
    less_then_one = np.where(diff < 1, 0, diff)
    
    # multi-column일 경우에도 계산 할 수 있도록 np.average를 한번 더 씌움
    score = np.average(np.average(less_then_one ** 2, axis = 0))
    
    return score


# Submission 파일 작성
def writeSubm(pred):
    subm = pd.read_csv('./data/sample_submission.csv')
    subm['Y18'] = pred
    dt = datetime.datetime.now().strftime('%m%d%H%M%S')
    
    os.makedirs('./subm', exist_ok=True)
    file_path = './subm/subm_{}.csv'.format(dt)
    subm.to_csv(file_path, index=False)
    print('Submission file is written on {}'.format(file_path))
