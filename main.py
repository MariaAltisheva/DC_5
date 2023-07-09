import pandas as pd

from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn import metrics

def function_mo():
    data = pd.read_csv('dataset.csv')
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3)
    rg = sklearn.ensemble.GradientBoostingRegressor()
    rg.fit(x_train, y_train)
    # print(x_train)
    # return rg
# rg.predict()
# rmse = (sklearn.metrics.mean_squared_error(rg.predict(x_test), y_test)**0.5).astype(float).round(3)
# r2 = (sklearn.metrics.r2_score(rg.predict(x_test), y_test)**0.5).astype(float).round(3)
#
# print(r2)
#     dic = {0: 1.0769230769230769, 1: 24.26546827384644,  2:157.95153635385225,  3:0.02,  4:0.01, 5:0.01, 6:0.0,  7:0.0,  8:0.0, 9:0.0, 10:0.02, 11:0.0, 12:0.02, 13: 2121}
#
#     df = pd.DataFrame.from_dict(dic, orient='index').reset_index().T
#     print(df)
#     print(rg.predict(df))
    return rg

function_mo()