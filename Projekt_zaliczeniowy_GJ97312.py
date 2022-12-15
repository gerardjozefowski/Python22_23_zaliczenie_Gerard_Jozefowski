#0 Importing library

import pandas as pd
import numpy as np
import math
import seaborn as sn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

#1 Loading Data 
def load_data():
    path =  'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
    cc_apps = pd.read_csv(path, header = None)
    return cc_apps

cc_apps = load_data()

#2 Analysis of descriptive statistics

def describe_data(data):
    print("Data description")
    
    print(data.describe())
    
    print('\n')

    print("Data Info")
    
    print(data.info())
    
    print('\n')
    
    print("Null values by columns")
    
    print(data.isnull().sum())
    
describe_data(cc_apps)

# Conlusions:
    #there is no null values in the data - possible problems



print(cc_apps.tail(20))
# there is definietly some question marks

# counting '?' by column: 

def counter_function(data):
    counter = {}
    for row in data:
        if row not in counter:
            counter[row] = 0
        counter[row] += 1
    print(counter)


for n in range(16):
    counter_function(cc_apps[n])


# Deleting '?' with NaN
c_apps = cc_apps.replace('?', np.NaN)
describe_data(cc_apps)


# Dropping NA - the NaN doesn't represent that 
cc_apps = cc_apps.dropna()
describe_data(cc_apps)


#2 Data Visualisation - viusual analysis of the results

#visualisation of the de

def my_hist_function(data):
    data.hist(bins=10, figsize=(4, 5))

my_hist_function(cc_apps)
# as we can see the data is highly inbalanced, in the future it may violate the modelling results # we need to logarithm it 


# Correlation function matrix

def correlation_matrix(data):
    corr_matrix = data.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()
    
correlation_matrix(cc_apps)

# We do not need to remove any variable as multilonearity is unlikely


# 3 Data transformation
print(cc_apps.dtypes)


for n in [2,7,10,14]:
    print(min(cc_apps[n]))

my_hist_function(cc_apps)

# we normalised it a little bit now we can proceed to model it mathematically

for n in [2,7,10,14]:
    cc_apps[n] = cc_apps[n] + 10
    cc_apps[n] = np.log(cc_apps[n])
my_hist_function(cc_apps)

# now we can proceed to model it


#4 Data pre-processing

# Dropping 11 and 13 - zip codes and driver license
cc_apps = cc_apps.drop([11, 13], axis=1)


# Dividing the data

cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)

# Transforamcja zmiennych jakosciowych na zmienne dummies
def dummies(data):
    data = pd.get_dummies(data)


dummies(cc_apps_train)
dummies(cc_apps_test)

cc_apps_test = cc_apps_test.reindex(columns=cc_apps_train.columns,
                                    fill_value=0)


# Podział na zmienną zależną i zmienną niezależną
X_train, y_train = cc_apps_train.iloc[:, :-1].values, cc_apps_train.iloc[:, [-1]].values
X_test, y_test = cc_apps_test.iloc[:, :-1].values, cc_apps_test.iloc[:, [-1]].values

# Creating an object logreg
reg_log = LogisticRegression()

# Creating statistical model
reg_log.fit(X_train,y_train)

# Using logistic regression on test data
y_pred = reg_log.predict(X_test)
print("Accuracy: ", reg_log.score(X_test,y_test))
confusion_matrix(y_test,y_pred)

# We do not need to proceed with GridSearch, the logistic regression results are perfect