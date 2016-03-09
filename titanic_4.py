import csv as csv
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

titanic_data = csv.reader(open('./train.csv'))
header = next(titanic_data)

#PANDAS STUFF
data = []
for row in titanic_data:
	data.append(row)
data = np.array(data)
df = pd.read_csv('train.csv', header = 0)
#print(df.describe())

#SCIKIT STUFF
#Should we include Survived in crunched_data?
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['AgeFill'] = df['Age']
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']]

crunched_data = df[ ['PassengerId', 'Pclass', 'Gender', 'AgeFill', 'SibSp', 'Parch', 'Fare'] ]
fate = df['Survived']

### Cross validation: k-fold ###
# Num of folds:
k = 10
# Num of total examples:
n = len(crunched_data)
# Shuffle rows of data before sampling, to avoid any
# implicit structures in the data.
kf = KFold(n, n_folds = k, shuffle = True)

sum_1 = 0
sum_2 = 0
for train_i, test_i in kf:
  # Separate data into training and testing sets.
  cd_train, cd_test = crunched_data.iloc[train_i], crunched_data.iloc[test_i]
  f_train, f_test = fate[train_i], fate[test_i]
  # Train and predict with DT Regressor.
  # Should we use DT Classifier?
  regr_1 = DecisionTreeRegressor(max_depth=2)
  regr_2 = DecisionTreeRegressor(max_depth=5)
  regr_1.fit(cd_train, f_train)
  regr_2.fit(cd_train, f_train)
  pred_1 = regr_1.predict(cd_test)
  pred_2 = regr_2.predict(cd_test)
  pred_1_mod = []
  # Modify continuous predictions to be discrete.
  # Threshold at 0.5.
  threshold = 0.5
  for pred in pred_1:
    if pred > threshold:
      pred_1_mod.append(1)
    else:
      pred_1_mod.append(0)
  pred_2_mod = []
  for pred in pred_2:
    if pred > threshold:
      pred_2_mod.append(1)
    else:
      pred_2_mod.append(0)
  accu_1 = accuracy_score(f_test, pred_1_mod)
  accu_2 = accuracy_score(f_test, pred_2_mod)
  sum_1 += accu_1
  sum_2 += accu_2
  print('Accuracy (depth 2):',accu_1)
  print('Accuracy (depth 5):',accu_2)

# Average of all k folds on both depths.
print('Average accuracy (depth 2):', sum_1/k)
print('Average accuracy (depth 5):', sum_2/k)

