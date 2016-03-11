import csv as csv
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import svm
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import itertools

titanic_data = csv.reader(open('./train.csv'))
header = next(titanic_data)

#PANDAS STUFF
data = []
for row in titanic_data:
	data.append(row)
data = np.array(data)
df = pd.read_csv('train.csv', header = 0)

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

# Use 5 particular features that appear the most informative,
# after running different combinations (of sizes 2 or 3) of the original set.
subset_data = df[['Gender', 'SibSp', 'Parch', 'Pclass', 'AgeFill']]

sum = 0
for train_i, test_i in kf:
  # Separate data into training and testing sets.
  cd_train, cd_test = subset_data.iloc[train_i], subset_data.iloc[test_i]
  f_train, f_test = fate[train_i], fate[test_i]
  # Train and predict with SVM.
  clf = None
  clf = svm.SVC()
  clf.fit(cd_train, f_train)
  f_pred = clf.predict(cd_test)
  accu = accuracy_score(f_test, f_pred)
  print(accu)
  sum += accu

print('Average accuracy in cross. valid.:', float(sum/k),'\n')

clf = svm.SVC()
clf.fit(subset_data, fate)
pred = clf.predict(subset_data)
accu = accuracy_score(fate, pred)
print('Accuracy on all examples:', accu,'\n')



# Average of all k folds.
#print('average:',float(sum/k))

