import csv as csv
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import svm
import matplotlib.pyplot as plt

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
feats = ['PassengerId', 'Pclass', 'Gender', 'AgeFill', 'SibSp', 'Parch', 'Fare' ]
fate = df['Survived']

clf = svm.SVC()

clf.fit(crunched_data, fate)
y_1 = clf.predict(crunched_data)

print (y_1, "\n", fate)

temp = None
correct = 0
for i in range(len(y_1)):
	if y_1[i] > 0.5:
		temp = 1
	else:
		temp = 0
	if temp == fate[i]:
		correct +=1
	else:
		print(i,'off by',fate[i]-y_1[i])
print('accuracy:',correct/len(y_1))
