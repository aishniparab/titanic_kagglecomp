import csv as csv
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

titanic_data = csv.reader(open('./train.csv'))
header = next(titanic_data)

#PANDAS STUFF

data = []
for row in titanic_data:
	data.append(row)
data = np.array(data)

#print(data)

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

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(crunched_data, fate)
regr_2.fit(crunched_data, fate)
#regr_1.fit(crunched_data, crunched_data)
#regr_2.fit(crunched_data, crunched_data)
#regr_1.fit(fate, crunched_data)
#regr_2.fit(fate, crunched_data)

#X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(crunched_data)
y_2 = regr_2.predict(crunched_data)


l = list()
for pred in y_1:
  if pred > 0.5:
    l.append(1)
  else:
    l.append(0)

plt.figure()

print(crunched_data.size, fate.size)

#plt.scatter(crunched_data, fate, c="k", label="data")
plt.scatter(df['PassengerId'], fate, c="k", label="data")
plt.plot(df['PassengerId'], l, c="g", label="max_depth=2", linewidth=2)
#plt.plot(df['PassengerId'], y_2, c="r", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
