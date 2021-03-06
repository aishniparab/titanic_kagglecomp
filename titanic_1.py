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
print(df.describe())

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
#crunched_data = [list(df[feat]) for feat in feats]
fate = df['Survived']

print(fate, crunched_data)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(crunched_data, fate)
regr_2.fit(crunched_data, fate)
#regr_1.fit(fate, crunched_data)
#regr_2.fit(fate, crunched_data)
#regr_1.fit(fate, crunched_data)
#regr_2.fit(fate, crunched_data)

#X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(crunched_data)
y_2 = regr_2.predict(crunched_data)

plt.figure()

crunched_data = np.array(crunched_data)
fate = np.array(fate)
print(crunched_data.size, fate.size)
#print(dir(crunched_data))

rows = [row for row in crunched_data]

temp = None
correct = 0
for i in range(len(y_1)):
	if y_1[i] > 0.5:
		temp = 1
	else:
		temp = 0
	if temp == fate[i]:
		correct +=1
print('accuracy:',correct/len(y_1))

print (y_1, y_2, "\n", fate)

#plt.scatter(crunched_data, fate, c="k", label="data")
#plt.scatter(rows, fate, c="k", label="data")
#plt.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
#plt.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
#plt.xlabel("data")
#plt.ylabel("target")
#plt.title("Decision Tree Regression")
#plt.legend()
#plt.show()
