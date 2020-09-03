import pandas as pd
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
import random
from random import randint
from sklearn.metrics import accuracy_score
import statistics
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("Iris.csv")

print(df.Species.unique())

# graph differences in flower length stuff between which 
setosa = df.loc[df['Species'] == 'Iris-setosa']
versi = df.loc[df['Species'] == 'Iris-versicolor']
virgin = df.loc[df['Species'] == 'Iris-virginica']

# Plotting Petal Numbers Scatter
plt.figure()
plt.scatter(setosa['PetalWidthCm'],setosa['PetalLengthCm'])
plt.scatter(versi['PetalWidthCm'],versi['PetalLengthCm'])
plt.scatter(virgin['PetalWidthCm'],virgin['PetalLengthCm'])

plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
plt.xlabel('PetalWidthCm')
plt.ylabel('PetalLengthCm')
plt.title('Scatterplot of Petal Ratios of 3 Species')

# Plotting Sepal Numbers Scatter
plt.figure()
plt.scatter(setosa['SepalWidthCm'],setosa['SepalLengthCm'])
plt.scatter(versi['SepalWidthCm'],versi['SepalLengthCm'])
plt.scatter(virgin['SepalWidthCm'],virgin['SepalLengthCm'])

plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
plt.xlabel('SepalWidthCm')
plt.ylabel('SepalLengthCm')
plt.title('Scatterplot of Sepal Ratios of 3 Species')

plt.figure()
sb.distplot(setosa['PetalLengthCm'])
sb.distplot(versi['PetalLengthCm'])
sb.distplot(virgin['PetalLengthCm'])
plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
plt.xlabel('Cm')
plt.title('Iceborn Histogram of Petal Length 3 Species')

plt.figure()
sb.distplot(setosa['SepalLengthCm'])
sb.distplot(versi['SepalLengthCm'])
sb.distplot(virgin['SepalLengthCm'])
plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
plt.xlabel('Cm')
plt.title('Iceborn Histogram of Sepal Length 3 Species')

plt.show()
# graphing

# Seperating and removing dependent categorical variable
tclass = df['Species']
del df['Species']
del df['Id']

# Petal data only
dfPetal = df[['PetalLengthCm','PetalWidthCm']]
# Sepal data only
dfSepal = df[['SepalLengthCm','SepalLengthCm']]

# Holding all instances of models created and predicted
accArrAll = []
accArrPetal = []
accArrSepal = []

# randomized list of nums between 1 and 100
randnumArr = []

for x in range(0,101):
    temp = randint(1,10001)
    if temp not in randnumArr:
        randnumArr.append(temp)

print(str(len(randnumArr)))
# running 100 test models per type of model
count = 0
while count < 101:

    # All data
    Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(df, tclass, test_size = 0.2, random_state = randnumArr[count])
    treeModel1 = tree.DecisionTreeClassifier()
    treeModel1.fit(Xtrain1,Ytrain1)
    Ypredict1 = treeModel1.predict(Xtest1)
    acc1 = accuracy_score(Ytest1,Ypredict1)
    accArrAll.append(acc1)

    # Petal only data
    Xtrain2, Xtest2, Ytrain2, Ytest2 = train_test_split(dfPetal, tclass, test_size = 0.2, random_state = randnumArr[count])
    treeModel2 = tree.DecisionTreeClassifier()
    treeModel2.fit(Xtrain2,Ytrain2)
    Ypredict2 = treeModel2.predict(Xtest2)
    acc2 = accuracy_score(Ytest2,Ypredict2)
    accArrPetal.append(acc2)

    # Sepal only data
    Xtrain3, Xtest3, Ytrain3, Ytest3 = train_test_split(dfSepal, tclass, test_size = 0.2, random_state = randnumArr[count])
    treeModel3 = tree.DecisionTreeClassifier()
    treeModel3.fit(Xtrain3,Ytrain3)
    Ypredict3 = treeModel3.predict(Xtest3)
    acc3 = accuracy_score(Ytest3,Ypredict3)
    accArrSepal.append(acc3)

    count += 1


print("Accuracy average for all data and 100 models: " + str(round(mean(accArrAll),3)))
print("Accuracy average for Petal only data and 100 models: " + str(round(mean(accArrPetal),3)))
print("Accuracy average for Sepal only data and 100 models: " + str(round(mean(accArrSepal),3)))