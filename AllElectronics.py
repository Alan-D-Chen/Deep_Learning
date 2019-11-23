from sklearn.feature_extraction import DictVectorizer
import csv
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'E:/dataset/dataset_1/01DTree/AllElectronics.csv', 'rt')

reader = csv.reader(allElectronicsData)

headers = next(reader)

print("reader: \n")
print(reader)
featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print("featureList: \n" , featureList)
print("labelList: \n" ,labelList)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

print("dummyX: \n" + str(dummyX))
print("feature_name: \n" , vec.get_feature_names())

print("labelList: \n" + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: \n" + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: \n" + str(clf))


# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: \n" + str(oneRowX))
####### add a new person.
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: \n" + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY: \n" + str(predictedY))


