# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VAISHNAVI S
RegisterNumber:  212222230165
*/
```
```
import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
     print('Result output')
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam (1).csv",encoding="windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```

## Output:

# DATA.HEAD() :
![Screenshot 2023-11-01 220342](https://github.com/Vaishnavi-saravanan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118541897/5c1777d5-1c5f-4513-858e-9305404b166f)


# DATA.INFO() :

![Uploading Screenshot 2023-11-01 220349.png…]()

# DATA.ISNULL().SUM():
![Screenshot 2023-11-01 220423](https://github.com/Vaishnavi-saravanan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118541897/f7e6a90b-6fa0-48c6-a860-465e165b1c61)


# Y_PRED :
![Screenshot 2023-11-01 220429](https://github.com/Vaishnavi-saravanan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118541897/8d53af89-6948-4af4-b297-3ec142bbd2ad)


# ACCURACY :
![Screenshot 2023-11-01 220440](https://github.com/Vaishnavi-saravanan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118541897/28a0eb7a-05eb-4cb0-a9e7-f41fd453a810)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
