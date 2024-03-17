RAINFALL PREDICTION


 Demo video: https://drive.google.com/file/d/1qLNu6yrVAw6Vh2Holbg2SrOv0fyBXADK/view?usp=drivesdk

 Team details
 Team  number:VH080
 --------------------------------------------------
 |   Name               |        E-mail           |
 | ---------------------|------------------------ |
 |Y.BHARATH             |  ybharath2093@gmail.com |
 ||
 |Y.BHARATH SIMHA REDDY | 99210041662@klu.ac.in   |
 ||
 | V.MAHESH REDDY       |  9921004746@klu.ac.in   |
 | |
 | G.SHYAM              | 9921009023@Klu.ac.in    |
 --------------------------------------------------

 Problem statement:
 The ability to predict rainfall iste a crucial issue for both meteorology and agriculture. Accurate rainfall forecasts can aid farmers in crop planning, irrigation control, and the prevention of crop loss due to drought or flooding. Accurate rainfall forecasts can also aid in the management of disasters and floods.

About the project:
Machine learning has been widely applied to rainfall prediction. The basic idea behind machine learning is to create a model that can forecast future rainfall based on past rainfall data and other pertinent components. To do this, a significant quantity of historical data must be analyzed, including details on weather patterns, air pressure, temperature, wind speed, and other environmental factors.
Many machine learning techniques, such as logistic regression, support vector machines (SVMs), decision trees, neural networks, regression, and random forests, can be used to predict rainfall.

Technical implemntion:
They can also handle non-linear relationships between meteorological variables and rainfall and provide real-time updates, making them more suitable for applications such as flood forecasting and disaster management.
User: The user is the primary contributor to the model because he is the one who worked on it
Input: Various weather-related variables that are known to affect rainfall patterns are frequently included in the input data for rainfall prediction using machine learning. Here are a few typical illustrations of input variables that can be used to forecast rainfall: 
1.Temperature
2.Humidity
3.Wind speed and direction
Data pre-processing: Cleanse, eliminate any outliers or errors from the data, and then transform it into a format that the machine learning program can use
Data splitting: The preprocessed data should be divided into training and testing sets. The machine learning algorithm will be trained using the training set, and its performance will be assessed using the testing set.

Data training: Utilise the training data set to train the machine learning algorithm. To forecast future rainfall, the computer will learn from past weather information.

Data Testing: The performance of the model can be measured using metrics such as accuracy, precision, recall, and F1 score.

Result or prediction: Machine learning models for rainfall prediction are generally accurate and can be used in many applications, such as agricultural planning, flood predictions, and water resource management.
![image](https://github.com/ybhara/rainfall-prediction/assets/161044637/5dde7191-8100-4c78-b7ca-c237f5bd943c)

code:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns ; sns.set()
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("weatherAUS.csv")
data.head()

data.cloums

data.shape

data.isnull().sum()

data=data.dropna()
data
data.isnull().sum()

data.describe()
data.columns
data.RainToday.value_counts()
g=sns.countplot(x='RainToday',data=data)
g=sns.distplot(data['Rainfall'])

data['RainToday']=data['RainToday'].apply(lambda x:1 if x=='Yes' else 0)
train=data[['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']]
data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x:1 if x=='Yes' else 0)
label=data['RainTomorrow']
from sklearn.model_selection import train_test_split
x_train ,y_train ,x_test ,y_test = train_test_split(train,label,test_size=0.6 )

LOGISTIC REGRESSION

from sklearn. linear_model import LogisticRegression from sklearn.metrics import accuracy_score

 mod1 Logistickegression()

 mod1.fit(x_train,x_test)


 1f-mod1.predict(y_train)

 acc-accuracy_score(y_test,lf) acc

from sklearn.metrics import confusion matrix

 cn-confusion_matrix(y_test, 1f)

 labels [0,1]

sns.heatmap(cn, annot True, cmap="Y1GnBu", fmt=",3f", xticklabels labels, yticklabels plt.show()

RANDOM FOREST 

from sklearn.ensemble import Random ForestClassifier

mod2 Random ForestClassifier()

 mod2.fit(x_train,x_test)

 RFwmod2.predict(y_train)

 acc1 accuracy score(y_test, RF) acci



from sklearn.metrics import confusion_matrix

cn1 confusion_matrix(y_test, mod2.predict(y_train))

 labels = [0,1] sns.heatmap(cn1, annot True, cmap="YlGnllu", fat".3f", xticklabels labels,yticklabel plt.show()

 import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

objects (Logistic Regression', 'Random Forest')

y_pos=np.arange(len(objects)) performance [acc, acc1]

plt.bar(y_pos, performance, align='center', alpha=0.5) plt.xticks(y_pos,objects)

plt.ylabel("Accuracy Score')

plt.title("Logistic Regression v/s Random Forest')

plt.show()


output:
![Screenshot](https://github.com/ybhara/rainfall-prediction/assets/161044637/737f2db8-903a-4425-b807-4a6c66523971)
![Screenshot ](https://github.com/ybhara/rainfall-prediction/assets/161044637/ddf254de-ad35-4dda-b5a0-6c601fec2c21)
![Screenshot ()](https://github.com/ybhara/rainfall-prediction/assets/161044637/76031f43-480f-4d29-b849-4109a0fa14f8)
![Screenshot](https://github.com/ybhara/rainfall-prediction/assets/161044637/4c868506-7de6-45c5-b6ec-699004f9fc8c)

what's next?
we are plan future devolpment of websit to our project people  can see  rainfall.

Declaration:
We confirm that the project showcased here was either developed entirely during the hackathon or underwent significant updates within the hackathon timeframe. We understand that if any plagiarism from online sources is detected, our project will be disqualified, and our participation in the hackathon will be revoked.





















