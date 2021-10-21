import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/decision tree/Iris.csv")
df.head()

df.isnull().sum()
df.shape
df.info()
df.describe()

df.drop('Id',axis=1, inplace=True)
df.shape
df['Species'].value_counts().plot(kind='pie', autopct="%.1f%%")
df.corr()
sns.heatmap(df.corr(), annot=True)
x = df.iloc[:,:4].values
y = df.iloc[:,4:5]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

print("Accuracy: ", metrics.accuracy_score(y_test,y_pred))

new_data = [[3.5, 3.0, 1.2, 1.7]]
y_pred = model.predict(new_data)
print(y_pred)

from sklearn import tree
import matplotlib.pyplot as plt
plt.figure(figsize = (20,10))
tree.plot_tree(model, filled=True, rounded=True)
plt.show()



