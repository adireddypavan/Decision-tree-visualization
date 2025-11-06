# Decision-tree-visualization
python program to build the decision tree data visualization


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
iris=load_iris()
X=pd.DataFrame(iris.data,columns=iris.feature_names)
y=pd.Series(iris.target,name='target')
print("Dataset Previews:")
print(X.head(),"\n")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(criterion='entropy',random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Decision Tree Classification Results:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))
plt.figure(figsize=(12,8))
plot_tree(model,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.title("Decision Tree Visualization")
plt.show()

