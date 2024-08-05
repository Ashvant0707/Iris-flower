import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("E:/Machine Learning\data/Iris_new.csv")
x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,-1].values
##splitting 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=11,criterion="entropy",random_state=0)
##nestimators  -> number of trees
classifier.fit(x_train,y_train)
##predicting
y_pred=classifier.predict(x_test)
dataset.columns
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
acc=(sum(np.diag(cm))/len(y_test))
acc



feature_col=dataset
feature_col=feature_col.drop("spectype",axis=1)
##visuaization
from sklearn import tree
plt.figure(dpi=300)
tree.plot_tree(classifier.estimators_[10],
                    feature_names=feature_col.columns,
                    class_names=["setosa","vercicolor","virginica"],
                    filled=False,precision=4,rounded=False,fontsize=4)
plt.show()







from suprise import Dataset, Reader
from suprise import SVD
from suprise.model_selection import train_test_split
from suprise import accuracy
data=Dataset.load_builtin("ml-10k")
trainset,testset = train_test_split(data, test_sie=0.25)
algo=SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)