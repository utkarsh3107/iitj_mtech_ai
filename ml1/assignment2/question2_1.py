import pandas as pd
import numpy as np
from GNB import GNB

from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

winedata = load_wine()

# converting the sklearn bunch to dataframce
df=pd.DataFrame(data=winedata.data,columns=[winedata.feature_names])
df['target']=pd.Series(winedata.target)
print(df.shape)
print(df.info())
#print(df.head)

X=df.iloc[:,0:12]
y=df.iloc[:,13]

# plotting feature class distribution
#df.hist(bins=50, figsize=(20,20))
#plt.show()

# plotting heatmap to understand which feature's have high covarience

# splitting the data in 70:30 ratio with shffule and default stratified fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42, shuffle=True, stratify=y)

# plotting class-wise distribution

unique=y.unique()
frequency_y_train=y_train.value_counts(sort=False)

frequency_y_test=y_test.value_counts(sort=False)
plt.bar(list(map(str, unique)),frequency_y_train)
plt.bar(list(map(str, unique)),frequency_y_test)
plt.show()


gaussian=GNB(priors=None)
gaussian.fit2(X_train,y_train)

print('class priors: '+str(gaussian.class_priors()))

print('mean and variance of each feature per class')
for class_val,feature_metrics in gaussian.summarize.items():
    print('\nmean and variance for class: '+str(class_val))
    for each_feature in range(len(feature_metrics)):
        print(feature_metrics[each_feature][0], feature_metrics[each_feature][1] ** 2)


y_pred=gaussian.predict(np.asarray(X_test))
cf_matrix=confusion_matrix(y_test,y_pred)
print('test data confusion matrix')
print(cf_matrix)

accuracy=round(accuracy_score(y_test,y_pred)*100,2)
print('test data accuracy: %.3f' %accuracy)


X_pred=gaussian.predict(np.asarray(X_train))
cf_matrix=confusion_matrix(y_train,X_pred)
print('train data confusion matrix')
print(cf_matrix)

accuracy=round(accuracy_score(y_train,X_pred)*100,2)
print('train data accuracy: %.3f' %accuracy)