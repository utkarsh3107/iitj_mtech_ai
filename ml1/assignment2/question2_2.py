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

X=df.iloc[:,0:12]
y=df.iloc[:,13]


# splitting the data in 70:30 ratio with shffule and default stratified fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42, shuffle=True, stratify=y)

'''
40-40-20 ratio.
40+40+20=100
178/100=1.78
1.78*40,1.78*40,1.78*20
178 dividing into 40-40-20 ratio's mean 71,71,36 and probabilities are

71/178=0.39887640449438202247191011235955
71/178=0.39887640449438202247191011235955
36/178=0.2022471910112359550561797752809
'''

'''
80-10-10 ratio.
80+10+10=100
178/100=1.78
1.78*80,1.78*10,1.78*10
142.4,17.8,17.8
178 dividing into 80-10-10 ratio's mean 142,18,18 and probabilities are

142/178=0.7977528089887640449438202247191
18/178=0.10112359550561797752808988764045
18/178=0.10112359550561797752808988764045
'''
gaussian2=GNB(priors=[0.79, 0.10, 0.10])
gaussian2.fit2(X_train,y_train)

print('class priors: '+str(gaussian2.class_priors()))

print('mean and variance of each feature per class')
for class_val,feature_metrics in gaussian2.summarize.items():
    print('\nmean and variance for class: '+str(class_val))
    for each_feature in range(len(feature_metrics)):
        print(feature_metrics[each_feature][0], feature_metrics[each_feature][1] ** 2)

y_pred=gaussian2.predict(np.asarray(X_test))
cf_matrix=confusion_matrix(y_test,y_pred)
print('test data confusion matrix')
print(cf_matrix)

accuracy=round(accuracy_score(y_test,y_pred)*100,2)
print('test data accuracy: %.3f' %accuracy)

X_pred=gaussian2.predict(np.asarray(X_train))
cf_matrix=confusion_matrix(y_train,X_pred)
print('train data confusion matrix')
print(cf_matrix)

accuracy=round(accuracy_score(y_train,X_pred)*100,2)
print('train data accuracy: %.3f' %accuracy)