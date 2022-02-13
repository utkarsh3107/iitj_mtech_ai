"""

@author Utkarsh Thusoo
"""
import pandas as pd

df = pd.read_excel("dataset.xlsx")
print(df)


# Decision threshold & ROC for different thresholds
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y = [1,0,1,1,0,1]
y_prob = [0.8, 0.96, 0.4, 0.3, 0.2, 0.7]
y1 = [1,1,1,1,1,1] # for threshold value 0
cf_matrix1=confusion_matrix(y,y1)
print("confusion matrix")
print(cf_matrix1)

fpr, tpr, thresholds = roc_curve(y, y1)
print("True positive rate")
print(tpr)
print("False positive rate")
print(fpr)
print("Thresholds")
print(thresholds)
auc = roc_auc_score(y, y1)
print('AUC - Test Set: %.2f%%' % (auc*100))

plt.plot(fpr, tpr, color='orange', label='ROC at y prob 0')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

y2 = [1,1,1,1,0,1]  # for threshold value 0.2
cf_matrix1=confusion_matrix(y,y2)
print("confusion matrix")
print(cf_matrix1)

fpr, tpr, thresholds = roc_curve(y, y2)
print("True positive rate")
print(tpr)
print("False positive rate")
print(fpr)
print("Thresholds")
print(thresholds)
auc = roc_auc_score(y, y2)
print('AUC - Test Set: %.2f%%' % (auc*100))

plt.plot(fpr, tpr, color='orange', label='ROC at y prob 0.2')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

y3 = [1,1,1,0,0,1]  # for threshold value 0.3
cf_matrix1=confusion_matrix(y,y3)
print("confusion matrix")
print(cf_matrix1)

fpr, tpr, thresholds = roc_curve(y, y3)
print("True positive rate")
print(tpr)
print("False positive rate")
print(fpr)
print("Thresholds")
print(thresholds)
auc = roc_auc_score(y, y3)
print('AUC - Test Set: %.2f%%' % (auc*100))

plt.plot(fpr, tpr, color='orange', label='ROC at y prob 0.3')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

y4 = [1,1,0,0,0,1]  # for threshold value 0.4
cf_matrix1=confusion_matrix(y,y4)
print("confusion matrix")
print(cf_matrix1)

fpr, tpr, thresholds = roc_curve(y, y4)
print("True positive rate")
print(tpr)
print("False positive rate")
print(fpr)
print("Thresholds")
print(thresholds)
auc = roc_auc_score(y, y4)
print('AUC - Test Set: %.2f%%' % (auc*100))

plt.plot(fpr, tpr, color='orange', label='ROC at y prob 0.4')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

y5 = [1,1,0,0,0,0]  # for threshold value 0.4
cf_matrix1=confusion_matrix(y,y5)
print("confusion matrix")
print(cf_matrix1)

fpr, tpr, thresholds = roc_curve(y, y5)
print("True positive rate")
print(tpr)
print("False positive rate")
print(fpr)
print("Thresholds")
print(thresholds)
auc = roc_auc_score(y, y5)
print('AUC - Test Set: %.2f%%' % (auc*100))

plt.plot(fpr, tpr, color='orange', label='ROC at y prob 0.7')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()