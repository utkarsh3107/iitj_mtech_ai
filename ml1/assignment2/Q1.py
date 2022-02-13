import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from CustomGNB import CustomGNB

df = pd.read_csv("Iris.csv")
X = df.iloc[:, 1:5].values
Y = df.Species.to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

obj = CustomGNB()
obj.fit(X_train, Y_train, X_test, Y_test)
enumeration_dict = obj.compute_enumeration(obj.priors)
predicted_features = obj.predict(X_test)
confusion_matrix = obj.compute_confusion_matrix(predicted_features, Y_test)
print(obj.compute_stats(confusion_matrix))

label_binaries = LabelBinarizer()
y_binaries = label_binaries.fit_transform(Y_test)
prob_scores = obj.predict_prob(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
auc_val = dict()
index = 0
for each in enumeration_dict:
    fpr[index], tpr[index], _ = roc_curve(y_binaries[:, index], prob_scores[:, index])
    roc_auc = auc(fpr[index], tpr[index])
    auc_val[index] = roc_auc_score(y_binaries[:, index], prob_scores[:, index])
    plt.plot(fpr[index], tpr[index], color='Red',
             label='class ' + each + ', accuracy = ' + str(round(auc_val[index], 10) * 100) + '%')
    index = index + 1

plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
