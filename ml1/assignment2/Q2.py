from CustomGNB import CustomGNB
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine_data = load_wine()
X = wine_data.data
y = wine_data.target_names[wine_data.target]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

obj = CustomGNB()
obj.fit1(X_train, Y_train)


prediction = obj.predict(X_test)
cf_matrix = obj.compute_confusion_matrix(prediction, Y_test)
print(obj.compute_stats(cf_matrix))


prediction = obj.predict(X_train)
cf_matrix = obj.compute_confusion_matrix(Y_train, prediction)
print(obj.compute_stats(cf_matrix))


priors = {'class_0': 0.79, 'class_1': 0.10, 'class_2': 0.10}
obj = CustomGNB(priors)

obj.fit1(X_train, Y_train)


prediction = obj.predict(X_test)
cf_matrix = obj.compute_confusion_matrix(prediction, Y_test)
print(obj.compute_stats(cf_matrix))


prediction = obj.predict(X_train)
cf_matrix = obj.compute_confusion_matrix(Y_train, prediction)
print(obj.compute_stats(cf_matrix))






