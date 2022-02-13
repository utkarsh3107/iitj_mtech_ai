import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer


class CustomNB:

    def __init__(self, X_train, X_test, Y_train, Y_test):
        """
        :param total_data: Actual dataset provided by the user. Will have all the features
        :param total_predictions: Predictions for each row present in test_data
        :param test_size: Sample distribution value
        :param random_state: Random state where you want to start the test data bifurcation from

        Init operation for the class. Init basic values required for the class
        """
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.total_data = np.vstack((X_train, X_test))
        self.total_predictions = np.concatenate([Y_train, Y_test])
        self.classes = np.unique(self.total_predictions)
        self.total_class = len(self.classes)
        self.training_mean_dict = dict()
        self.training_stddev_dict = dict()
        self.prior_dict = dict()
        self.final_prediction = np.empty(len(self.X_test), dtype=object)
        self.posterior_probability = dict()
        self.stats_df = pd.DataFrame()
        self.confusion_matrix = np.zeros([self.total_class, self.total_class], dtype=int)

    def fit(self):
        temp_dict = self.init_dict(self.X_train, self.Y_train)
        self.training_mean_dict = self.compute_mean(temp_dict)
        self.training_stddev_dict = self.compute_sd(temp_dict, self.training_mean_dict)
        self.prior_dict = self.compute_prior()

    @staticmethod
    def init_dict(training, prediction):
        temp_dict = {}
        index = 0
        for each in prediction:
            if each in temp_dict:
                dict_list = temp_dict[each]
                temp_dict[each] = np.vstack((dict_list, training[index]))
            else:
                temp_dict[each] = training[index]
            index = index + 1
        return temp_dict

    @staticmethod
    def compute_mean(temp_dict):
        mean_dict = {}

        for each in temp_dict:
            arr = temp_dict[each].sum(axis=0) / len(temp_dict[each])
            mean_dict[each] = arr

        return mean_dict

    @staticmethod
    def compute_sd(temp_dict, mean_dict):
        std_dev_dict = {}

        for each in temp_dict:
            mean_list = mean_dict[each]
            index = 0

            for row in temp_dict[each]:
                if each in std_dev_dict:
                    new_row = (row - mean_list) ** 2
                    std_dev_dict[each] = np.add(new_row, std_dev_dict[each])
                else:
                    std_dev_dict[each] = (row - mean_list) ** 2

            arr = np.sqrt(std_dev_dict[each] / len(temp_dict[each]))
            std_dev_dict[each] = arr

        return std_dev_dict

    def compute_prior(self):
        temp_dict = self.init_dict(self.total_data, self.total_predictions)
        prior_dict = {}

        for each in temp_dict:
            prior_dict[each] = len(temp_dict[each]) / len(self.total_data)

        return prior_dict

    def predict_prob(self):
        probabilities = []
        for _test in self.X_test:
            probability_values = []
            temp_arr = np.empty(self.total_class, dtype=object)
            for class_val, sample in self.compute_class_conditionals(_test).items():
                probability_values.append(sample)
            probabilities.append(probability_values)

        return np.array(probabilities)

    def predict(self):
        index = 0
        enumeration_dict = self.compute_enumeration()

        for _test in self.X_test:
            temp_arr = np.empty(self.total_class, dtype=object)

            posterior_prob = self.compute_class_conditionals(_test)
            _max = 0
            for _class in posterior_prob:
                probability = posterior_prob[_class]
                temp_arr[enumeration_dict[_class]] = probability
                if _max < probability:
                    _max = probability
                    self.final_prediction[index] = _class

            self.posterior_probability[index] = temp_arr
            index = index + 1

        return self.final_prediction

    def compute_class_conditionals(self, row):
        class_conditional_dict = {}
        posterior_prob = {}
        # constant_gaussian = np.sqrt(2 * np.pi)
        jiont_prob = 1
        total_marginal_probability = 0
        for each in self.classes:
            mean_arr = self.training_mean_dict[each]
            stddev_arr = self.training_stddev_dict[each]

            numerator = ((row - mean_arr) ** 2) * -1
            denominator = 2 * (stddev_arr ** 2)

            exponent = np.exp(numerator / denominator)
            gaussian_pdf = (1 / (np.sqrt(2 * np.pi * stddev_arr))) * exponent
            class_conditional_dict[each] = np.prod(gaussian_pdf) * self.prior_dict[each]

        print(class_conditional_dict)
        for each in class_conditional_dict:
            total_marginal_probability = total_marginal_probability + class_conditional_dict[each]

        for each in class_conditional_dict:
            posterior_prob[each] = class_conditional_dict[each] / total_marginal_probability

        # print(class_conditional_dict)
        return posterior_prob

    def compute_confusion_matrix(self):
        index = 0
        enumeration_dict = self.compute_enumeration()
        for each in self.Y_test:
            i = enumeration_dict[each]
            j = enumeration_dict[self.final_prediction[index]]
            np.add.at(self.confusion_matrix[i], j, 1)
            index = index + 1

        print('\033[1m' + 'Confusion Matrix:' + '\033[0m')
        print(self.confusion_matrix)
        return self.confusion_matrix

    def compute_enumeration(self):
        enumeration_dict = {}
        index = 0

        for each in self.classes:
            enumeration_dict[each] = index
            index = index + 1

        return enumeration_dict

    def compute_stats(self):
        temp_dict = {}
        self.compute_confusion_matrix()
        actual_cases = np.sum(self.confusion_matrix, axis=1)
        temp_dict['Actual Cases'] = actual_cases

        predicted_cases = np.sum(self.confusion_matrix, axis=0)
        temp_dict['Predicted Cases'] = predicted_cases

        fn_cases = np.zeros([self.total_class], dtype=int)
        tp_cases = np.zeros([self.total_class], dtype=int)

        for row in range(0, self.total_class):
            _sum = 0
            for column in range(0, self.total_class):
                if row == column:
                    tp_cases[row] = self.confusion_matrix[row][column]
                    continue
                _sum = _sum + self.confusion_matrix[row][column]
            fn_cases[row] = _sum

        temp_dict['FN'] = fn_cases
        temp_dict['TP'] = tp_cases

        fp_cases = predicted_cases - tp_cases
        temp_dict['FP'] = fp_cases

        tn_cases = actual_cases.sum() - (actual_cases + fn_cases)
        temp_dict['TN'] = tn_cases

        tnr = tn_cases / (fp_cases + tn_cases)
        temp_dict['TNR'] = tnr

        tpr = tp_cases / (fn_cases + tp_cases)
        temp_dict['TPR'] = tpr

        fpr = fp_cases / (tn_cases + fp_cases)
        temp_dict['FPR'] = fpr

        fnr = fn_cases / (fn_cases + tp_cases)
        temp_dict['FNR'] = fnr

        accuracy = (tpr + tnr) / 2
        temp_dict['Accuracy'] = accuracy

        overall_accuracy = tp_cases.sum() / actual_cases.sum()
        self.print_values("Accuracy", overall_accuracy)

        self.stats_df = pd.DataFrame(temp_dict, index=self.classes)
        self.stats_df.style.set_properties(**{'text-align': 'right'})
        return self.stats_df

    def accuracy(self):
        index = 0
        for each in self.Y_test:
            print("Class %s - %s" % (self.final_prediction[index], each))
            index = index + 1

    def print_values(self, header, values):
        if isinstance(values, np.ndarray):
            print('\033[1m%s: \033[0m' % header)

            for each in range(0, self.total_class):
                if isinstance(values[each], np.float64):
                    print("Class %d: %f" % (each, values[each]))
                elif isinstance(values[each], np.int64):
                    print("Class %d: %d" % (each, values[each]))
        else:
            print('\033[1m%s: \033[0m %s' % (header, values))


df = pd.read_csv("Iris.csv")
X = df.iloc[:, 1:5].values
Y = df.Species.to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
obj = CustomNB(X_train, X_test, Y_train, Y_test)

# obj.fit()
# obj.predict()
# obj.compute_stats()
# print("----")
obj.fit()
obj.predict()
#obj.compute_stats()

label_binarizer = LabelBinarizer()
y_binarize = label_binarizer.fit_transform(Y_test)
prob_scores = obj.predict_prob()
#print(prob_scores)
fpr = dict()
tpr = dict()
roc_auc = dict()
auc_val = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_binarize[:, i], prob_scores[:, i])
    roc_auc = auc(fpr[i], tpr[i])
    auc_val[i] = roc_auc_score(y_binarize[:, i], prob_scores[:, i])
    plt.plot(fpr[i], tpr[i], color='Red',
             label='class ' + str(i) + ', accuracy = ' + str(round(auc_val[i], 10) * 100) + '%')

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# obj.accuracy()
# obj.compute_stats()


# gnb = GaussianNB()
# new_y_pred = gnb.fit(X_train, Y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != new_y_pred).sum()))
"""
wine_data = load_wine()
X = wine_data.data
y = wine_data.target_names[wine_data.target]

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.70, random_state=42)
sss.get_n_splits(X, y)
print(sss)
for train_index, test_index in sss.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    obj1 = CustomNB(X_train, X_test, y_train, y_test)
    obj1.fit()
    obj1.predict()
    obj1.compute_stats()

"""
