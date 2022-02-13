import numpy as np
import pandas as pd


class CustomGNB:

    def __init__(self, priors=None):
        """
        Init operation for the CustomGNB class
        :param priors: Will be the prior probabilities
        """
        self.priors = priors
        self.training_mean_dict = dict()
        self.training_stddev_dict = dict()

    def fit1(self, features, classes):
        """
        Fitness functions which calculates mean, standard deviation and priors for the provided dataset

        :param features: Features of dataset
        :param classes: Classes for each of that feature
        :return: null
        """
        temp_dict = self.init_dict(features, classes)
        self.training_mean_dict = self.compute_mean(temp_dict)
        self.training_stddev_dict = self.compute_sd(temp_dict, self.training_mean_dict)
        self.priors = self.compute_prior(features, classes)

    def fit(self, training_features, training_classes, testing_features, testing_classes):
        """
        Fitness functions which calculates mean, standard deviation and priors for the provided dataset
        :param training_features: Features of training dataset
        :param training_classes: Classes for each of that feature in training dataset
        :param testing_features: Features of testing dataset
        :param testing_classes: Classes for each of that feature in testing dataset
        :return: null
        """
        temp_dict = self.init_dict(training_features, training_classes)
        self.training_mean_dict = self.compute_mean(temp_dict)
        self.training_stddev_dict = self.compute_sd(temp_dict, self.training_mean_dict)
        total_data = np.vstack((training_features, testing_features))
        total_predictions = np.concatenate([training_classes, testing_classes])
        self.priors = self.compute_prior(total_data, total_predictions)

    @staticmethod
    def init_dict(training, prediction):
        """
        Inititalizes a dictionary with key as each class and value as the row provided

        :param training: training data
        :param prediction: training predictions
        :return:
        """
        temp_dict = dict()
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
        """
        Computes the mean of the dataset

        :param temp_dict: dictionary
        :return: dictionary with key as classes and value as mean feature array
        """
        mean_dict = {}

        for each in temp_dict:
            arr = temp_dict[each].sum(axis=0) / len(temp_dict[each])
            mean_dict[each] = arr

        return mean_dict

    @staticmethod
    def compute_sd(temp_dict, mean_dict):
        """
        Computes the standard deviation for dataset

        :param temp_dict: dictionary
        :param mean_dict: dictionary with key as classes and value as mean feature array
        :return: dictionary with key as classes and value as standard deviation feature array
        """
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

    def compute_prior(self, features, classes):
        """
        Computes the priors for the features and class provided

        :param features: Features provided
        :param classes: Classes provided
        :return: dictionary with key as classes and value as prior values for that class
        """
        prior_dict = dict()

        if self.priors is None:
            temp_dict = self.init_dict(features, classes)

            for each in sorted(temp_dict.keys()):
                prior_dict[each] = len(temp_dict[each]) / len(features)
        else:
            prior_dict = self.priors

        return prior_dict

    def predict(self, features):
        """
        Performs predictions for the features provided

        :param features: Feature array
        :return: prediction array
        """
        index = 0
        enumeration_dict = self.compute_enumeration(self.priors)
        prediction = np.empty(len(features), dtype=object)
        for _test in features:
            temp_arr = np.empty(len(self.priors), dtype=object)
            posterior_prob = self.compute_posterior_probability(_test)
            _max = 0
            for _class in posterior_prob:
                probability = posterior_prob[_class]
                temp_arr[enumeration_dict[_class]] = probability
                if _max < probability:
                    _max = probability
                    prediction[index] = _class

            index = index + 1

        return prediction

    def predict_prob(self, features):
        """
        Computes the probabilities for each of the feature provided in the dataset based based on each class.
        :param features:
        :return: probability array
        """
        probabilities = []
        for _test in features:
            probability_values = []
            for class_val, sample in self.compute_posterior_probability(_test).items():
                probability_values.append(sample)
            probabilities.append(probability_values)

        return np.array(probabilities)

    def compute_posterior_probability(self, row, classes=None):
        """
        Computues posterior probability for each feature.
        P(class|data) = P(X|class) * P(class) / marginal probability
        :param row: Array containing features
        :param classes: prior probabiities for each class
        :return: dictionary with key as classes and value as posterior probability
        """
        if classes is None:
            classes = self.priors

        class_conditional_dict = dict()
        posterior_prob = dict()

        total_marginal_probability = 0
        for each in classes.keys():
            mean_arr = self.training_mean_dict[each]
            stddev_arr = self.training_stddev_dict[each]

            numerator = ((row - mean_arr) ** 2) * -1
            denominator = 2 * (stddev_arr ** 2)

            exponent = np.exp(numerator / denominator)
            ## Formulae:  N(x; µ, σ) = (1 / 2πσ) * (e ^ (x–µ)^2/-2σ^2
            gaussian_pdf = (1 / (np.sqrt(2 * np.pi * stddev_arr))) * exponent
            class_conditional_dict[each] = np.prod(gaussian_pdf) * self.priors[each]
            total_marginal_probability = total_marginal_probability + class_conditional_dict[each]

        for each in class_conditional_dict:
            posterior_prob[each] = class_conditional_dict[each] / total_marginal_probability

        return posterior_prob

    @staticmethod
    def compute_enumeration(classes):
        """
        Computes a dictionary with key as class name and value as the index from 0 to N(number of classes)
        :param classes: Class list
        :return: dictionary with key-value pair where key is class name and value as the index
        """
        enumeration_dict = dict()
        index = 0

        for each in sorted(classes.keys()):
            enumeration_dict[each] = index
            index = index + 1

        return enumeration_dict

    def print_values(self, header, values):
        if isinstance(values, np.ndarray):
            print('\033[1m%s: \033[0m' % header)

            for each in range(0, len(self.priors)):
                if isinstance(values[each], np.float64):
                    print("Class %d: %f" % (each, values[each]))
                elif isinstance(values[each], np.int64):
                    print("Class %d: %d" % (each, values[each]))
        else:
            print('\033[1m%s: \033[0m %s' % (header, values))

    def compute_confusion_matrix(self, predictions, actual):
        """
        Computes confusion matrix
        :param predictions: predictions
        :param actual: actual values
        :return: NXN confusion matrix
        """
        enumeration_dict = self.compute_enumeration(self.priors)

        confusion_matrix = np.zeros([len(self.priors.keys()), len(self.priors.keys())], dtype=int)
        index = 0
        for each in actual:
            i = enumeration_dict[each]
            j = enumeration_dict[predictions[index]]
            np.add.at(confusion_matrix[i], j, 1)
            index = index + 1

        print('\033[1m' + 'Confusion Matrix:' + '\033[0m')
        print(confusion_matrix)
        return confusion_matrix

    @staticmethod
    def accuracy(predicted_classes, actual_classes):
        index = 0
        for each in actual_classes:
            print("Class %s - %s" % (predicted_classes[index], actual_classes[each]))
            index = index + 1

    def compute_stats(self, confusion_matrix):
        self.print_values("Priors", self.priors)
        self.print_values("Mean", self.training_mean_dict)
        self.print_values("Standard Deviation", self.training_stddev_dict)
        variance = dict()
        for each in self.training_stddev_dict:
            variance[each] = self.training_stddev_dict[each] ** 2
        self.print_values("Variance", variance)

        total_class = len(self.priors.keys())
        self.print_values("Total Classes", total_class)

        classes = self.compute_enumeration(self.priors).keys()
        self.print_values("Enumerations", classes)

        temp_dict = {}
        actual_cases = np.sum(confusion_matrix, axis=1)
        temp_dict['Actual Cases'] = actual_cases

        predicted_cases = np.sum(confusion_matrix, axis=0)
        temp_dict['Predicted Cases'] = predicted_cases

        fn_cases = np.zeros([total_class], dtype=int)
        tp_cases = np.zeros([total_class], dtype=int)

        for row in range(0, total_class):
            _sum = 0
            for column in range(0, total_class):
                if row == column:
                    tp_cases[row] = confusion_matrix[row][column]
                    continue
                _sum = _sum + confusion_matrix[row][column]
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
        temp_dict['Accuracy'] = accuracy * 100

        overall_accuracy = tp_cases.sum() / actual_cases.sum()
        self.print_values("Accuracy", overall_accuracy * 100)

        stats_df = pd.DataFrame(temp_dict, index=classes)
        stats_df.style.set_properties(**{'text-align': 'right'})
        return stats_df
