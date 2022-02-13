from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

from math import sqrt
from math import pi
from math import exp

"""
some values are hotcoded specific to program.
may not work on different dataset as iloc are hot coded.
"""


class GNB():

    # constructor
    def __init__(self, priors=None):
        self.priors = priors
        print('setting the priors: ' + str(self.priors))
        # self.data = pd.read_csv("iris.csv")
        # self.data.drop(columns='Id',inplace=True)

    # compute mean value
    def mean(self, numbers):
        result = sum(numbers) / float(len(numbers))
        return result

    # compute standard deviation
    def stdev(self, numbers):
        avg = self.mean(numbers)
        squared_diff_list = []
        for num in numbers:
            squared_diff = (num - avg) ** 2
            squared_diff_list.append(squared_diff)
        squared_diff_sum = sum(squared_diff_list)
        sample_n = float(len(numbers) - 1)
        var = squared_diff_sum / sample_n
        return var ** .5

    # for given observation, compute the probability with gaussian PDF
    def gaussian_normal_pdf(self, x, mean, stdev):
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        probability_of_given_sample = (1 / (sqrt(2 * pi) * stdev)) * exponent
        return probability_of_given_sample

    # prepare the frequency table
    def prepare_frequency_table(self):
        frequency_table = dict()
        for i in range(len(self.data)):
            class_value = self.data.iloc[i, -1]
            vector = self.data.iloc[:, 0:self.data.columns.size - 1].values[i]
            if class_value not in frequency_table:
                frequency_table[class_value] = list()
            frequency_table[class_value].append(vector)
        return frequency_table

    # prepare the summarize table with mean, stdev & number_of_observations for the given feature
    # returns (mean, stdev, number_of_observations for the given feature)
    def summarize_dataset(self, dataset):
        ## calculate mean, std dev, and count for each column in dataset
        # summaries = [(np.mean(dataset[column]), np.std(dataset[column]), len(dataset[column])) for column in dataset.iloc[:,0:4]]
        # summary = [(np.mean(col), np.std(col), len(col)) for col in zip(*dataset)]
        summary = [(self.mean(col), self.stdev(col), len(col)) for col in zip(*dataset)]
        # del(summary[-1]) # deleting the length
        return summary

    # prepare the summarize table with mean, stdev & number_of_observations for each class
    def summarize_dataset_by_class(self):
        summarize = dict()
        for class_val in self.freq_table.keys():
            summarize[class_val] = self.summarize_dataset(self.freq_table[class_val])
        return summarize

    # train the dataset
    def fit2(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.data = pd.concat([X_train, y_train], axis=1)
        self.freq_table = self.prepare_frequency_table()
        self.summarize = self.summarize_dataset_by_class()

    # train the dataset
    def fit(self, data):
        self.data = data
        self.freq_table = self.prepare_frequency_table()
        self.summarize = self.summarize_dataset_by_class()

    # predict the posterior probability for given dataset
    def predict(self, x):
        predictions = []
        for observation in x:
            prediction = self.predict_class_with_high_probability_for_given_observation(observation)
            predictions.append(prediction)
        return predictions

    # for given observation, returns the one of the class based on best probability among the classes.
    def predict_class_with_high_probability_for_given_observation(self, observation):
        posterior_probs = self.posterior_probabilities(observation)
        class_with_high_prob, best_prob = None, -1
        for class_value, probability in posterior_probs.items():
            # determine which target class has a higher probability, return this class
            if class_with_high_prob is None or probability > best_prob:
                best_prob = probability
                class_with_high_prob = class_value
        return class_with_high_prob

    def class_priors(self):
        total_number_of_observations = sum([self.summarize[each_class][0][2] for each_class in self.summarize])
        # find the sample size by aggregating the observations from each class
        prior_probabilities = dict()
        if not self.priors == None:
            prior_probabilities = self.priors
            return prior_probabilities
        for class_val, summary_by_features in self.summarize.items():
            # computing the prior probabilities
            # P(class)
            prior_probabilities[class_val] = self.summarize[class_val][0][2] / float(total_number_of_observations)
        return prior_probabilities

    def posterior_probabilities(self, observation):
        # find the sample size by aggregating the observations from each class
        total_number_of_observations = sum([self.summarize[each_class][0][2] for each_class in self.summarize])
        prior_probabilities = dict()
        joint_probabilities = dict()
        posterior_probabilities = dict()

        if not self.priors == None:
            for i in range(len(self.priors)):
                prior_probabilities[np.int32(i)] = self.priors[i]
        for class_val, summary_by_features in self.summarize.items():
            # computing the prior probabilities
            # P(class)
            if not len(prior_probabilities) == 3:
                # this condition has to change in generic way
                # it is working because of both iris and wine dataset has 3 classes.
                prior_probabilities[class_val] = self.summarize[class_val][0][2] / float(total_number_of_observations)
            total_features = len(summary_by_features)
            likelihood = 1

            # computing the joint probability
            # joint probability = prior_probability * likelihood
            # P(class|data) = P(X|class) * P(class)
            for i in range(total_features):
                summary_by_feature = observation[i]
                mean, stdev, no_of_observations = summary_by_features[i]
                normal_prob = self.gaussian_normal_pdf(summary_by_feature, mean, stdev)
                likelihood *= normal_prob
            # print the prior probability to check if it is actually taking the given prior or not.
            # print("prior_probabilities: "+str(prior_probabilities[class_val]))
            joint_probabilities[class_val] = prior_probabilities[class_val] * likelihood

        # computing the marignal probability
        # marginal_probability = sum of all joint probabilities for all classes
        marginal_prob = sum(joint_probabilities.values())

        for class_val, joint_prob in joint_probabilities.items():
            posterior_probabilities[class_val] = joint_prob / marginal_prob
        return posterior_probabilities

    def predict_proba(self, x):
        probabilities = []
        for observation in x:
            probability_values = []
            prob_by_class = self.posterior_probabilities(observation)
            for class_val, sample in prob_by_class.items():
                probability_values.append(sample)
            probabilities.append(probability_values)
        return np.array(probabilities)


data = pd.read_csv('Iris.csv')
print(data['Species'].unique())
data.drop(columns="Id",inplace=True)

x=data.iloc[:,0:4].values
y=data.iloc[:,4].values

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=0)

label_binarizer=LabelBinarizer()
y_binarize=label_binarizer.fit_transform(data['Species'])

gaussian=GNB(priors=None)
gaussian.fit(data)

#accuracy on the train data
y_pred=gaussian.predict(x)

accuracy_naivebayes=round(accuracy_score(y,y_pred)*100,2)
print('accuracy_naivebayes: %.3f' %accuracy_naivebayes)



