import math
import time

import Perceptron.helper as helper
import Project.dataParser as dtP


def avg(vector):
    vector_sum = 0
    for i in vector:
        vector_sum += i
    return float(vector_sum)/len(vector)


class NaiveBayes:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable

        self.attribute_set = {}
        self.attribute_probabilities = {}

        self.__create_attribute_set()
        self.raw_prior, self.label_counter = self.calculate_prior()
        self.labels = self.label_counter.keys()

    def __create_attribute_set(self):
        for line in self.data:
            for k, v in line.items():
                if k == "label":
                    continue
                if k in self.attribute_set:
                    self.attribute_set[k].add(v)
                else:
                    self.attribute_set[k] = {'0', v}

    def calculate_prior(self):
        label_counter = {}
        for dataLine in self.data:
            label = dataLine["label"]
            if label in label_counter:
                label_counter[label] += 1
            else:
                label_counter[label] = 1

        total = sum(label_counter.values())
        label_prior = {}
        for k, v in label_counter.items():
            label_prior[k] = math.log(v / total)
        return label_prior, label_counter

    def train_model(self, smoothing_term):
        attribute_counter = {}

        for data_line in self.data:
            label = data_line["label"]
            for attribute in self.attribute_set.keys():
                if attribute in data_line.keys():
                    value = data_line[attribute]
                else:
                    value = '0'
                if attribute in attribute_counter.keys():
                    if value in attribute_counter[attribute].keys():
                        if label in attribute_counter[attribute][value].keys():
                            attribute_counter[attribute][value][label] += 1
                        else:
                            attribute_counter[attribute][value][label] = 1
                    else:
                        attribute_counter[attribute][value] = {}
                        attribute_counter[attribute][value][label] = 1
                else:
                    attribute_counter[attribute] = {}
                    attribute_counter[attribute][value] = {}
                    attribute_counter[attribute][value][label] = 1

        for attribute, choices in attribute_counter.items():
            attr_total_choice = len(choices.values())
            for choice, labels in choices.items():
                for label in self.labels:
                    count = 0
                    if label in labels:
                        count = labels[label]
                    numerator = count + smoothing_term
                    denominator = self.label_counter[label] + (attr_total_choice * smoothing_term)
                    probability = numerator/denominator
                    log_prob = math.log(probability)
                    attribute_counter[attribute][choice][label] = log_prob

        self.attribute_probabilities = attribute_counter

    def predict_label_v1(self, data):
        output = []
        for example in data:
            label_prediction = '-1000'
            max_prob = -float("Inf")
            for label in self.labels:
                prob = self.raw_prior[label]
                for attribute in self.attribute_set.keys():
                    if attribute in example:
                        value = example[attribute]
                    else:
                        value = '0'
                    try:
                        additive = self.attribute_probabilities[attribute][value][label]
                    except KeyError:
                        additive = 0
                    prob += additive
                if prob >= max_prob:
                    max_prob = prob
                    label_prediction = label
            output.append(int(label_prediction))
        return output

    def predict_label(self, data):
        output = []
        for example in data:
            label_prob_counters = {}
            for label in self.labels:
                label_prob_counters[label] = self.raw_prior[label]

            for attribute in self.attribute_set.keys():
                if attribute in example:
                    value = example[attribute]
                else:
                    value = '0'
                for label in self.labels:
                    try:
                        additive = self.attribute_probabilities[attribute][value][label]
                    except KeyError:
                        additive = 0
                    label_prob_counters[label] += additive
            max = -float("Inf")
            label = "-1000"
            for key, value in label_prob_counters.items():
                if value > max:
                    max = value
                    label = key
            output.append(int(label))
        return output

    def get_accuracy(self, data):
        output = self.predict_label(data)
        mistakes = 0
        for index, prediction in enumerate(output):
            if prediction != data[index]["label"]:
                mistakes += 1

        return (1 - (mistakes/len(data))) * 100

    def get_classifier_stats(self, data):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        output = self.predict_label(data)
        mistakes = 0
        for index, prediction in enumerate(output):
            if prediction != data[index]["label"]:
                mistakes += 1
        print("mistakes made " + str(mistakes))
        accuracy = (1 - (mistakes / len(data))) * 100
        for index, prediction in enumerate(output):
            if int(prediction) == 1 and int(data[index]["label"]) == 1:
                true_positive += 1
            elif int(prediction) == 1 and int(data[index]["label"]) == 0:
                false_positive += 1
            elif int(prediction) == 0 and int(data[index]["label"]) == 1 :
                false_negative += 1

        precision = true_positive
        if precision > 0:
            precision /= (true_positive + false_positive)

        recall = true_positive
        if recall > 0:
            recall /= (true_positive + false_negative)

        f1 = 2 * precision * recall
        if f1 > 0:
            f1 /= (precision + recall)

        return f1, precision, recall, accuracy


dataTrain = dtP.DataParser("movie-ratings/data-splits/data.train")
dataTest = dtP.DataParser("movie-ratings/data-splits/data.test")
dataEval = dtP.DataParser("movie-ratings/data-splits/data.eval.anon")

max_f1 = 0
max_f1_accuracy = 0
max_f1_recall = 0
max_f1_precision = 0

max_f1_smoothing = 1

# folds = dataTrain.create_cross_fold()
#
# data_sets = [
#     folds[1] + folds[2] + folds[3] + folds[4],
#     folds[0] + folds[2] + folds[3] + folds[4],
#     folds[0] + folds[1] + folds[3] + folds[4],
#     folds[0] + folds[1] + folds[2] + folds[4],
#     folds[0] + folds[1] + folds[2] + folds[3]
# ]
#
# smoothing_term_set = [2, 1.5, 1, 0.5]
#
# for smoothing_term in smoothing_term_set:
#     print("smoothing Term = " + str(smoothing_term))
#     f1_set = []
#     recall_set = []
#     precision_set = []
#     accuracies_set = []
#     for test_set_num, data_set in enumerate(data_sets):
#         nb = NaiveBayes(data_set, dataTrain.max_variable)
#         nb.train_model(smoothing_term)
#
#         f1, precision, recall = nb.get_classifier_stats(folds[test_set_num].raw_data)
#
#         f1_set.append(f1)
#         precision_set.append(precision)
#         recall_set.append(recall)
#         f1_set.append(f1)
#         accuracies_set.append(nb.get_accuracy(folds[test_set_num].raw_data))
#
#     average_f1 = avg(f1_set)
#     average_precision = avg(precision_set)
#     average_recall = avg(recall_set)
#     average_accuracy = avg(accuracies_set)
#
#     print("\t\tAverage: F1 = " + str(average_f1) + ", Precision = " + str(average_precision) + ", Recall = "
#           + str(average_recall) + ", Accuracy = " + str(average_accuracy))
#
#     if average_f1 > max_f1:
#         max_f1_smoothing = smoothing_term
#         max_f1 = average_f1
#         max_f1_accuracy = average_accuracy
#         max_f1_precision = average_precision
#         max_f1_recall = average_recall

print("Naive bayes Start Time", time.asctime())

naive_bayes = NaiveBayes(dataTrain.raw_data, dataTrain.max_variable)
naive_bayes.train_model(max_f1_smoothing)
f1, precision, recall, accuracy = naive_bayes.get_classifier_stats(dataTest.raw_data)
print("\nStats:\nSmoothing Term = " + str(max_f1_smoothing) + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision)
      + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + str(accuracy))

results = naive_bayes.predict_label(dataEval.raw_data)
print("Naive bayes Stop Time", time.asctime())

resultFile = open("results.txt", "a")
resultFile.write("Naive Bayes smoothing: " + str(max_f1_smoothing) + "\n")
resultFile.write(str(results) + "\n")
