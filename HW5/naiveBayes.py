import math

import Perceptron.helper as helper
import HW5.dataParser as dtP



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

    def predict_label(self, data):
        output = []
        for example in data:
            label_prediction = 'NA'
            max_prob = -float("Inf")
            for label in self.labels:
                prob = self.raw_prior[label]
                for attribute in self.attribute_set.keys():
                    if attribute in example:
                        value = example[attribute]
                    else:
                        value = '0'
                    prob += self.attribute_probabilities[attribute][value][label]
                if prob >= max_prob:
                    max_prob = prob
                    label_prediction = label
            output.append(label_prediction)
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

        for index, prediction in enumerate(output):
            if int(prediction) == 1 and int(data[index]["label"]) == 1:
                true_positive += 1
            elif int(prediction) == 1 and int(data[index]["label"]) == -1:
                false_positive += 1
            elif int(prediction) == -1 and int(data[index]["label"]) == 1:
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

        return f1, precision, recall


dataTrain = dtP.DataParser("data/train.liblinear")
dataTest = dtP.DataParser("data/test.liblinear")
nb = NaiveBayes(dataTrain.raw_data, dataTrain.max_variable)
nb.train_model(1)
print(nb.get_accuracy(dataTest.raw_data))
