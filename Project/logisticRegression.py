import math
import random
import time
from Project import dataParser as dtP
from Project import helper


def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))


def get_predictions(data, weights):
    output = []
    if len(weights) != 0:
        for line in data:
            WX = helper.vector_dict_multiply(weights, line)
            if WX < 0:
                pred = 0
            else:
                pred = 1
            output.append(pred)

    return output


def count_mistakes(data, weights):
    mistakes = 0
    if len(weights) != 0:
        output = get_predictions(data, weights)
        for i in range(len(data)):
            label = int(data[i]["label"])
            if label != output[i]:
                mistakes += 1

    return mistakes


def model_accuracy(data, weights):
    return (1 - (count_mistakes(data, weights)/len(data))) * 100


def avg(vector):
    vector_sum = 0
    for i in vector:
        vector_sum += i
    return float(vector_sum)/len(vector)


def get_classifier_stats(data, weights):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    if len(weights) != 0:
        output = get_predictions(data, weights)
        for i, val in enumerate(output):
            if val == 1 and int(data[i]["label"]) == 1:
                true_positive += 1
            elif val == 1 and int(data[i]["label"]) == 0:
                false_positive += 1
            elif val == 0 and int(data[i]["label"]) == 1:
                false_negative += 1
    else:
        print("len of weights is 0\n")

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


class LogisticRegression:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = [0] * (max_variable + 1)
        # self.weights = helper.random_weight_vector(max_variable)

    def logistic_regression(self, data, lr, seed, sigma):
        learning_rate = lr / seed
        # random_data = helper.data_randomizer(data, seed)
        weight_modifier = 1 - (2 * learning_rate / sigma)
        random.seed(seed * 3)
        random.shuffle(data)
        # for index in random_data:
        for line in data:
            # line = data[index]
            WX = helper.vector_dict_multiply(self.weights, line)
            label = int(line["label"])
            if label == 0:
                label = -1

            self.weights = [weight_modifier * weight for weight in self.weights]
            if WX*label <=1:
                try:
                    exponent_term = math.exp(label * WX)
                except OverflowError:
                    exponent_term = float("Inf")

                for key, value in line.items():
                    if key == "label":
                        self.weights[0] += learning_rate * label / (1 + math.exp(label))
                        continue
                    self.weights[int(key)] += learning_rate * label * float(value) / (1 + exponent_term)

    def run_regression(self, epoch, learning_rate, sigma):
        for i in range(0, epoch):
            print("\t\t\t\t\tepoch Start Time", time.asctime())
            self.logistic_regression(self.data, learning_rate, i+1, sigma)
            print("\t\t\t\t\tepoch End Time", time.asctime())

        return self.weights


dataTrain = dtP.DataParser("movie-ratings/data-splits/data.train")
dataTest = dtP.DataParser("movie-ratings/data-splits/data.test")
dataEval = dtP.DataParser("movie-ratings/data-splits/data.eval.anon")


max_sigma = 1e3
max_learning_rate = 1e-01

# max_f1 = 0
# max_f1_accuracy = 0
# max_f1_recall = 0
# max_f1_precision = 0
#
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
# print("Starting Logistic Regression:\n")
# for lr in [1, 0.1, 0.01]:
#     print("learning rate = " + str(lr))
#     for sigma in [1000, 10000, 100000, 1000000, 10000000]:
#         print("\tsigma square = " + str(sigma))
#         f1_set = []
#         recall_set = []
#         precision_set = []
#         accuracies_set = []
#         for test_set_num, data_set in enumerate(data_sets):
#             # print("\t\tcross validation = " + str(test_set_num))
#             # print("\t\t\tStart Time", time.asctime())
#
#             split_logistic_regression = LogisticRegression(data_set, dataTrain.max_variable)
#             weights = split_logistic_regression.run_regression(10, lr, sigma)
#             f1, precision, recall = get_classifier_stats(folds[test_set_num], weights)
#             f1_set.append(f1)
#             precision_set.append(precision)
#             recall_set.append(recall)
#             accuracies_set.append(model_accuracy(folds[test_set_num], weights))
#             # print("\t\t\tStop Time", time.asctime())
#
#         average_f1 = avg(f1_set)
#         average_precision = avg(precision_set)
#         average_recall = avg(recall_set)
#         average_accuracy = avg(accuracies_set)
#
#         print("\t\tAverage: F1 = " + str(average_f1) + ", Precision = " + str(average_precision) + ", Recall = "
#               + str(average_recall) + ", Accuracy = " + str(average_accuracy))
#
#         if average_f1 > max_f1:
#             max_learning_rate = lr
#             max_sigma = sigma
#             max_f1 = average_f1
#             max_f1_accuracy = average_accuracy
#             max_f1_precision = average_precision
#             max_f1_recall = average_recall
resultFile = open("results.txt", "a")
regression = LogisticRegression(dataTrain.raw_data, dataTrain.max_variable)
weights = regression.run_regression(5, max_learning_rate, max_sigma)
f1, precision, recall = get_classifier_stats(dataTest.raw_data, weights)
train_accuracy = str(model_accuracy(dataTrain.raw_data, weights))
print(train_accuracy)
test_accuracy = str(model_accuracy(dataTest.raw_data, weights))

print("\nStats:\nBalancer = " + str(max_sigma) + "\nlearning rate = " + str(max_learning_rate)
      + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + test_accuracy)
results = get_predictions(dataEval.raw_data, weights)
resultFile.write("Logistic Regression learning rate: " + str(max_learning_rate) + " Sigma: " + str(max_sigma)
                 + " accuracy: " + str(test_accuracy) + "\n")
resultFile.write(str(results) + "\n")
