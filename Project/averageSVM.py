import time
import random
from Project import helper
from Project import dataParser as dtP


def get_labels(data, weights):
    output = []
    for line in data:
        WX = helper.vector_dict_multiply(weights, line)
        if WX >= 0:
            output.append(1)
        else:
            output.append(-1)
    return output


def count_mistakes(data, weights):
    mistakes = 0
    if len(weights) == 0:
        print("len of weight 0")
        return 0
    for line in data:
        WX = helper.vector_dict_multiply(weights, line)
        label = int(line["label"])
        if label == 0:
            label = -1
        if WX * label < 0:
            mistakes += 1

    return mistakes


def get_classifier_stats(data, weights):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    if len(weights) == 0:
        print("len of weights 0")
        return 0, 0, 0

    output = get_labels(data, weights)
    for index, line in enumerate(data):
        label = int(line["label"])
        if label == 0:
            label = -1

        if output[index] == 1 and label == 1:
            true_positive += 1
        elif output[index] == 1 and label == -1:
            false_positive += 1
        elif output[index] == -1 and label == 1:
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


def model_accuracy(data, weights):
    return (1 - (count_mistakes(data, weights)/len(data))) * 100


def avg(vector):
    vector_sum = 0
    for i in vector:
        vector_sum += i
    return float(vector_sum)/len(vector)


class AverageSVM:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = helper.random_weight_vector(max_variable)
        self.average_weights = [0] * (max_variable + 1)
        self.cached_weights = [0] * (max_variable + 1)
        self.final_weight_vector = []

    def average_svm(self, data, learning_rate, seed, balancer_c, counter):
        num_updates = 0
        random_data = helper.data_randomizer(data, seed)
        one_minus_learning_rate = 1 - learning_rate

        multiplier = balancer_c * learning_rate

        for index in random_data:
            counter += 1
            line = data[index]
            WX = helper.vector_dict_multiply(self.weights, line)
            label = int(line["label"])
            if label == 0:
                label = -1

            # self.weights = list(map(lambda weight: one_minus_learning_rate * weight, self.weights))
            self.weights = [one_minus_learning_rate * weight for weight in self.weights]
            self.cached_weights = [one_minus_learning_rate * weight for weight in self.weights]

            # checking whether the prediction made is correct or not
            if WX * label <= 1:
                for key, value in line.items():
                    if key == "label":
                        temp_value = multiplier * label
                        self.weights[0] += temp_value
                        self.cached_weights[0] += temp_value * counter
                    else:
                        temp_value = multiplier * label * float(value)
                        self.weights[int(key)] += temp_value
                        self.cached_weights[int(key)] += temp_value * counter
                num_updates += 1

        return num_updates

    def run_svm(self, epoch, learning_rate, balancer_c):
        num_updates = []
        counter = 0
        for i in range(0, epoch):
            print("\t\t\t\t\tepoch Start Time", time.asctime())
            epoch_learning_rate = learning_rate / (1 + (i * learning_rate / balancer_c))
            num_updates.append(self.average_svm(self.data, epoch_learning_rate, i+1, balancer_c, counter))
            print("\t\t\t\t\tepoch End Time", time.asctime())

        self.final_weight_vector = vector_subtraction(self.weights, const_div_vector(self.cached_weights, counter))
        return self.final_weight_vector

    def get_final_weight_vector(self):
        return self.final_weight_vector


def const_div_vector(base_vector, factor):
    new_vector = []
    for i in range(len(base_vector)):
        new_vector.append(base_vector[i] / factor)
    return new_vector


def vector_subtraction(vector_a, vector_b):
    new_vector = []
    for i in range(len(vector_a)):
        new_vector.append(vector_a[i] - vector_b[i])
    return new_vector


def cross_validation():
    dataTrain = dtP.DataParser("movie-ratings/data-splits/data.train")
    dataTest = dtP.DataParser("movie-ratings/data-splits/data.test")
    dataEval = dtP.DataParser("movie-ratings/data-splits/data.eval.anon")

    max_balancer = 1e3
    max_learning_rate = 1e-06

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
    # max_f1 = 0
    # max_f1_accuracy = 0
    # max_f1_recall = 0
    # max_f1_precision = 0
    #
    # learning_rate_test_set = [0.00001, 0.000001, 0.0000001]
    # balancer_test_set = [10000, 1000, 100]
    #
    # print("Starting svm:\n")
    # for lr in learning_rate_test_set:
    #     print("learning rate = " + str(lr))
    #     for balancer_c in balancer_test_set:
    #         print("\tbalancer = " + str(balancer_c))
    #         f1_set = []
    #         recall_set = []
    #         precision_set = []
    #         accuracies_set = []
    #         for test_set_num, data_set in enumerate(data_sets):
    #             print("\t\tcross validation = " + str(test_set_num))
    #             split_svm = \
    #                 Svm(data_set, dataTrain.max_variable)
    #             weights = split_svm.run_svm(2, lr, balancer_c)
    #             f1, precision, recall = get_classifier_stats(folds[test_set_num], weights)
    #             f1_set.append(f1)
    #             precision_set.append(precision)
    #             recall_set.append(recall)
    #             f1_set.append(f1)
    #             accuracy = model_accuracy(folds[test_set_num], weights)
    #             accuracies_set.append(accuracy)
    #             print("\t\t\taccuracy = " + str(accuracy) + " F1 = " + str(f1))
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
    #             max_balancer = balancer_c
    #             max_f1 = average_f1
    #             max_f1_accuracy = average_accuracy
    #             max_f1_precision = average_precision
    #             max_f1_recall = average_recall
    #         elif average_f1 < (max_f1/3):
    #             break

    print("Starting SVM: learning rate: " + str(max_learning_rate) + " Load Balancer: " + str(max_balancer))
    svm = AverageSVM(dataTrain.raw_data, dataTrain.max_variable)
    weights = svm.run_svm(10, max_learning_rate, max_balancer)
    accuracy = model_accuracy(dataTest.raw_data, weights)

    f1, precision, recall = get_classifier_stats(dataTest.raw_data, weights)
    print("\t\tAverage: F1 = " + str(f1) + ", Precision = " + str(precision) + ", Recall = "
          + str(recall) + ", Accuracy = " + str(accuracy))

    results = get_labels(dataEval.raw_data, weights)

    resultFile = open("results.txt", "a")
    resultFile.write("SVM learning rate: " + str(max_learning_rate) + " Balancer: " + str(max_balancer) + " Accuracy: "
                     + str(accuracy) + "\n")
    resultFile.write(str(results) + "\n")


cross_validation()
