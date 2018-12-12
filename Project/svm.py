import Perceptron.helper as helper
import Project.dataParser as dtP
import numpy as np
import time


def count_mistakes(data, weights):
    mistakes = 0
    if len(weights) != 0:
        for line in data:
            WX = helper.vector_dict_multiply(weights, line)
            label = int(line["label"])
            if label == 0:
                label = -1

            if WX * label <= 0:
                mistakes += 1

    return mistakes


def get_predictions(data, weights):
    output = []
    if len(weights) != 0:
        for line in data:
            WX = helper.vector_dict_multiply(weights, line)
            if WX <= 0:
                pred = -1
            else:
                pred = 1
            output.append(pred)

    return output


def model_accuracy(data, weights):
    return (1 - (count_mistakes(data, weights)/len(data))) * 100


def get_classifier_stats(data, weights):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    if len(weights) != 0:
        for line in data:
            WX = helper.vector_dict_multiply(weights, line)
            label = int(line["label"])
            if label == 0:
                label = -1

            if WX > 0 and label == 1:
                true_positive += 1
            elif WX > 0 and label == -1:
                false_positive += 1
            elif WX <= 0 and label == 1:
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


def avg(vector):
    vector_sum = 0
    for i in vector:
        vector_sum += i
    return float(vector_sum)/len(vector)


class Svm:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = helper.random_weight_vector(max_variable)

    def basic_svm(self, data, learning_rate, seed, balancer_c):
        num_updates = 0
        random_data = helper.data_randomizer(data, seed)
        one_minus_learning_rate = 1 - learning_rate

        multiplier = balancer_c * learning_rate

        for index in random_data:
            line = data[index]
            WX = helper.vector_dict_multiply(self.weights, line)
            label = int(line["label"])
            if label == 0:
                label = -1

            # self.weights = list(map(lambda weight: one_minus_learning_rate * weight, self.weights))
            self.weights = [one_minus_learning_rate * weight for weight in self.weights]

            # checking whether the prediction made is correct or not
            if WX * label <= 1:
                num_updates += 1
                for key, value in line.items():
                    if key == "label":
                        self.weights[0] += multiplier * label
                    else:
                        self.weights[int(key)] += multiplier * label * float(value)

        return num_updates

    def run_svm(self, epoch, learning_rate, balancer_c):
        num_updates = []
        for i in range(0, epoch):
            print("\t\t\t\t\tepoch Start Time", time.asctime())
            epoch_learning_rate = learning_rate / (1 + (i * learning_rate / balancer_c))
            # epoch_learning_rate = learning_rate / (1 + i)
            # print("epoch: " + str(i) + " learning rate: " + str(epoch_learning_rate))
            num_updates.append(self.basic_svm(self.data, epoch_learning_rate, i+1, balancer_c))
            print("\t\t\t\t\tepoch End Time", time.asctime())

        return self.weights


def cross_validation():
    dataTrain = dtP.DataParser("movie-ratings/data-splits/data.train")
    dataTest = dtP.DataParser("movie-ratings/data-splits/data.test")
    dataEval = dtP.DataParser("movie-ratings/data-splits/data.eval.anon")

    max_balancer = 1e6
    max_learning_rate = 1e-07
    # max_balancer = 100
    # max_learning_rate = 0.1

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
    # learning_rate_test_set = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    # balancer_test_set = [10, 1, 0.1, 0.01, 0.001, 0.0001]
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
    #             weights = split_svm.run_svm(10, lr, balancer_c)
    #             f1, precision, recall = get_classifier_stats(folds[test_set_num], weights)
    #             f1_set.append(f1)
    #             precision_set.append(precision)
    #             recall_set.append(recall)
    #             f1_set.append(f1)
    #             accuracies_set.append(model_accuracy(folds[test_set_num], weights))
    #             print("\t\t\taccuracy = " + str() + " F1 = " + str(f1))
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

    resultFile = open("results.txt", "a")

    print("Starting SVM: learning rate: " + str(max_learning_rate) + " Load Balancer: " + str(max_balancer))
    svm = Svm(dataTrain.raw_data, dataTrain.max_variable)
    weights = svm.run_svm(10, max_learning_rate, max_balancer)
    accuracy = model_accuracy(dataTest.raw_data, weights)

    f1, precision, recall = get_classifier_stats(dataTest.raw_data, weights)
    print("\t\tAverage: F1 = " + str(f1) + ", Precision = " + str(precision) + ", Recall = "
          + str(recall) + ", Accuracy = " + str(accuracy))

    results = get_predictions(dataEval.raw_data, weights)
    resultFile.write("SVM learning rate: " + str(max_learning_rate) + " Balancer: " + str(max_balancer) + "\n")
    resultFile.write(str(results) + "\n")


cross_validation()
