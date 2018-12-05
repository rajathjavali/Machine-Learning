# import time
import random

import HW5.dataParser as dtP
import HW5.decisionTree as dT
import HW5.svm as SVM


def feature_transformation_decision_trees(test_set, data_set, num_trees, limiting_depth):
    data_length = len(data_set)
    ten_percent_data = int(0.1 * data_length + 1)
    bagged_forest = []
    # this loop is needed to sample 10% of examples with repetition
    # this loop helps create 200 decision trees
    for i in range(num_trees):
        mini_tree_data = []
        for dataLines in range(ten_percent_data):
            index = random.randint(0, data_length - 1)
            try:
                mini_tree_data.append(data_set[index])
            except IndexError:
                print(index)
        decision_tree = dT.DecisionTree(mini_tree_data)
        decision_tree.initiate(limiting_depth)

        bagged_forest.append(decision_tree)

    # Feature transformation of training set using the decision trees
    train_output_decision_tree = []
    for tree in bagged_forest:
        train_output_decision_tree.append(tree.get_labels(data_set))

    svm_training_data_set = []
    for index in range(len(data_set)):
        example_list = {"label": data_set[index]["label"]}
        for number, line in enumerate(train_output_decision_tree):
            example_list[str(number + 1)] = line[index]
        svm_training_data_set.append(example_list)

    # Feature transformation of testing set using the decision trees
    test_output_decision_tree = []

    for tree in bagged_forest:
        test_output_decision_tree.append(tree.get_labels(test_set))

    svm_testing_data_set = []
    for index in range(len(test_set)):
        example_list = {"label": test_set[index]["label"]}
        for number, line in enumerate(test_output_decision_tree):
            example_list[str(number + 1)] = line[index]
        svm_testing_data_set.append(example_list)

    return svm_testing_data_set, svm_training_data_set


dataTrain = dtP.DataParser("data/train.liblinear")
dataTest = dtP.DataParser("data/test.liblinear")
folds = [dtP.DataParser("data/CVSplits/training00.data"),
         dtP.DataParser("data/CVSplits/training01.data"),
         dtP.DataParser("data/CVSplits/training02.data"),
         dtP.DataParser("data/CVSplits/training03.data"),
         dtP.DataParser("data/CVSplits/training04.data")]

data_sets = [
    folds[1].raw_data + folds[2].raw_data + folds[3].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[2].raw_data + folds[3].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[1].raw_data + folds[3].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[1].raw_data + folds[2].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[1].raw_data + folds[2].raw_data + folds[3].raw_data
]

max_f1 = 0
best_f1_depth = 0
max_f1_accuracy = 0
max_f1_recall = 0
max_f1_precision = 0
max_balancer = 0
max_learning_rate = 0


depth_test_set = [10, 20, 30]
# learning_rate_test_set = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# balancer_test_set = [10, 1, 0.1, 0.01, 0.001, 0.0001]
learning_rate_test_set = [1, 0.1]
balancer_test_set = [10, 1, 0.01]

cross_fold_data_set = []
cross_fold_test_set = []

cross_validation_trees = 20
for limiting_depth in depth_test_set:
    for test_set_num, data_set in enumerate(data_sets):
        test_set = folds[test_set_num].raw_data

        svm_testing_set, svm_training_set = \
            feature_transformation_decision_trees(test_set, data_set, cross_validation_trees, limiting_depth)

        cross_fold_data_set.append(svm_training_set)
        cross_fold_test_set.append(svm_testing_set)

    for lr in learning_rate_test_set:
        print("learning rate = " + str(lr))
        for balancer_c in balancer_test_set:
            print("\tbalancer = " + str(balancer_c))
            f1_set = []
            recall_set = []
            precision_set = []
            accuracies_set = []
            for index, data_set in enumerate(cross_fold_data_set):
                split_svm = SVM.Svm(data_set, cross_validation_trees)
                weights = split_svm.run_svm(10, lr, balancer_c)

                f1, precision, recall = SVM.get_classifier_stats(cross_fold_test_set[index], weights)
                f1_set.append(f1)
                precision_set.append(precision)
                recall_set.append(recall)
                f1_set.append(f1)
                accuracies_set.append(SVM.model_accuracy(cross_fold_test_set[index], weights))

            average_f1 = SVM.avg(f1_set)
            average_precision = SVM.avg(precision_set)
            average_recall = SVM.avg(recall_set)
            average_accuracy = SVM.avg(accuracies_set)

            print("\t\tAverage: F1 = " + str(average_f1) + ", Precision = " + str(average_precision) + ", Recall = "
                  + str(average_recall) + ", Accuracy = " + str(average_accuracy))

            if average_f1 > max_f1:
                max_f1 = average_f1
                max_learning_rate = lr
                max_balancer = balancer_c
                max_f1_precision = average_precision
                max_f1_recall = average_recall
                max_f1_accuracy = average_accuracy
                best_f1_depth = limiting_depth

final_bagged_trees_count = 200
final_svm_testing_set, final_svm_training_set = \
            feature_transformation_decision_trees(dataTest.raw_data, dataTrain.raw_data,
                                                  final_bagged_trees_count, best_f1_depth)

svm = SVM.Svm(final_svm_training_set, final_bagged_trees_count)
weights = svm.run_svm(10, max_learning_rate, max_balancer)
f1, precision, recall = SVM.get_classifier_stats(final_svm_testing_set, weights)
print("\nStats:\nBalancer = " + str(max_balancer) + "\nlearning rate = " + str(max_learning_rate)
      + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + str(str(SVM.model_accuracy(dataTest.raw_data, weights))))
