import random
import time

import Project.dataParser as dtP
import Project.perceptron as aP
import Project.svm as SVM


def feature_transformation_averaged_perceptron(test_set, data_set, eval_set, num_treesperceptron, max_variable, learning_rate):
    data_length = len(data_set)
    ten_percent_data = int(0.3 * data_length + 1)
    bagged_perceptron_weights = []
    # this loop is needed to sample 10% of examples with repetition
    # this loop helps create 200 decision trees
    for i in range(num_treesperceptron):
        print("\t\t perceptron " + str(i+1))
        mini_training_data = []
        for dataLines in range(ten_percent_data):
            index = random.randint(0, data_length - 1)
            try:
                mini_training_data.append(data_set[index])
            except IndexError:
                print(index)

        avg_perceptron = aP.Perceptron(mini_training_data, max_variable)
        weight = avg_perceptron.run_perceptron(10, learning_rate, None)

        bagged_perceptron_weights.append(weight)

    # Feature transformation of training set using the decision trees
    train_output_perceptron = []
    for weight in bagged_perceptron_weights:
        train_output_perceptron.append(aP.get_labels(data_set, weight))

    svm_training_data_set = []
    for index in range(len(data_set)):
        example_list = {"label": data_set[index]["label"]}
        for number, line in enumerate(train_output_perceptron):
            example_list[str(number + 1)] = line[index]
        svm_training_data_set.append(example_list)

    # Feature transformation of testing set using the decision trees
    test_output_perceptron = []

    for weight in bagged_perceptron_weights:
        test_output_perceptron.append(aP.get_labels(test_set, weight))

    svm_testing_data_set = []
    for index in range(len(test_set)):
        example_list = {"label": test_set[index]["label"]}
        for number, line in enumerate(test_output_perceptron):
            example_list[str(number + 1)] = line[index]
        svm_testing_data_set.append(example_list)

    # Feature transformation of training set using the decision trees
    if eval_set is None:
        return svm_testing_data_set, svm_training_data_set, None

    eval_output_perceptron = []
    for weight in bagged_perceptron_weights:
        eval_output_perceptron.append(aP.get_labels(eval_set, weight))

    svm_eval_data_set = []
    for index in range(len(eval_set)):
        example_list = {"label": eval_set[index]["label"]}
        for number, line in enumerate(eval_output_perceptron):
            example_list[str(number + 1)] = line[index]
        svm_eval_data_set.append(example_list)
    return svm_testing_data_set, svm_training_data_set, svm_eval_data_set


dataTrain = dtP.DataParser("movie-ratings/data-splits/data.train")
dataTest = dtP.DataParser("movie-ratings/data-splits/data.test")
dataEval = dtP.DataParser("movie-ratings/data-splits/data.eval.anon")

max_f1 = 0
max_f1_accuracy = 0
max_f1_recall = 0
max_f1_precision = 0
max_learning_rate = 0.01
max_perceptron_learning = 0.01

learning_rate_test_set = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

cross_fold_data_set = []
cross_fold_test_set = []
max_variable = dataTrain.max_variable

cross_validation_perceptron = 10

folds = dataTrain.create_cross_fold()

data_sets = [
    folds[1] + folds[2] + folds[3] + folds[4],
    folds[0] + folds[2] + folds[3] + folds[4],
    folds[0] + folds[1] + folds[3] + folds[4],
    folds[0] + folds[1] + folds[2] + folds[4],
    folds[0] + folds[1] + folds[2] + folds[3]
]

for lr in learning_rate_test_set:
    for test_set_num, data_set in enumerate(data_sets):
        print("cross fold perceptron build start Time", time.asctime())
        test_set = folds[test_set_num]

        svm_testing_set, svm_training_set, _ = \
            feature_transformation_averaged_perceptron(test_set, data_set, None, cross_validation_perceptron,
                                                       max_variable, lr)

        cross_fold_data_set.append(svm_training_set)
        cross_fold_test_set.append(svm_testing_set)
        print("cross fold perceptron build end Time", time.asctime())

        print("learning rate = " + str(lr))

        f1_set = []
        recall_set = []
        precision_set = []
        accuracies_set = []
        for index, data_set in enumerate(cross_fold_data_set):
            avg_perceptron = aP.Perceptron(data_set, max_variable)
            weights = avg_perceptron.run_perceptron(10, lr, None)

            f1, precision, recall = aP.get_classifier_stats(cross_fold_test_set[index], weights)
            f1_set.append(f1)
            precision_set.append(precision)
            recall_set.append(recall)
            f1_set.append(f1)
            accuracies_set.append(aP.model_accuracy(cross_fold_test_set[index], weights))

        average_f1 = aP.avg(f1_set)
        average_precision = aP.avg(precision_set)
        average_recall = SVM.avg(recall_set)
        average_accuracy = SVM.avg(accuracies_set)

        print("\t\tAverage: F1 = " + str(average_f1) + ", Precision = " + str(average_precision) + ", Recall = "
              + str(average_recall) + ", Accuracy = " + str(average_accuracy))

        if average_f1 > max_f1:
            max_f1 = average_f1
            max_learning_rate = lr
            max_f1_precision = average_precision
            max_f1_recall = average_recall
            max_f1_accuracy = average_accuracy


final_bagged_perceptron_count = 200
final_testing_set, final_training_set, final_eval_set = \
            feature_transformation_averaged_perceptron(dataTest.raw_data, dataTrain.raw_data, dataEval.raw_data,
                                                       final_bagged_perceptron_count, max_variable,
                                                       max_perceptron_learning)

avg_perceptron = aP.Perceptron(final_training_set, final_bagged_perceptron_count)
weights = avg_perceptron.run_perceptron(10, max_learning_rate, None)

f1, precision, recall = aP.get_classifier_stats(final_testing_set, weights)
print("\nStats:\nlearning rate = " + str(max_learning_rate)
      + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + str(str(SVM.model_accuracy(final_testing_set, weights))))
resultFile = open("results.txt", "a")
results = aP.get_labels(final_eval_set, weights)
resultFile.write("Bagged avg. perceptron learning rate: " + str(max_learning_rate) + "\n")
resultFile.write(str(results) + "\n")
