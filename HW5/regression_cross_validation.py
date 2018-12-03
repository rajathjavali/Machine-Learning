# import time
import HW5.dataParser as dtP
import HW5.logisticRegression as lregression

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
max_f1_accuracy = 0
max_f1_recall = 0
max_f1_precision = 0
max_sigma = 1
max_learning_rate = 1

print("Starting Logistic Regression:\n")
for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    print("learning rate = " + str(lr))
    for sigma in [0.1, 1, 10, 100, 1000, 10000]:
        print("\tsigma square = " + str(sigma))
        f1_set = []
        recall_set = []
        precision_set = []
        accuracies_set = []
        for test_set_num, data_set in enumerate(data_sets):
            # print("\t\tcross validation = " + str(test_set_num))
            split_logistic_regression = lregression.LogisticRegression(data_set, dataTrain.max_variable)
            weights = split_logistic_regression.run_regression(10, lr, sigma)
            f1, precision, recall = lregression.get_classifier_stats(folds[test_set_num].raw_data, weights)
            f1_set.append(f1)
            precision_set.append(precision)
            recall_set.append(recall)
            accuracies_set.append(lregression.model_accuracy(folds[test_set_num].raw_data, weights))

        average_f1 = lregression.avg(f1_set)
        average_precision = lregression.avg(precision_set)
        average_recall = lregression.avg(recall_set)
        average_accuracy = lregression.avg(accuracies_set)

        print("\t\tAverage: F1 = " + str(average_f1) + ", Precision = " + str(average_precision) + ", Recall = "
              + str(average_recall) + ", Accuracy = " + str(average_accuracy))

        if average_f1 > max_f1:
            max_learning_rate = lr
            max_sigma = sigma
            max_f1 = average_f1
            max_f1_accuracy = average_accuracy
            max_f1_precision = average_precision
            max_f1_recall = average_recall

regression = lregression.LogisticRegression(dataTrain.raw_data, dataTrain.max_variable)
weights = regression.run_regression(10, max_learning_rate, max_sigma)
f1, precision, recall = lregression.get_classifier_stats(dataTest.raw_data, weights)
print("\nStats:\nBalancer = " + str(max_sigma) + "\nlearning rate = " + str(max_learning_rate)
      + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + str(str(lregression.model_accuracy(dataTest.raw_data, weights))))

# for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
#     print("Starting svm: learning rate:" + str(lr))
#     print("Start Time", time.asctime())
#     for balancer_c in [1, 0.1, 0.01, 0.001, 0.0001]:
#         weights = svm.run_svm(10, lr, balancer_c)
#         print("model accuracy with balancer = " + str(balancer_c) + " is = " +
#               str(model_accuracy(dataTest.raw_data, weights)))
#
#     print("End Time", time.asctime())
