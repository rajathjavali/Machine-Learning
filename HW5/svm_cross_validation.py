# import time
import HW5.dataParser as dtP
import HW5.svm as SVM

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
max_balancer = 0
max_learning_rate = 0

learning_rate_test_set = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
balancer_test_set = [10, 1, 0.1, 0.01, 0.001, 0.0001]

# learning_rate_test_set = [1]
# balancer_test_set = [10, 1]
#
print("Starting svm:\n")
for lr in learning_rate_test_set:
    print("learning rate = " + str(lr))
    for balancer_c in balancer_test_set:
        print("\tbalancer = " + str(balancer_c))
        f1_set = []
        recall_set = []
        precision_set = []
        accuracies_set = []
        for test_set_num, data_set in enumerate(data_sets):
            # print("\t\tcross validation = " + str(test_set_num))
            split_svm = \
                SVM.Svm(data_set, dataTrain.max_variable)
            weights = split_svm.run_svm(10, lr, balancer_c)
            f1, precision, recall = SVM.get_classifier_stats(folds[test_set_num].raw_data, weights)
            f1_set.append(f1)
            precision_set.append(precision)
            recall_set.append(recall)
            f1_set.append(f1)
            accuracies_set.append(SVM.model_accuracy(folds[test_set_num].raw_data, weights))
            # print("\t\t\taccuracy = " + str() + " F1 = " + str(f1))

        average_f1 = SVM.avg(f1_set)
        average_precision = SVM.avg(precision_set)
        average_recall = SVM.avg(recall_set)
        average_accuracy = SVM.avg(accuracies_set)

        print("\t\tAverage: F1 = " + str(average_f1) + ", Precision = " + str(average_precision) + ", Recall = "
              + str(average_recall) + ", Accuracy = " + str(average_accuracy))

        if average_f1 > max_f1:
            max_learning_rate = lr
            max_balancer = balancer_c
            max_f1 = average_f1
            max_f1_accuracy = average_accuracy
            max_f1_precision = average_precision
            max_f1_recall = average_recall
        elif average_f1 < (max_f1/3):
            break

svm = SVM.Svm(dataTrain.raw_data, dataTrain.max_variable)
weights = svm.run_svm(10, max_learning_rate, max_balancer)
f1, precision, recall = SVM.get_classifier_stats(dataTest.raw_data, weights)
print("\nStats:\nBalancer = " + str(max_balancer) + "\nlearning rate = " + str(max_learning_rate)
      + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + str(str(SVM.model_accuracy(dataTest.raw_data, weights))))

# for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
#     print("Starting svm: learning rate:" + str(lr))
#     print("Start Time", time.asctime())
#     for balancer_c in [1, 0.1, 0.01, 0.001, 0.0001]:
#         weights = svm.run_svm(10, lr, balancer_c)
#         print("model accuracy with balancer = " + str(balancer_c) + " is = " +
#               str(model_accuracy(dataTest.raw_data, weights)))
#
#     print("End Time", time.asctime())
