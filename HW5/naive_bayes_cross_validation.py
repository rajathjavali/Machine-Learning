# import time
import HW5.dataParser as dtP
import HW5.naiveBayes as NB

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

max_f1_smoothing = 0

smoothing_term_set = [2, 1.5, 1, 0.5]

for smoothing_term in smoothing_term_set:
    print("smoothing Term = " + str(smoothing_term))
    f1_set = []
    recall_set = []
    precision_set = []
    accuracies_set = []
    for test_set_num, data_set in enumerate(data_sets):
        nb = NB.NaiveBayes(data_set, dataTrain.max_variable)
        nb.train_model(smoothing_term)

        f1, precision, recall = nb.get_classifier_stats(folds[test_set_num].raw_data)

        f1_set.append(f1)
        precision_set.append(precision)
        recall_set.append(recall)
        f1_set.append(f1)
        accuracies_set.append(nb.get_accuracy(folds[test_set_num].raw_data))

    average_f1 = NB.avg(f1_set)
    average_precision = NB.avg(precision_set)
    average_recall = NB.avg(recall_set)
    average_accuracy = NB.avg(accuracies_set)

    print("\t\tAverage: F1 = " + str(average_f1) + ", Precision = " + str(average_precision) + ", Recall = "
          + str(average_recall) + ", Accuracy = " + str(average_accuracy))

    if average_f1 > max_f1:
        max_f1_smoothing = smoothing_term
        max_f1 = average_f1
        max_f1_accuracy = average_accuracy
        max_f1_precision = average_precision
        max_f1_recall = average_recall

naive_bayes = NB.NaiveBayes(dataTrain.raw_data, dataTrain.max_variable)
naive_bayes.train_model(max_f1_smoothing)
f1, precision, recall = naive_bayes.get_classifier_stats(dataTest.raw_data)
print("\nStats:\nSmoothing Term = " + str(max_f1_smoothing) + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision)
      + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + str(str(naive_bayes.get_accuracy(dataTest.raw_data))))
