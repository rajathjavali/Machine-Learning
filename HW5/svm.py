import time
import HW5.dataParser as dtP
import Perceptron.helper as helper


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
            if WX < 0:
                pred = 0
            else:
                pred = 1
            output.append(pred)

    return output


def model_accuracy(data, weights):
    return (1 - (count_mistakes(data, weights)/len(data))) * 100


def get_svm_counts(data, weights):
    mistakes = 0
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
            else:
                false_negative += 1
            if WX * label <= 0:
                mistakes += 1

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
        for index in random_data:
            line = data[index]
            WX = helper.vector_dict_multiply(self.weights, line)
            label = int(line["label"])
            one_minus_learning_rate = 1 - learning_rate

            # checking whether the prediction made is correct or not
            if WX * label <= 1:
                num_updates += 1
                # modifying weights on wrong prediction
                for i in range(len(self.weights)):
                    if i == 0:
                        self.weights[0] = self.weights[0] * one_minus_learning_rate + \
                                          balancer_c * learning_rate * label
                    elif str(i) in line:
                        self.weights[i] = self.weights[i] * one_minus_learning_rate + \
                                          balancer_c * learning_rate * label * float(line[str(i)])
            else:
                for i in range(len(self.weights)):
                    if i == 0:
                        self.weights[0] *= one_minus_learning_rate

                    elif str(i) in line:
                        self.weights[i] *= one_minus_learning_rate

        return num_updates

    def run_svm(self, epoch, learning_rate, balancer_c):
        num_updates = []
        for i in range(0, epoch):
            epoch_learning_rate = learning_rate / (1 + i)
            num_updates.append(self.basic_svm(self.data, epoch_learning_rate, i+1, balancer_c))

        return self.weights


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

f1_set = []
max_f1 = 0
max_balancer = 1
max_learning_rate = 1

# for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
#     print("Starting svm:\nlearning rate = " + str(lr))
#     for balancer_c in [10, 1, 0.1, 0.01, 0.001, 0.0001]:
#         print("\tbalancer = " + str(balancer_c))
#         for test_set_num, data_set in enumerate(data_sets):
#             print("\t\tcross validation = " + str(test_set_num))
#             split_svm = \
#                 Svm(data_set, dataTrain.max_variable)
#             weights = split_svm.run_svm(10, lr, balancer_c)
#             f1, _, _ = get_svm_counts(folds[test_set_num].raw_data, weights)
#             f1_set.append(f1)
#             print("\t\t\taccuracy = " + str(model_accuracy(folds[test_set_num].raw_data, weights))
#                   + " F1 = " + str(f1))
#
#         average = avg(f1_set)
#         if average > max_f1:
#             max_learning_rate = lr
#             max_balancer = balancer_c
#             max_f1 = average
#         elif average < (max_f1/3):
#             break

svm = Svm(dataTrain.raw_data, dataTrain.max_variable)
weights = svm.run_svm(10, max_learning_rate, max_balancer)
f1, precision, recall = get_svm_counts(dataTest.raw_data, weights)
print("\nStats:\nBalancer = " + str(max_balancer) + "\nlearning rate = " + str(max_learning_rate)
      + "\nF1 = " + str(f1) + "\nPrecision = " + str(precision) + "\nRecall = " + str(recall)
      + "\nAccuracy test data = " + str(str(model_accuracy(dataTest.raw_data, weights))))

# for lr in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
#     print("Starting svm: learning rate:" + str(lr))
#     print("Start Time", time.asctime())
#     for balancer_c in [1, 0.1, 0.01, 0.001, 0.0001]:
#         weights = svm.run_svm(10, lr, balancer_c)
#         print("model accuracy with balancer = " + str(balancer_c) + " is = " +
#               str(model_accuracy(dataTest.raw_data, weights)))
#
#     print("End Time", time.asctime())
