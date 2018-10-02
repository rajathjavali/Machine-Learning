import helper as helper
import readData as readData
import marginPerceptron as mP
import simplePerceptron as sP
import averagePerceptron as aP
import decayLearningPerceptron as dLP
import aggressiveMarginPerceptron as aMP
# import matplotlib.pyplot as plt


def avg (vector):
    vector_sum = 0
    for i in vector:
        vector_sum += i
    return float(vector_sum)/len(vector)


learning_rate_set = [1, 0.1, 0.01]
margin_value_set = [1, 0.1, 0.01]

train_data_set = readData.ReadData("dataset/diabetes.train")
dev_data = readData.ReadData("dataset/diabetes.dev")
test_data_set = readData.ReadData("dataset/diabetes.test")

folds = [readData.ReadData("dataset/CVSplits/training00.data"),
         readData.ReadData("dataset/CVSplits/training01.data"),
         readData.ReadData("dataset/CVSplits/training02.data"),
         readData.ReadData("dataset/CVSplits/training03.data"),
         readData.ReadData("dataset/CVSplits/training04.data")]

data_sets = [
    folds[1].raw_data + folds[2].raw_data + folds[3].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[2].raw_data + folds[3].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[1].raw_data + folds[3].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[1].raw_data + folds[2].raw_data + folds[4].raw_data,
    folds[0].raw_data + folds[1].raw_data + folds[2].raw_data + folds[3].raw_data
]

epoch = 10
plot = False


def plot_graph(accuracy):
    print("Plotting Graph")
    # x = list(range(0, 20))
    # x = [i for i in range(1, 21)]
    # plt.plot(range(1, 21), accuracy)
    # plt.ylabel('epoch accuracy')
    # plt.xlabel('epoch')
    # plt.show()


def basic_working_test():
    seperable_data = readData.ReadData("dataset/diabetes.basictest")

    simple_perceptron = sP.SimplePerceptron(seperable_data.raw_data, seperable_data.max_variable)
    weights = simple_perceptron.run_perceptron(epoch, 1)
    print("Simple Perceptron: " + str(helper.model_accuracy(seperable_data.raw_data, weights)))

    margin_perceptron = mP.MarginPerceptron(seperable_data.raw_data, seperable_data.max_variable)
    weights = margin_perceptron.run_perceptron(epoch, 1, 1)
    print("Margin Perceptron: " + str(helper.model_accuracy(seperable_data.raw_data, weights)))

    average_perceptron = aP.AveragePerceptron(seperable_data.raw_data, seperable_data.max_variable)
    weights = average_perceptron.run_perceptron(epoch, 1)
    print("Average Perceptron: " + str(helper.model_accuracy(seperable_data.raw_data, weights)))

    decay_learning_perceptron = dLP.DecayLearningPercepton(seperable_data.raw_data, seperable_data.max_variable)
    weights = decay_learning_perceptron.run_perceptron(epoch, 1)
    print("Decay Learning Perceptron: " + str(helper.model_accuracy(seperable_data.raw_data, weights)))

    aggressive_margin_perceptron = aMP.AggressiveMarginPerceptron(seperable_data.raw_data, seperable_data.max_variable)
    weights = aggressive_margin_perceptron.run_perceptron(epoch, 1)
    print("Aggressive Margin Perceptron: " + str(helper.model_accuracy(seperable_data.raw_data, weights)))


def training_simple_perceptron():
    print("-------------------------------------Simple Perceptron----------------------------------")
    max_accuracy = 0
    max_learning_rate = 0
    for learning_rate in learning_rate_set:
        # print(
        #    "\n--------------------------------Learning Rate: " + str(learning_rate) + "---------------------------\n")

        accuracies = []

        for test_set_num, data_set in enumerate(data_sets):
            simple_perceptron = \
                sP.SimplePerceptron(data_set, train_data_set.max_variable)
            weights = simple_perceptron.run_perceptron(epoch, learning_rate)
            accuracies.append(helper.model_accuracy(folds[test_set_num].raw_data, weights))

        # print(accuracies)
        average = avg(accuracies)
        if average > max_accuracy:
            max_learning_rate = learning_rate
            max_accuracy = average

    # print("\n-----------------------------------------------------------------------------------")
    print("Simple Perceptron Max Accurary: " + str(max_accuracy) + " at learning rate: " + str(max_learning_rate))

    final_epoch = 20
    simple_perceptron = sP.SimplePerceptron(train_data_set.raw_data, train_data_set.max_variable)
    weights, num_updates, epoch_accuracies = simple_perceptron.run_perceptron(final_epoch, max_learning_rate, dev_data.raw_data)
    print(weights)
    if plot:
        plot_graph(epoch_accuracies)

    print("Total Updates on Train set: " + str(num_updates))
    print("Accuracy on Dev set: " + str(helper.model_accuracy(dev_data.raw_data, weights)))
    print("Accuracy on Test set: " + str(helper.model_accuracy(test_data_set.raw_data, weights)))


def training_decayl_perceptron():
    print("-------------------------------------Decay Learning Perceptron--------------------------")
    max_accuracy = 0
    max_learning_rate = 0
    for learning_rate in learning_rate_set:
        # print(
        #    "\n--------------------------------Learning Rate: " + str(learning_rate) + "---------------------------\n")

        accuracies = []

        for test_set_num, data_set in enumerate(data_sets):
            decay_learning_perceptron = \
                dLP.DecayLearningPercepton(data_set, train_data_set.max_variable)
            weights = decay_learning_perceptron.run_perceptron(epoch, learning_rate)
            accuracies.append(helper.model_accuracy(folds[test_set_num].raw_data, weights))

        # print(accuracies)
        average = avg(accuracies)
        if average > max_accuracy:
            max_learning_rate = learning_rate
            max_accuracy = average

    # print("\n-----------------------------------------------------------------------------------")
    print("Decay Learning Perceptron Max Accurary: " + str(max_accuracy) + " at learning rate: " + str(max_learning_rate))

    final_epoch = 20
    decay_learning_perceptron = dLP.DecayLearningPercepton(train_data_set.raw_data,
                                                                     train_data_set.max_variable)
    weights, num_updates, epoch_accuracies = decay_learning_perceptron.run_perceptron(final_epoch, max_learning_rate, dev_data.raw_data)
    print(weights)
    if plot:
        plot_graph(epoch_accuracies)

    print("Total Updates on Train set: " + str(num_updates))
    print("Accuracy on Dev set: " + str(helper.model_accuracy(dev_data.raw_data, weights)))
    print("Accuracy on Test set: " + str(helper.model_accuracy(test_data_set.raw_data, weights)))


def training_margin_perceptron():
    print("-------------------------------------Margin Perceptron---------------------------------")
    max_accuracy = 0
    max_learning_rate = 0
    best_margin_value = 0
    for learning_rate in learning_rate_set:
        for margin_value in margin_value_set:
            # print("\n--------------------------------Learning Rate: "
            # + str(learning_rate) + "---------------------------\n")

            accuracies = []

            for test_set_num, data_set in enumerate(data_sets):
                margin_perceptron = \
                    mP.MarginPerceptron(data_set, train_data_set.max_variable)
                weights = margin_perceptron.run_perceptron(epoch, margin_value, learning_rate)
                accuracies.append(helper.model_accuracy(folds[test_set_num].raw_data, weights))

            # print(accuracies)
            average = avg(accuracies)
            if average > max_accuracy:
                max_learning_rate = learning_rate
                max_accuracy = average
                best_margin_value = margin_value

    # print("\n-----------------------------------------------------------------------------------")
    print("Margin Perceptron Max Accurary: " + str(max_accuracy) + " at learning rate: "
          + str(max_learning_rate) + " and margin: " + str(best_margin_value))

    final_epoch = 20
    margin_perceptron = \
        mP.MarginPerceptron(train_data_set.raw_data, train_data_set.max_variable)

    weights, num_updates, epoch_accuracies = \
        margin_perceptron.run_perceptron(final_epoch, best_margin_value, max_learning_rate, dev_data.raw_data)
    print(weights)
    if plot:
        plot_graph(epoch_accuracies)

    print("Total Updates on Train set: " + str(num_updates))
    print("Accuracy on Dev set: " + str(helper.model_accuracy(dev_data.raw_data, weights)))
    print("Accuracy on Test set: " + str(helper.model_accuracy(test_data_set.raw_data, weights)))


def training_average_perceptron():
    print("-------------------------------------Average Perceptron---------------------------------")
    max_accuracy = 0
    max_learning_rate = 0
    for learning_rate in learning_rate_set:
        # print("\n--------------------------------Learning Rate: "
        # + str(learning_rate) + "---------------------------\n")

        accuracies = []

        for test_set_num, data_set in enumerate(data_sets):
            average_perceptron = \
                aP.AveragePerceptron(data_set, train_data_set.max_variable)
            weights = average_perceptron.run_perceptron(epoch, learning_rate)
            accuracies.append(helper.model_accuracy(folds[test_set_num].raw_data, weights))

        # print(accuracies)
        average = avg(accuracies)
        if average > max_accuracy:
            max_learning_rate = learning_rate
            max_accuracy = average

    # print("\n-----------------------------------------------------------------------------------")
    print("Average Perceptron Max Accurary: " + str(max_accuracy) + " at learning rate: "
          + str(max_learning_rate))

    final_epoch = 20
    average_perceptron = \
        aP.AveragePerceptron(train_data_set.raw_data, train_data_set.max_variable)

    weights, num_updates, epoch_accuracies = \
        average_perceptron.run_perceptron(final_epoch, max_learning_rate, dev_data.raw_data)
    print(weights)
    if plot:
        plot_graph(epoch_accuracies)

    print("Total Updates on Train set: " + str(num_updates))
    print("Accuracy on Dev set: " + str(helper.model_accuracy(dev_data.raw_data, weights)))
    print("Accuracy on Test set: " + str(helper.model_accuracy(test_data_set.raw_data, weights)))


def training_aggressive_margin_perceptron():
    print("------------------------------Aggressive Margin Perceptron---------------------------------")
    max_accuracy = 0
    best_margin_value = 0
    for margin_value in margin_value_set:
        # print("\n--------------------------------Learning Rate: "
        # + str(learning_rate) + "---------------------------\n")

        accuracies = []

        for test_set_num, data_set in enumerate(data_sets):
            aggressive_margin_perceptron = \
                aMP.AggressiveMarginPerceptron(data_set, train_data_set.max_variable)
            weights = aggressive_margin_perceptron.run_perceptron(epoch, margin_value)
            accuracies.append(helper.model_accuracy(folds[test_set_num].raw_data, weights))

        # print(accuracies)
        average = avg(accuracies)
        if average > max_accuracy:
            max_accuracy = average
            best_margin_value = margin_value

    # print("\n-----------------------------------------------------------------------------------")
    print("Aggressive Margin Perceptron Max Accurary: " + str(max_accuracy) + " at margin: " + str(best_margin_value))

    final_epoch = 20
    aggressive_margin_perceptron = \
        aMP.AggressiveMarginPerceptron(train_data_set.raw_data, train_data_set.max_variable)

    weights, num_updates, epoch_accuracies = \
        aggressive_margin_perceptron.run_perceptron(final_epoch, best_margin_value, dev_data.raw_data)

    print(weights)
    if plot:
        plot_graph(epoch_accuracies)

    print("Total Updates on Train set: " + str(num_updates))
    print("Accuracy on Dev set: " + str(helper.model_accuracy(dev_data.raw_data, weights)))
    print("Accuracy on Test set: " + str(helper.model_accuracy(test_data_set.raw_data, weights)))


def majority_classifier():
    train_label_dict = {}
    for line in train_data_set.raw_data:
        label = line["label"]
        if label in train_label_dict:
            train_label_dict[label] += 1
        else:
            train_label_dict[label] = 1
    max_label = ' '
    max_value = 0
    print("--------------------------------Majority Classifier---------------------------------")
    for k, v in train_label_dict.items():
        # print("label: " + k + " value: " + str(v))
        if max_value < v:
            max_value = v
            max_label = k
    print("Majority classifier prediction: " + max_label)
    dev_accuracy = 0
    for line in dev_data.raw_data:
        label = line["label"]
        if label == max_label:
            dev_accuracy += 1
    dev_accuracy /= len(dev_data.raw_data)
    print("Dev Accuracy with Majority classifier: " + str(dev_accuracy))

    test_accuracy = 0
    for line in test_data_set.raw_data:
        label = line["label"]
        if label == max_label:
            test_accuracy += 1

    test_accuracy /= len(test_data_set.raw_data)
    print("Test Accuracy with Majority classifier: " + str(test_accuracy))


majority_classifier()
training_simple_perceptron()
training_decayl_perceptron()
training_margin_perceptron()
training_average_perceptron()
training_aggressive_margin_perceptron()

# basic_working_test()
