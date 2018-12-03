import math

import Perceptron.helper as helper


def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))


def get_predictions(data, weights):
    output = []
    if len(weights) != 0:
        for line in data:
            WX = helper.vector_dict_multiply(weights, line)
            try:
                exponent_term = math.exp(-WX)
                factor = 1.0 / (1 + exponent_term)
            except OverflowError:
                factor = 0
            if factor <= 0.5:
                pred = -1
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
        for line in data:
            label = int(line["label"])
            if label == 0:
                label = -1

            WX = helper.vector_dict_multiply(weights, line)
            try:
                exponent_term = math.exp(-WX)
                factor = 1.0 / (1 + exponent_term)
            except OverflowError:
                factor = 0

            if factor > 0.5 and label == 1:
                true_positive += 1
            elif factor > 0.5 and label == -1:
                false_positive += 1
            elif factor <= 0.5 and label == 1:
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


class LogisticRegression:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = helper.random_weight_vector(max_variable)

    def logistic_regression(self, data, learning_rate, seed, sigma):
        num_updates = 0
        random_data = helper.data_randomizer(data, seed)
        for index in random_data:
            line = data[index]
            WX = helper.vector_dict_multiply(self.weights, line)
            label = int(line["label"])

            # checking whether the prediction made is correct or not
            if WX * label <= 0:
                num_updates += 1
                # modifying weights on wrong prediction
                exponent_term = math.exp(label * WX)
                for i in range(len(self.weights)):
                    if i == 0:
                        self.weights[0] += learning_rate * label / (1 + math.exp(label))
                    elif str(i) in line:
                        self.weights[i] += learning_rate * label * float(line[str(i)]) / (1 + exponent_term) \
                                           - 2 * learning_rate * self.weights[i] / sigma

        return num_updates

    def run_regression(self, epoch, learning_rate, sigma):
        num_updates = []
        for i in range(0, epoch):
            num_updates.append(self.logistic_regression(self.data, learning_rate, i+1, sigma))

        return self.weights
