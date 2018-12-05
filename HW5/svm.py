import Perceptron.helper as helper


def count_mistakes(data, weights):
    mistakes = 0
    if len(weights) != 0:
        for line in data:
            WX = helper.vector_dict_multiply(weights, line)
            label = int(line["label"])
            if WX * label <= 0:
                mistakes += 1

    return mistakes


def get_predictions(data, weights):
    output = []
    if len(weights) != 0:
        for line in data:
            WX = helper.vector_dict_multiply(weights, line)
            if WX < 0:
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

            if WX >= 0 and label == 1:
                true_positive += 1
            elif WX >= 0 and label == -1:
                false_positive += 1
            elif WX < 0 and label == 1:
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
