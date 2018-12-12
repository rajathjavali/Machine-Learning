import Perceptron.helper as helper


def get_labels(data, weights):
    output = []
    for line in data:
        WX = helper.vector_dict_multiply(weights, line)
        if WX >= 0:
            output.append(1)
        else:
            output.append(-1)
    return output


def count_mistakes(data, weights):
    mistakes = 0
    if len(weights) == 0:
        print("len of weight 0")
        return 0
    for line in data:
        WX = helper.vector_dict_multiply(weights, line)
        label = int(line["label"])
        if label == 0:
            label = -1
        if WX * label <= 0:
            mistakes += 1

    return mistakes


def get_classifier_stats(data, weights):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    if len(weights) == 0:
        print("len of weights 0")
        return 0, 0, 0

    output = get_labels(data, weights)
    for index, line in enumerate(data):
        label = int(line["label"])
        if label == 0:
            label = -1

        if output[index] == 1 and label == 1:
            true_positive += 1
        elif output[index] == 1 and label == -1:
            false_positive += 1
        elif output[index] == -1 and label == 1:
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


def model_accuracy(data, weights):
    return (1 - (count_mistakes(data, weights)/len(data))) * 100


def avg(vector):
    vector_sum = 0
    for i in vector:
        vector_sum += i
    return float(vector_sum)/len(vector)


class Perceptron:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = helper.random_weight_vector(max_variable)
        self.average_weights = [0] * (max_variable + 1)
        self.cached_weights = [0] * (max_variable + 1)
        self.final_weight_vector = []

    # using average perceptron
    # inputs:
    #   data: data set used to learn the weights using perceptron algorithm
    #   learning_rate: the quantitative measure by which u modify the weight vector on an error
    #   seed: to pseudo randomize the data set used for learning across epochs
    def perceptron(self, data, learning_rate, seed, counter):
        num_updates = 0
        random_data = helper.data_randomizer(data, seed)
        for index in random_data:
            counter += 1
            line = data[index]
            WX = helper.vector_dict_multiply(self.weights, line)
            label = int(line["label"])
            if label == 0:
                label = -1
            # checking whether the prediction made is correct or not
            if WX * label <= 0:
                num_updates += 1
                # modifying weights on wrong prediction
                for key, value in line.items():
                    if key == "label":
                        temp_value = learning_rate * label
                        self.weights[0] += temp_value
                        self.cached_weights[0] += temp_value * counter
                    else:
                        temp_value = learning_rate * label * float(value)
                        self.weights[int(key)] += temp_value
                        self.cached_weights[int(key)] += temp_value * counter
            # self.average_weights = helper.add_vectors(self.average_weights, self.weights)

        return num_updates, counter

    def run_perceptron(self, epoch, learning_rate, test=None):
        num_updates = []
        counter = 0
        for i in range(0, epoch):
            # epoch_learning_rate = learning_rate / (1 + i)
            updates, counter = self.perceptron(self.data, learning_rate, i+1, counter)
            num_updates.append(updates)
            # print(num_updates)
            if test is not None:
                weights = vector_subtraction(self.weights, const_div_vector(self.cached_weights, counter))
                accuracy = model_accuracy(test, weights)
                print("epoch : " + str(i) + " acc: " + str(accuracy))

        self.final_weight_vector = vector_subtraction(self.weights, const_div_vector(self.cached_weights, counter))
        return self.final_weight_vector

    def get_final_weight_vector(self):
        return self.final_weight_vector


def const_div_vector(base_vector, factor):
    new_vector = []
    for i in range(len(base_vector)):
        new_vector.append(base_vector[i] / factor)
    return new_vector


def vector_subtraction(vector_a, vector_b):
    new_vector = []
    for i in range(len(vector_a)):
        new_vector.append(vector_a[i] - vector_b[i])
    return new_vector

