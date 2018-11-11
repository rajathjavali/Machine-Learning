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


def model_accuracy(data, weights):
    return (1 - (count_mistakes(data, weights)/len(data))) * 100


class Perceptron:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = helper.random_weight_vector(max_variable)
        self.average_weights = [0] * (max_variable + 1)
        self.cached_weights = [0] * (max_variable + 1)

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

    def run_perceptron(self, epoch, learning_rate, test):
        num_updates = []
        counter = 0
        for i in range(0, epoch):
            updates, counter = self.perceptron(self.data, learning_rate, i+1, counter)
            num_updates.append(updates)
            # print(num_updates)
            weights = vector_subtraction(self.weights, const_div_vector(self.cached_weights, counter))
            print("epoch : " + str(i) + " acc: " + str(model_accuracy(test, weights)))

        return vector_subtraction(self.weights, const_div_vector(self.cached_weights, counter))


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

