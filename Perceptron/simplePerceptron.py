import helper


class SimplePerceptron:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = helper.random_weight_vector(max_variable)

    def simple_perceptron(self, data, learning_rate, seed):
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
                for i in range(len(self.weights)):
                    if i == 0:
                        self.weights[0] += learning_rate * label
                    elif str(i) in line:
                        self.weights[i] += learning_rate * label * float(line[str(i)])

        return num_updates

    def run_perceptron(self, epoch, learning_rate, testing_data=None):
        epoch_accuracies = []
        max_epoch_accuracy = 0
        max_accuracy_weights = []
        num_updates = []
        best_position = 0
        position = 0
        for i in range(0, epoch):
            num_updates.append(self.simple_perceptron(self.data, learning_rate, i+1))
            if testing_data is not None:
                accuracy = helper.model_accuracy(testing_data, self.weights)
                epoch_accuracies.append(accuracy)
                if max_epoch_accuracy < accuracy:
                    best_position = position
                    max_epoch_accuracy = accuracy
                    max_accuracy_weights = list(self.weights)
                position += 1
        if testing_data is not None:
            total_updates = 0
            for i, v in enumerate(num_updates):
                total_updates += v
                if i == best_position:
                    break
            return max_accuracy_weights, total_updates, epoch_accuracies
        return self.weights
