from Project import helper


def dict_square(data):
    result = 0
    for key, value in data.items():
        if key == "label":
            continue
        result += float(value) * float(value)
    return result


class AggressivePerceptron:

    def __init__(self, data, max_variable):
        self.data = data
        self.max_variable = max_variable
        self.weights = helper.random_weight_vector(max_variable)

    def perceptron(self, data, margin_value, seed):
        num_updates = 0
        random_data = helper.data_randomizer(data, seed)
        for index in random_data:
            line = data[index]
            WX = helper.vector_dict_multiply(self.weights, line)
            label = int(line["label"])
            if label == 0:
                label = -1
            # checking whether the prediction made is correct or not
            if WX * label <= margin_value:
                num_updates += 1
                learning_rate = float(margin_value - (WX * label)) / (dict_square(line) + 1)
                # modifying weights on wrong prediction
                for key, value in line.items():
                    if key == "label":
                        self.weights[0] += learning_rate * label
                    else:
                        self.weights[int(key)] += learning_rate * label * float(value)

        return num_updates

    def run_perceptron(self, epoch, margin_value):
        num_updates = []
        for i in range(0, epoch):
            num_updates.append(self.perceptron(self.data, margin_value, i+1))

        return self.weights
