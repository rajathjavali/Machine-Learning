import random


def data_randomizer(data):
    # num_data_lines = len(data)
    # random selection of data line without repetition - shuffling of data
    # random_data = random.sample(range(0, num_data_lines), num_data_lines - 1)
    # return random_data
    data_indexes = list(range(0, len(data)))
    random.seed(6)
    random.shuffle(data_indexes)
    return data_indexes


def count_mistakes(data, weights):
    mistakes = 0
    if len(weights) != 0:
        for line in data:
            WX = vector_multiply(weights, line)
            label = int(line["label"])

            if WX * label <= 0:
                mistakes += 1

    return mistakes


def random_weight_vector(max_column):
    weights = []
    # +1 of total variables in done to include bias term within the weight vector
    for i in range(max_column + 1):
        rand_num = (random.random() * 2 - 1) / 100
        weights.append(rand_num)
    return weights


def vector_multiply(weights, data_line):
    # weight vector transpose * input vector
    mul_result = 0
    for key, value in data_line.items():
        if key == "label":
            mul_result += weights[0]
        else:
            mul_result += weights[int(key)] * float(value)
    return mul_result


def model_accuracy(data, weights):
    return (1 - (count_mistakes(data, weights)/len(data))) * 100


def add_vectors(vector_a, vector_b):
    for i in range(len(vector_b)):
        vector_a[i] += vector_b[i]
    return vector_a
