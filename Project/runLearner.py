import Project.dataParser as dtP
import Project.perceptron as P
import Project.aggressivePerceptron as aP
import Perceptron.helper as helper
import Project.decisionTree as dT
import time


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


dataTrain = dtP.DataParser("movie-ratings/data-splits/data.train")
dataTest = dtP.DataParser("movie-ratings/data-splits/data.test")
dataEval = dtP.DataParser("movie-ratings/data-splits/data.eval.anon")

resultFile = open("results.txt", "a")

print("Starting Average Perceptron: learning rate: 1")
average_perceptron = P.Perceptron(dataTrain.raw_data, dataTrain.max_variable)
print("Start Time", time.asctime())
weights = average_perceptron.run_perceptron(10, 1, dataTest.raw_data)
print("End Time", time.asctime())
# accuracy = model_accuracy(dataTest.raw_data, weights)
# print("Acc Perceptron: " + str(accuracy))
results = get_predictions(dataEval.raw_data, weights)
resultFile.write("Averaged Perceptron \n")
resultFile.write(str(results) + "\n")

# print("Starting Average Perceptron: learning rate: 0.1")
# average_perceptron = P.Perceptron(dataTrain.raw_data, dataTrain.max_variable)
# weights = average_perceptron.run_perceptron(10, 0.1)
# accuracy = helper.model_accuracy(dataTest.raw_data, weights)
# print("Acc Perceptron: " + str(accuracy))
#
# print("Starting Average Perceptron: learning rate: 0.01")
# average_perceptron = P.Perceptron(dataTrain.raw_data, dataTrain.max_variable)
# weights = average_perceptron.run_perceptron(10, 0.01)
# accuracy = helper.model_accuracy(dataTest.raw_data, weights)
# print("Acc Perceptron: " + str(accuracy))

# print("Starting Aggressive Perceptron: Margin: 1")
# aggressive_perceptron = aP.AggressivePerceptron(dataTrain.raw_data, dataTrain.max_variable)
# weights = aggressive_perceptron.run_perceptron(10, 1)
# accuracy = model_accuracy(dataTest.raw_data, weights)
# print("Acc Perceptron: " + str(accuracy))
# results = get_predictions(dataEval.raw_data, weights)
# resultFile.write(str(results))
#
# # print("Starting Aggressive Perceptron: Margin: 0.1")
# # aggressive_perceptron = aP.AggressivePerceptron(dataTrain.raw_data, dataTrain.max_variable)
# # weights = aggressive_perceptron.run_perceptron(10, 0.1)
# # accuracy = helper.model_accuracy(dataTest.raw_data, weights)
# # print("Acc Perceptron: " + str(accuracy))
# #
# # print("Starting Aggressive Perceptron: Margin: 0.01")
# # aggressive_perceptron = aP.AggressivePerceptron(dataTrain.raw_data, dataTrain.max_variable)
# # weights = aggressive_perceptron.run_perceptron(10, 0.01)
# # accuracy = helper.model_accuracy(dataTest.raw_data, weights)
# # print("Acc Perceptron: " + str(accuracy))


# decisionTree = dT.DecisionTree(dataTrain.raw_data)
# decisionTree.initiate()