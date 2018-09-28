import Perceptron.helper
import Perceptron.readData
import Perceptron.simplePerceptron
import Perceptron.marginPerceptron
import Perceptron.decayLearningPerceptron

epoch_set = [10, 20, 30, 40, 50, 100]
learning_rate_set = [1, 0.1, 0.01]
train_data_set = Perceptron.readData.ReadData("dataset/diabetes.train")

for learning_rate in learning_rate_set:
    print("\n--------------------------------Learning Rate: " + str(learning_rate) + "---------------------------\n")
    for epoch in epoch_set:
        print("\n----------------------Epoch: " + str(epoch) + "---------------------------\n")

        simple_perceptron = Perceptron.simplePerceptron.\
            SimplePerceptron(train_data_set.raw_data, train_data_set.max_variable)
        weights = simple_perceptron.run_perceptron(epoch, learning_rate)

        print("Simple Perceptron Accuracy: " + str(Perceptron.helper.model_accuracy(train_data_set.raw_data, weights)))

        decay_perceptron = Perceptron.decayLearningPerceptron.\
            DecayLearningPercepton(train_data_set.raw_data, train_data_set.max_variable)
        weights = decay_perceptron.run_perceptron(epoch, learning_rate)

        print("Decay Learning Perceptron Accuracy: " + str(
            Perceptron.helper.model_accuracy(train_data_set.raw_data, weights)))

        margin_value_set = [1, 0.1, 0.01]
        for margin_value in margin_value_set:
            margin_perceptron = Perceptron.marginPerceptron.\
                MarginPerceptron(train_data_set.raw_data, train_data_set.max_variable)
            weights = margin_perceptron.run_perceptron(epoch, margin_value, learning_rate)

            print("Margin Perceptron (" + str(margin_value) + ") Accuracy: " + str(
                Perceptron.helper.model_accuracy(train_data_set.raw_data, weights)))

    break
