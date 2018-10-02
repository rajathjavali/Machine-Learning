# Perceptron
To run the cross validation on all the variants of perceptron execute: ./run.sh

Each perceptron is defined as  a class which stores the training data on which perceptron is trained. It also needs the max variable / features available in the training set.

run_perceptron takes in epoch value which uses the same training set to train the perceptron *(times) epoch.

run_perceptron also takes in a testing set. If testing set is passed then after every epoch accuracy is calculated. And weights resulting in max accuracy on the testing set is returned along with the total updates made accross all the epochs.

I am using a seeded random data shuffler for training on every epoch. The seeds are the same value as epochs i.e (1 ..... max).
