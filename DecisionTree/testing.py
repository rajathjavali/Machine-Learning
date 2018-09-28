# Following are some of the test cases for analysis and reporting:
import DecisionTree.data as data
import DecisionTree.learnDT as learnDT
import math
import numpy as np

data_obj = data.Data("./data/train.csv", None)
data_obj_test = data.Data("./data/test.csv", None)


def build_and_test_decision_tree_no_depth_limit():
    label_counter = {}
    for k in data_obj.raw_data:
        label = k[data_obj.column_index_dict["label"]]
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1

    # assuming every attribute appears from root to any leaf
    # this is always the max depth possible for a decision tree
    depth = len(data_obj.attributes)

    # Learning phase on the decision tree
    decision_tree = learnDT.LearnDT(label_counter.values(), data_obj.column_index_dict)
    decision_tree.initiate(data_obj.attributes, data_obj.raw_data, depth)

    print("Test 1: Build and test a decision tree based on the given training data set")
    print("Accuracy on Training Set: " + str(decision_tree.run_id3(data_obj.raw_data)))
    print("Accuracy on Test Set: " + str(decision_tree.run_id3(data_obj_test.raw_data)))


def build_and_test_decision_tree_with_depth_limit(limit_depth):
    label_counter = {}
    for k in data_obj.raw_data:
        label = k[data_obj.column_index_dict["label"]]
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1

    # assuming every attribute appears from root to any leaf
    # this is always the max depth possible for a decision tree
    depth = limit_depth

    # Learning phase on the decision tree
    decision_tree = learnDT.LearnDT(label_counter.values(), data_obj.column_index_dict)
    decision_tree.initiate(data_obj.attributes, data_obj.raw_data, depth)

    print("Test 1: Build and test a decision tree based on the given training data set")
    print("Accuracy on Training Set: " + str(decision_tree.run_id3(data_obj.raw_data)))
    print("Accuracy on Test Set: " + str(decision_tree.run_id3(data_obj_test.raw_data)))


# here parameter k refers to the number of cross folds
def build_and_test_decision_tree_using_cross_validation_and_limiting_depth(k):
    all_data_set = {}
    column_index_dict = data_obj.column_index_dict
    attributes = data_obj.attributes
    label_index = column_index_dict["label"]

    for i in range(1, k+1):
        key_value = "fold{0}.csv".format(i)
        all_data_set[key_value] = data.Data("./data/CVfolds/{0}".format(key_value), None)

    test_dict, std_dev = {}, {}
    # testing for a range of depths
    depth_set = {1, 2, 3, 4, 5, 10, 15}
    for depth in depth_set:
        # k fold cross validation
        print("Depth: " + str(depth))
        accuracies = []
        for i in range(1, k+1):
            test_set = "fold{0}.csv".format(i)
            data_set = None
            # creating data set with subset of the folds
            for j in range(1, k+1):
                if j != i:
                    key_value = "fold{0}.csv".format(j)
                    if data_set is None:
                        data_set = all_data_set[key_value].raw_data
                    else:
                        data_set = np.concatenate((data_set.tolist(), all_data_set[key_value].raw_data.tolist()))
                    # list_of_dictionaries.append(all_data_set[key_value].raw_data)

            label_counter = {}
            for data_line in data_set:
                label = data_line[label_index]
                if label in label_counter:
                    label_counter[label] += 1
                else:
                    label_counter[label] = 1

            decision_tree = learnDT.LearnDT(label_counter.values(), column_index_dict)
            decision_tree.initiate(attributes, data_set, depth)

            print("Cross Fold Test " + str(i))
            print("Accuracy on Training Set: " + str(decision_tree.run_id3(data_set)))
            accuracy = decision_tree.run_id3(all_data_set[test_set].raw_data)
            accuracies.append(accuracy)
            print("Accuracy on Test Set: " + str(accuracy))

        mean = sum(accuracies) / len(accuracies)
        deviation = 0
        for i in accuracies:
            deviation += (i - mean) * (i - mean)

        deviation /= len(accuracies)
        deviation = math.sqrt(deviation)
        test_dict["depth " + str(depth)] = mean
        std_dev["depth " + str(depth)] = deviation

    print("\n\nDepth\t\tAccuracy\t\t\tStd. Dev")
    for key, value in test_dict.items():
        print(str(key) + " : " + str(value) + " : " + str(std_dev[key]))


# invoking all the test cases
print("Learning and build full decision tree\n")
build_and_test_decision_tree_no_depth_limit()
print("\n\n\n\n\n\nCross Validation using training data set\n")
build_and_test_decision_tree_using_cross_validation_and_limiting_depth(5)
print("\n\n\n\n\n\nLearning and building decision tree limited by depth\n")
build_and_test_decision_tree_with_depth_limit(5)
