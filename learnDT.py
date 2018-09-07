import data
import math
import copy
import numpy as np
from enum import Enum


class NodeType(Enum):
    NODE = 1
    LABEL = 2


class Node:
    def __init__(self, node_type):
        self.type = node_type
        self.attribute_val = "default"
        self.children = {}

    # creates branches for all the choices possible from the current node
    def set_choices(self, choices):
        for i in choices:
            self.children[i] = Node(NodeType.NODE)
    #
    # def set_attribute_label(self, value):
    #     self.attribute_val = value


class LearnDT:

    def __init__(self, entropy_label_array, column_index_map):
        self.tree = None
        self.total_samples = sum(entropy_label_array)
        self.entropy_label = self.__calculate_entropy(entropy_label_array)
        self.column_index_map = column_index_map

    @staticmethod
    def __calculate_entropy(array_values):
        entropy = 0
        total_samples = sum(array_values)
        if total_samples == 0:
            print("sample space is 0")
            exit(-1)
        for i in array_values:
            fraction = i / total_samples
            entropy += -fraction * math.log(fraction, 2)
        return entropy

    def __calculate_information_gain(self, entropy_list):
        entropy_value = 0
        for key, value in entropy_list.items():
            entropy_value += value[1] / self.total_samples * value[0]
        return self.entropy_label - entropy_value

    @staticmethod
    def __get_best_information_gain(information_gain_list):
        max_info_gain = 0
        best_attr = 0
        for key, value in information_gain_list.items():
            if value > max_info_gain:
                max_info_gain = value
                best_attr = key
        return best_attr

    # goes through the list of attributes, computes entropy and Information gain
    # returns the attribute having the highest information gain
    def __get_best_attribute(self, attribute_set, data_set):
        information_gain = {}
        # for every attribute still left in the list
        for i in attribute_set:
            entropy_list = {}
            # for every possible outcomes for the chosen attribute
            for j in attribute_set[i].possible_vals:
                attr_label_counter = {}
                # iterate through the data set to count the number of labels matching against
                # the specific [attribute, value] pair
                for data_line in data_set:
                    # checking if the data line at the chosen attribute index has the value 'j'
                    if data_line[self.column_index_map[i]] == j:
                        # getting the label for this data line to count every such occurrences
                        label = data_line[self.column_index_map["label"]]
                        if label in attr_label_counter:
                            attr_label_counter[label] += 1
                        else:
                            attr_label_counter[label] = 1
                # calculate the entropy of all the choices for the chosen attribute
                values = attr_label_counter.values()
                if len(values) != 0:
                    entropy_list[j] = [self.__calculate_entropy(values), sum(values)]
            # calculate the information gain for the attribute
            information_gain[i] = self.__calculate_information_gain(entropy_list)
        # return the highest information gain attribute among the list
        return self.__get_best_information_gain(information_gain)

    # this function helps in fixing a label at the leaf node
    # this label is the result returned for the input data traversing through
    # the decision tree
    def __best_label_finder(self, data_set):
        label_counter = {}
        for data_line in data_set:
            label = data_line[self.column_index_map["label"]]
            if label in label_counter:
                label_counter[label] += 1
            else:
                label_counter[label] = 1

        max_label = 0
        which_label = ''
        for key, value in label_counter.items():
            if value > max_label:
                max_label = value
                which_label = key

        return which_label

    # this function helps build a decision tree based on the training set given as parameter
    # it is a recursive function which builds a decision tree using the ID3 algorithm
    def __build_id3(self, attributes, data_set, node, tree_depth):
        if tree_depth <= 0:
            node.type = NodeType.LABEL
            node.attribute_val = self.__best_label_finder(data_set)
            return
        # if either we exhaust our data set or feature attributes
        if len(attributes) == 0 or len(data_set) == 0:
            node.type = NodeType.LABEL
            node.attribute_val = "Error"
            return

        # checking if the subset of data set have the same label for the current path taken
        flag = data_set[0][0]
        for i in data_set:
            if i[0] != flag:
                flag = -1
                break

        # assigning the label if all the data lines have the same output
        if flag != -1:
            node.type = NodeType.LABEL
            node.attribute_val = flag
            return
        # choosing the best attribute possible which is based on highest information gain
        # information gain computation is using entropy as the measure of disorder/uncertainty
        best_attr = self.__get_best_attribute(attributes, data_set)

        # remove the selected attribute from the attributes list and
        # remove corresponding attribute value data from the data set
        node.attribute_val = best_attr
        choices = attributes[best_attr].possible_vals

        # print("best attr: " + best_attr + " choices: " + str(choices))
        new_attributes = copy.deepcopy(attributes)
        new_attributes.pop(best_attr, None)
        # print("in node: " + node.attribute_val + " depth: " + str(tree_depth))
        if len(new_attributes) != 0:
            node.set_choices(choices)
            for i in choices:
                # modify data set and attribute set
                new_data_set = []
                for data_line in data_set:
                    if data_line[self.column_index_map[best_attr]] == i:
                        new_data_set.append(data_line)
                if len(new_data_set) != 0:
                    self.__build_id3(new_attributes, new_data_set, node.children[i], tree_depth - 1)
                else:
                    node.children[i] = Node(NodeType.LABEL)
                    node.children[i].attribute_val = self.__best_label_finder(data_set)
        else:
            node.children["label"] = Node(NodeType.LABEL)
            node.children["label"].attribute_val = self.__best_label_finder(data_set)

    def max_depth_id3(self, node):
        if node.type == NodeType.LABEL:
            return 0
        max_depth = 0
        for i in node.children:
            depth = self.max_depth_id3(node.children[i])
            if max_depth < depth:
                max_depth = depth
        return max_depth + 1

    # function to print the generated decision tree
    def print_tree(self, node, choice, parent):
        print("on choice:" + str(choice) + " from parent: " + str(parent))
        print("Node: "+node.attribute_val)
        for key, value in node.children.items():
            self.print_tree(value, key, node.attribute_val)

    # public facing function which initiates building a decision tree given a training data set
    def initiate(self, attributes, data_set, tree_depth):
        if self.tree is None:
            self.tree = Node(NodeType.NODE)
        self.__build_id3(attributes, data_set, self.tree, tree_depth)
        # print(str(self.max_depth_id3(self.tree)))
        # self.print_tree(self.tree, None, None)

    # this function uses the decision tree to traverse through the tree given a data line
    # it returns back a decision label which the tree makes
    def __check_decision_tree(self, node, data_line):
        if node.type == NodeType.LABEL:
            # print(node.attribute_val)
            return node.attribute_val
        choice = data_line[self.column_index_map[node.attribute_val]]
        # print(node.attribute_val + " Choice: " + str(choice))
        return self.__check_decision_tree(node.children[choice], data_line)

    # public facing api which takes in a data set uses the ID3 decision tree build on the training set
    # to make a decision on the data lines and also calculates the accuracy of the tree
    def run_id3(self, data_set):
        decision = []
        for i in data_set:
            decision.append(self.__check_decision_tree(self.tree, i))

        correct_pred = 0
        total_examples = len(decision)
        for i in range(total_examples):
            if decision[i] == data_set[i][self.column_index_map["label"]]:
                correct_pred += 1
        return correct_pred / total_examples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Of Class LearnDT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Following are some of the test cases for analysis and reporting:


def build_and_test_decision_tree_no_depth_limit():
    data_obj = data.Data("./data/train.csv", None)
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
    decision_tree = LearnDT(label_counter.values(), data_obj.column_index_dict)
    decision_tree.initiate(data_obj.attributes, data_obj.raw_data, depth)

    print("Test 1: Build and test a decision tree based on the given training data set")
    print("Accuracy on Training Set: " + str(decision_tree.run_id3(data_obj.raw_data)))

    data_obj_test = data.Data("./data/test.csv", None)
    print("Accuracy on Test Set: " + str(decision_tree.run_id3(data_obj_test.raw_data)))


# here parameter k refers to the number of cross folds
def build_and_test_decision_tree_using_cross_validation_and_limiting_depth(k):
    all_data_set = {}
    total_data = data.Data("./data/train.csv", None)
    column_index_dict = total_data.column_index_dict
    attributes = total_data.attributes
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
                        np.concatenate((data_set, all_data_set[key_value].raw_data))
                    # list_of_dictionaries.append(all_data_set[key_value].raw_data)

            label_counter = {}
            for data_line in data_set:
                label = data_line[label_index]
                if label in label_counter:
                    label_counter[label] += 1
                else:
                    label_counter[label] = 1

            decision_tree = LearnDT(label_counter.values(), column_index_dict)
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
        test_dict["depth" + str(depth)] = mean
        std_dev["depth"+str(depth)] = deviation

    for key, value in test_dict.items():
        print(str(key) + " : " + str(value) + " : " + str(std_dev[key]))


# invoking all the test cases
# build_and_test_decision_tree_no_depth_limit()
build_and_test_decision_tree_using_cross_validation_and_limiting_depth(5)


# # dictionary merger helper function
# def merge_dicts(*dict_args):
#     """
#     Given any number of dicts, shallow copy and merge into a new dict,
#     precedence goes to key value pairs in latter dicts.
#     """
#     result = {}
#     for dictionary in dict_args:
#         result.update(dictionary)
#     return result
