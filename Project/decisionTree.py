import time
from enum import Enum
import math
import copy


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


class DecisionTree:

    def __init__(self, data_set):
        self.tree = None
        self.data_set = data_set

        # calculate count of all labels
        label_counter = {}
        for line in data_set:
            label = line["label"]
            if label in label_counter:
                label_counter[label] += 1
            else:
                label_counter[label] = 1
        label_count_array = label_counter.values()

        self.total_samples = sum(label_count_array)
        self.entropy_label = self.__calculate_entropy(label_count_array)

        self.attribute_set = {}
        self.__create_attribute_set()

    def __create_attribute_set(self):
        for line in self.data_set:
            for k, v in line.items():
                if k == "label":
                    continue
                if k in self.attribute_set:
                    self.attribute_set[k].add(v)
                else:
                    self.attribute_set[k] = {'0', v}

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

    @staticmethod
    def __calculate_information_gain(entropy_label, entropy_list):
        entropy_value = 0
        total = 0
        for _, value in entropy_list.items():
            total += value[1]
        for key, value in entropy_list.items():
            entropy_value += value[1] / total * value[0]
        return entropy_label - entropy_value

    @staticmethod
    def __get_best_information_gain(information_gain_list):
        max_info_gain = 0
        best_attr = 0
        for key, value in information_gain_list.items():
            if value >= max_info_gain:
                max_info_gain = value
                best_attr = key
        return best_attr

    # # goes through the list of attributes, computes entropy and Information gain
    # # returns the attribute having the highest information gain
    # def __get_best_attribute(self, attribute_set, data_set):
    #     information_gain = {}
    #     # for every attribute still left in the list
    #     for attribute, possible_vals in attribute_set.items():
    #         entropy_list = {}
    #         # for every possible outcomes for the chosen attribute
    #         for j in possible_vals:
    #             attr_label_counter = {}
    #             # iterate through the data set to count the number of labels matching against
    #             # the specific [attribute, value] pair
    #             for data_line in data_set:
    #                 if j == 0:
    #                     if attribute not in data_line:
    #                         label = data_line["label"]
    #                         if label in attr_label_counter:
    #                             attr_label_counter[label] += 1
    #                         else:
    #                             attr_label_counter[label] = 1
    #                 elif attribute in data_line:
    #                     # checking if the data line at the chosen attribute index has the value 'j'
    #                     if data_line[attribute] == j:
    #                         # getting the label for this data line to count every such occurrences
    #                         label = data_line["label"]
    #                         if label in attr_label_counter:
    #                             attr_label_counter[label] += 1
    #                         else:
    #                             attr_label_counter[label] = 1
    #
    #             # calculate the entropy of all the choices for the chosen attribute
    #             values = attr_label_counter.values()
    #             if len(values) != 0:
    #                 entropy_list[j] = [self.__calculate_entropy(values), sum(values)]
    #         # calculate the information gain for the attribute
    #         information_gain[attribute] = self.__calculate_information_gain(entropy_list)
    #     # return the highest information gain attribute among the list
    #         return self.__get_best_information_gain(information_gain)
    #
    # def __get_best_attribute_v2(self, attribute_set, data_set):
    #     information_gain = {}
    #     # for every attribute still left in the list
    #     for attribute, possible_vals in attribute_set.items():
    #         entropy_list = {}
    #
    #         possible_vals_counter = {}
    #         for data_line in data_set:
    #             if attribute not in data_line:
    #                 key = '0'
    #             else:
    #                 key = data_line[attribute]
    #
    #             label = data_line["label"]
    #             if key in possible_vals_counter:
    #                 if label in possible_vals_counter[key]:
    #                     possible_vals_counter[key][label] += 1
    #                 else:
    #                     possible_vals_counter[key][label] = 1
    #             else:
    #                 possible_vals_counter[key] = {}
    #                 possible_vals_counter[key][label] = 1
    #
    #         for key, value in possible_vals_counter.items():
    #             values = value.values()
    #             if len(values) != 0:
    #                 entropy_list[key] = [self.__calculate_entropy(values), sum(values)]
    #         information_gain[attribute] = self.__calculate_information_gain(entropy_list)
    #     return self.__get_best_information_gain(information_gain)

    def __get_best_attribute_v3(self, attribute_set, data_set):
        information_gain = {}
        # for every attribute still left in the list
        entropy_list = {}
        attribute_counter = {}
        # i = 0

        label_counter = {}
        for line in data_set:
            label = line["label"]
            if label in label_counter:
                label_counter[label] += 1
            else:
                label_counter[label] = 1
        label_count_array = label_counter.values()

        entropy_label = self.__calculate_entropy(label_count_array)

        print("Start Time", time.asctime())
        for data_line in data_set:
            label = data_line["label"]
            # if i % 500 == 0:
            #     print(i)
            # i += 1
            # for key, value in data_line.items():
            for attribute in attribute_set.keys():
                if attribute in data_line.keys():
                    value = data_line[attribute]
                else:
                    value = '0'
                if attribute in attribute_counter.keys():
                    if value in attribute_counter[attribute].keys():
                        if label in attribute_counter[attribute][value].keys():
                            attribute_counter[attribute][value][label] += 1
                        else:
                            attribute_counter[attribute][value][label] = 1
                    else:
                        attribute_counter[attribute][value] = {}
                        attribute_counter[attribute][value][label] = 1
                else:
                    attribute_counter[attribute] = {}
                    attribute_counter[attribute][value] = {}
                    attribute_counter[attribute][value][label] = 1
        print("End Time", time.asctime())

        for attribute, possible_vals in attribute_counter.items():
            for possible_val, label_counts in possible_vals.items():
                values = label_counts.values()
                if len(values) != 0:
                    entropy_list[possible_val] = [self.__calculate_entropy(values), sum(values)]
            information_gain[attribute] = self.__calculate_information_gain(entropy_label, entropy_list)
        return self.__get_best_information_gain(information_gain)

    # this function helps in fixing a label at the leaf node
    # this label is the result returned for the input data traversing through
    # the decision tree
    @staticmethod
    def __best_label_finder(data_set):
        label_counter = {}
        for data_line in data_set:
            label = data_line["label"]
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
    def __build_id3(self, attributes, data_set, tree_node, tree_depth):
        if tree_depth <= 0:
            tree_node.type = NodeType.LABEL
            tree_node.attribute_val = self.__best_label_finder(data_set)
            return
        # if either we exhaust our data set or feature attributes
        if len(attributes) == 0 or len(data_set) == 0:
            tree_node.type = NodeType.LABEL
            tree_node.attribute_val = "Error"
            return

        # checking if the subset of data set have the same label for the current path taken
        flag = data_set[0]["label"]
        for line in data_set:
            if line["label"] != flag:
                flag = -1
                break

        # assigning the label if all the data lines have the same output
        if flag != -1:
            tree_node.type = NodeType.LABEL
            tree_node.attribute_val = flag
            return
        # choosing the best attribute possible which is based on highest information gain
        # information gain computation is using entropy as the measure of disorder/uncertainty
        best_attr = self.__get_best_attribute_v3(attributes, data_set)

        # remove the selected attribute from the attributes list and
        # remove corresponding attribute value data from the data set
        tree_node.attribute_val = best_attr
        choices = attributes[best_attr]

        # print("best attr: " + best_attr + " choices: " + str(choices))
        new_attributes = copy.deepcopy(attributes)
        new_attributes.pop(best_attr, None)
        # print("in node: " + node.attribute_val + " depth: " + str(tree_depth))
        if len(new_attributes) != 0:
            tree_node.set_choices(choices)
            tree_node.children["other"] = Node(NodeType.LABEL)
            tree_node.children["other"].attribute_val = self.__best_label_finder(data_set)
            for i in choices:
                # modify data set and attribute set
                new_data_set = []
                for data_line in data_set:
                    if i == '0':
                        if best_attr not in data_line:
                            new_data_set.append(data_line)
                    elif best_attr in data_line:
                        if data_line[best_attr] == i:
                            new_data_set.append(data_line)
                if len(new_data_set) != 0:
                    self.__build_id3(new_attributes, new_data_set, tree_node.children[i], tree_depth - 1)
                else:
                    tree_node.children[i] = Node(NodeType.LABEL)
                    tree_node.children[i].attribute_val = self.__best_label_finder(data_set)
        else:
            tree_node.children["label"] = Node(NodeType.LABEL)
            tree_node.children["label"].attribute_val = self.__best_label_finder(data_set)

    # finds the max depth of the tree
    def max_depth_id3(self, tree_node):
        if tree_node.type == NodeType.LABEL:
            return 0
        max_depth = 0
        for i in tree_node.children:
            depth = self.max_depth_id3(tree_node.children[i])
            if max_depth < depth:
                max_depth = depth
        return max_depth + 1

    # function to print the generated decision tree
    def print_tree(self, tree_node, choice, parent):
        print("on choice:" + str(choice) + " from parent: " + str(parent))
        print("Node: "+tree_node.attribute_val)
        for key, value in tree_node.children.items():
            self.print_tree(value, key, tree_node.attribute_val)

    # public facing function which initiates building a decision tree given a training data set
    def initiate(self, tree_depth=None):
        if tree_depth is None:
            tree_depth = len(self.attribute_set)
        if self.tree is None:
            self.tree = Node(NodeType.NODE)
        self.__build_id3(self.attribute_set, self.data_set, self.tree, tree_depth)
        # print(str(self.max_depth_id3(self.tree)))
        # self.print_tree(self.tree, None, None)

    # this function uses the decision tree to traverse through the tree given a data line
    # it returns back a decision label which the tree makes
    def __check_decision_tree(self, tree_node, data_line):
        if tree_node.type == NodeType.LABEL:
            # print(node.attribute_val)
            return tree_node.attribute_val
        if tree_node.attribute_val in data_line:
            choice = data_line[tree_node.attribute_val]
        else:
            choice = 0

        # print(node.attribute_val + " Choice: " + str(choice))
        if choice in tree_node.children:
            return self.__check_decision_tree(tree_node.children[choice], data_line)
        else:
            return self.__check_decision_tree(tree_node.children["other"], data_line)

    # public facing api which takes in a data set uses the ID3 decision tree build on the training set
    # to make a decision on the data lines and also calculates the accuracy of the tree
    def run_id3(self, data_set):
        decision = []
        for i in data_set:
            decision.append(self.__check_decision_tree(self.tree, i))

        correct_pred = 0
        total_examples = len(decision)
        for i in range(total_examples):
            if decision[i] == data_set[i]["label"]:
                correct_pred += 1
        return correct_pred / total_examples

    def get_labels(self, data_set):
        decision = []
        for i in data_set:
            decision.append(self.__check_decision_tree(self.tree, i))

        return decision
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Of Class LearnDT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from Project import dataParser as dtP

dataTrain = dtP.DataParser("movie-ratings/data-splits/data.train")
dataTest = dtP.DataParser("movie-ratings/data-splits/data.test")
dataEval = dtP.DataParser("movie-ratings/data-splits/data.eval.anon")

decision_tree = DecisionTree(dataTrain.raw_data)
decision_tree.initiate(15)
test_accuracy = decision_tree.run_id3(dataTest.raw_data)
print("test accuracy: " + str(test_accuracy))
results = decision_tree.get_labels(dataEval.raw_data)

resultFile = open("results.txt", "a")
resultFile.write("decision Tree: " + str(test_accuracy) + "\n")
resultFile.write(str(results) + "\n")
