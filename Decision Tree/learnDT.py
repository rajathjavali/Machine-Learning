import math
import copy
import node


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
    def __build_id3(self, attributes, data_set, tree_node, tree_depth):
        if tree_depth <= 0:
            tree_node.type = node.NodeType.LABEL
            tree_node.attribute_val = self.__best_label_finder(data_set)
            return
        # if either we exhaust our data set or feature attributes
        if len(attributes) == 0 or len(data_set) == 0:
            tree_node.type = node.NodeType.LABEL
            tree_node.attribute_val = "Error"
            return

        # checking if the subset of data set have the same label for the current path taken
        flag = data_set[0][0]
        for i in data_set:
            if i[0] != flag:
                flag = -1
                break

        # assigning the label if all the data lines have the same output
        if flag != -1:
            tree_node.type = node.NodeType.LABEL
            tree_node.attribute_val = flag
            return
        # choosing the best attribute possible which is based on highest information gain
        # information gain computation is using entropy as the measure of disorder/uncertainty
        best_attr = self.__get_best_attribute(attributes, data_set)

        # remove the selected attribute from the attributes list and
        # remove corresponding attribute value data from the data set
        tree_node.attribute_val = best_attr
        choices = attributes[best_attr].possible_vals

        # print("best attr: " + best_attr + " choices: " + str(choices))
        new_attributes = copy.deepcopy(attributes)
        new_attributes.pop(best_attr, None)
        # print("in node: " + node.attribute_val + " depth: " + str(tree_depth))
        if len(new_attributes) != 0:
            tree_node.set_choices(choices)
            tree_node.children["other"] = node.Node(node.NodeType.LABEL)
            tree_node.children["other"].attribute_val = self.__best_label_finder(data_set)
            for i in choices:
                # modify data set and attribute set
                new_data_set = []
                for data_line in data_set:
                    if data_line[self.column_index_map[best_attr]] == i:
                        new_data_set.append(data_line)
                if len(new_data_set) != 0:
                    self.__build_id3(new_attributes, new_data_set, tree_node.children[i], tree_depth - 1)
                else:
                    tree_node.children[i] = node.Node(node.NodeType.LABEL)
                    tree_node.children[i].attribute_val = self.__best_label_finder(data_set)
        else:
            tree_node.children["label"] = node.Node(node.NodeType.LABEL)
            tree_node.children["label"].attribute_val = self.__best_label_finder(data_set)

    # finds the max depth of the tree
    def max_depth_id3(self, tree_node):
        if tree_node.type == node.NodeType.LABEL:
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
    def initiate(self, attributes, data_set, tree_depth):
        if self.tree is None:
            self.tree = node.Node(node.NodeType.NODE)
        self.__build_id3(attributes, data_set, self.tree, tree_depth)
        # print(str(self.max_depth_id3(self.tree)))
        # self.print_tree(self.tree, None, None)

    # this function uses the decision tree to traverse through the tree given a data line
    # it returns back a decision label which the tree makes
    def __check_decision_tree(self, tree_node, data_line):
        if tree_node.type == node.NodeType.LABEL:
            # print(node.attribute_val)
            return tree_node.attribute_val
        choice = data_line[self.column_index_map[tree_node.attribute_val]]
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
            if decision[i] == data_set[i][self.column_index_map["label"]]:
                correct_pred += 1
        return correct_pred / total_examples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Of Class LearnDT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
