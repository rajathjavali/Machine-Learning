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

