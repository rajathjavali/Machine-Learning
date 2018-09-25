import numpy as np


class ReadData:

    def __init__(self, fpath="", data=None):
        if not fpath and data is None:
            raise Exception("Must pass either a path to a data file or a numpy array object")

        self.raw_data, self.max_variable = self._load_data(fpath, data)

    #
    # input: path to a file or data array
    # output:
    #   1. raw data : parses the file or data array and converts it into a list of dict
    #                 for each example indicating the value for the variable.
    #                 The label for each example is added to the dict with the key as 'label'
    #   2. max variable : the highest variable number it came across in the data set
    #
    @staticmethod
    def _load_data(fpath="", data=None):
        if data is None:
            data = np.loadtxt(fpath, delimiter=',', dtype=str)
        raw_data = []
        max_variable = 0
        for line in data:
            line_data = {}
            line_split = line.split()

            #
            # Assuming data lines to be of the format "label [index:data ....]"
            # => label is at column 0 and rest all are index:value formatted data
            #
            for index, column in enumerate(line_split):
                if index == 0:
                    line_data['label'] = column
                    continue
                value = column.split(':')
                if int(value[0]) > max_variable:
                    max_variable = int(value[0])
                line_data[value[0]] = value[1]
            raw_data.append(line_data)

        return raw_data, max_variable


ReadData("dataset/diabetes.test")
