#!/usr/bin/env python3
import sys
import random
import numpy as np
from sklearn.model_selection import KFold
from collections import Counter

# Class for instances with operations
class Instances(object):
    def __init__(self):
        self.label = []
        self.attrs = []
        self.num_attrs = -1
        self.num_instances = 0
        self.attr_set = []
        

    def add_instance(self, _lbl, _attrs):
        self.label.append(_lbl)
        self.attrs.append(_attrs)
        if self.num_attrs == -1:
            self.num_attrs = len(_attrs)
        else:
            assert(self.num_attrs == len(_attrs))
        self.num_instances += 1
        assert(self.num_instances == len(self.label))

    
    def make_attr_set(self):
        self.attr_set = [set([self.attrs[i][j] for i in range(self.num_instances)]) for j in range(self.num_attrs)]


    def load_file(self, file_name):
        with open(file_name, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                self.add_instance(data[0], data[1:])
        self.make_attr_set()
        return self
    

    def split(self, att_idx):
        assert(0 <= att_idx < self.num_attrs)
        split_data = {x: Instances() for x in self.attr_set[att_idx]}
        for i in range(self.num_instances):
            key = self.attrs[i][att_idx] 
            split_data[key].add_instance(self.label[i], self.attrs[i])
        for key in split_data:
            split_data[key].attr_set = self.attr_set
        return split_data
    

    def shuffle(self):
        indices = list(range(len(self.label)))
        random.shuffle(indices)
        res = Instances()
        for x in indices:
            res.add_instance(self.label[x], self.attrs[x])
        res.attr_set = self.attr_set
        return res


    def get_subset(self, keys):
        res = Instances()
        for x in keys:
            res.add_instance(self.label[x], self.attrs[x])
        res.attr_set = self.attr_set
        return res


def compute_entropy(data):
    total_entropy = 0.0
    ########## Please Fill Missing Lines Here ##########
    if data[0] + data[1] == 0:
        return total_entropy
    else:
        val1 = data[0]/(data[0] + data[1])
        val2 = data[1]/(data[0] + data[1])

    if val1 == 0 or val2 == 0:
        return total_entropy
    else:
        total_entropy = (-val1) * np.log2(val1) - (val2 * np.log2(val2))

    return total_entropy
    

def compute_info_gain(data, att_idx):
    info_gain = 0.0
    ########## Please Fill Missing Lines Here ##########
    #compute_entropy(data[att_index])

    myList = []

    dem_yes = 0
    rep_yes = 0
    dem_no =  0
    rep_no =  0
    dem_unsure = 0
    rep_unsure = 0

    pos_b = 0
    neg_b = 0
    pos_x = 0
    neg_x = 0
    pos_o = 0
    neg_o = 0

    for i in range(0, data.num_instances): #goes from 0 to 347
        myList.append(data.attrs[i][att_idx])   #get the value at the attribute index of the ith row

    if data.label[0] == "republican" or data.label[0] == "democrat":  #house-voting-dataset
        #do something
        #indices of dems who votes yes
        for i in range(0, data.num_instances):
            if myList[i] == "y" and data.label[i] == "democrat":
                dem_yes += 1
            elif myList[i] == "y" and data.label[i] == "republican":
                rep_yes += 1
            elif myList[i] == "n" and data.label[i] == "democrat":
                dem_no +=  1
            elif myList[i] == "n" and data.label[i] == "republican":
                rep_no +=  1
            elif myList[i] == "?" and data.label[i] == "democrat":
                dem_unsure += 1
            elif myList[i] == "?" and data.label[i] == "republican":
                rep_unsure += 1

        total = dem_yes + rep_yes + dem_no + rep_no + dem_unsure + rep_unsure
        total_for_yes = dem_yes + rep_yes
        total_for_no = dem_no + rep_no
        total_for_unsure = dem_unsure + rep_unsure

        total_dems = dem_yes + dem_no + dem_unsure
        total_reps = rep_yes + rep_no + rep_unsure


        entropy_array1 = []
        entropy_array2 = []
        entropy_array3 = []
        entropy_array4 = []


        entropy_array1.append(dem_yes)
        entropy_array1.append(rep_yes)

        entropy_array2.append(dem_no)
        entropy_array2.append(rep_no)

        entropy_array3.append(dem_unsure)
        entropy_array3.append(rep_unsure)

        entropy_array4.append(total_dems)
        entropy_array4.append(total_reps)

        #call compute entropy
        val1 = compute_entropy(entropy_array1)
        val2 = compute_entropy(entropy_array2)
        val3 = compute_entropy(entropy_array3)

        current_info = compute_entropy(entropy_array4)

        expected_info = val1*(total_for_yes/total) + val2*(total_for_no/total) + val3*(total_for_unsure/total)

        info_gain = current_info - expected_info
        #print("printing info gain")
        #print(info_gain)


    elif data.label[0] == "positive" or data.label[0] == "negative":  #tic-tac-toe dataset
        for i in range(0, data.num_instances):
            if myList[i] == "b" and data.label[i] == "positive":
                pos_b += 1
            elif myList[i] == "b" and data.label[i] == "negative":
                neg_b += 1
            elif myList[i] == "x" and data.label[i] == "positive":
                pos_x +=  1
            elif myList[i] == "x" and data.label[i] == "negative":
                neg_x +=  1
            elif myList[i] == "o" and data.label[i] == "positive":
                pos_o += 1
            elif myList[i] == "o" and data.label[i] == "negative":
                neg_o += 1

        total = pos_b + neg_b + pos_x + neg_x + pos_o + neg_o
        total_for_b = pos_b + neg_b
        total_for_x = pos_x + neg_x
        total_for_o = pos_o + neg_o

        total_pos = pos_b + pos_x + pos_o
        total_neg = neg_b + neg_x + neg_o


        entropy_array1 = []
        entropy_array2 = []
        entropy_array3 = []
        entropy_array4 = []


        entropy_array1.append(pos_b)
        entropy_array1.append(neg_b)

        entropy_array2.append(pos_x)
        entropy_array2.append(neg_x)

        entropy_array3.append(pos_o)
        entropy_array3.append(neg_o)

        entropy_array4.append(total_pos)
        entropy_array4.append(total_neg)

        #call compute entropy
        val1 = compute_entropy(entropy_array1)
        val2 = compute_entropy(entropy_array2)
        val3 = compute_entropy(entropy_array3)

        current_info = compute_entropy(entropy_array4)

        expected_info = val1*(total_for_b/total) + val2*(total_for_x/total) + val3*(total_for_o/total)

        info_gain = current_info - expected_info
        #print("printing info gain")
        #print(info_gain)


    return info_gain


def comput_gain_ratio(data, att_idx):
    gain_ratio = 0.0
    ########## Please Fill Missing Lines Here ##########
    myList = []

    dem_yes = 0
    rep_yes = 0
    dem_no = 0
    rep_no = 0
    dem_unsure = 0
    rep_unsure = 0

    pos_b = 0
    neg_b = 0
    pos_x = 0
    neg_x = 0
    pos_o = 0
    neg_o = 0

    for i in range(0, data.num_instances):  # goes from 0 to 347
        myList.append(data.attrs[i][att_idx])  # get the value at the attribute index of the ith row

    if data.label[0] == "republican" or data.label[0] == "democrat":  # house-voting-dataset
        # do something
        # indices of dems who votes yes
        for i in range(0, data.num_instances):
            if myList[i] == "y" and data.label[i] == "democrat":
                dem_yes += 1
            elif myList[i] == "y" and data.label[i] == "republican":
                rep_yes += 1
            elif myList[i] == "n" and data.label[i] == "democrat":
                dem_no += 1
            elif myList[i] == "n" and data.label[i] == "republican":
                rep_no += 1
            elif myList[i] == "?" and data.label[i] == "democrat":
                dem_unsure += 1
            elif myList[i] == "?" and data.label[i] == "republican":
                rep_unsure += 1

        total = dem_yes + rep_yes + dem_no + rep_no + dem_unsure + rep_unsure
        total_for_yes = dem_yes + rep_yes
        total_for_no = dem_no + rep_no
        total_for_unsure = dem_unsure + rep_unsure

        total_dems = dem_yes + dem_no + dem_unsure
        total_reps = rep_yes + rep_no + rep_unsure

        entropy_array1 = []
        entropy_array2 = []
        entropy_array3 = []
        entropy_array4 = []

        entropy_array1.append(dem_yes)
        entropy_array1.append(rep_yes)

        entropy_array2.append(dem_no)
        entropy_array2.append(rep_no)

        entropy_array3.append(dem_unsure)
        entropy_array3.append(rep_unsure)

        entropy_array4.append(total_dems)
        entropy_array4.append(total_reps)

        # call compute entropy
        val1 = compute_entropy(entropy_array1)
        val2 = compute_entropy(entropy_array2)
        val3 = compute_entropy(entropy_array3)

        current_info = compute_entropy(entropy_array4)

        expected_info = val1 * (total_for_yes / total) + val2 * (total_for_no / total) + val3 * (
            total_for_unsure / total)

        info_gain = current_info - expected_info

        split_info = 0
        if total > 0:
            if total_for_yes > 0:
                split_info = (-(total_for_yes)/total)*np.log2(total_for_yes/total)
            if total_for_no > 0:
                split_info -= ((total_for_no/total)*np.log2(total_for_no/total))
            if total_for_unsure > 0:
                split_info -= (total_for_unsure)/total*np.log2(total_for_unsure/total)
            if split_info > 0:
                gain_ratio = info_gain/split_info


        return gain_ratio


    elif data.label[0] == "positive" or data.label[0] == "negative":  # tic-tac-toe dataset
        for i in range(0, data.num_instances):
            if myList[i] == "b" and data.label[i] == "positive":
                pos_b += 1
            elif myList[i] == "b" and data.label[i] == "negative":
                neg_b += 1
            elif myList[i] == "x" and data.label[i] == "positive":
                pos_x += 1
            elif myList[i] == "x" and data.label[i] == "negative":
                neg_x += 1
            elif myList[i] == "o" and data.label[i] == "positive":
                pos_o += 1
            elif myList[i] == "o" and data.label[i] == "negative":
                neg_o += 1

        total = pos_b + neg_b + pos_x + neg_x + pos_o + neg_o
        total_for_b = pos_b + neg_b
        total_for_x = pos_x + neg_x
        total_for_o = pos_o + neg_o

        total_pos = pos_b + pos_x + pos_o
        total_neg = neg_b + neg_x + neg_o

        entropy_array1 = []
        entropy_array2 = []
        entropy_array3 = []
        entropy_array4 = []

        entropy_array1.append(pos_b)
        entropy_array1.append(neg_b)

        entropy_array2.append(pos_x)
        entropy_array2.append(neg_x)

        entropy_array3.append(pos_o)
        entropy_array3.append(neg_o)

        entropy_array4.append(total_pos)
        entropy_array4.append(total_neg)

        # call compute entropy
        val1 = compute_entropy(entropy_array1)
        val2 = compute_entropy(entropy_array2)
        val3 = compute_entropy(entropy_array3)

        current_info = compute_entropy(entropy_array4)

        expected_info = val1 * (total_for_b / total) + val2 * (total_for_x / total) + val3 * (total_for_o / total)

        info_gain = current_info - expected_info

        split_info = 0
        if total > 0:
            if total_for_b > 0:
                split_info = (-(total_for_b) / total) * np.log2(total_for_b / total)
            if total_for_o > 0:
                split_info -= ((total_for_o / total) * np.log2(total_for_o / total))
            if total_for_x > 0:
                split_info -= ((total_for_x) / total) * np.log2(total_for_x / total)
            if split_info > 0:
                gain_ratio = info_gain / split_info

        return gain_ratio


# Class of the decision tree model based on the ID3 algorithm
class DecisionTree(object):
    def __init__(self, _instances, _sel_func):
        self.instances = _instances
        self.sel_func = _sel_func
        self.gain_function = compute_info_gain if _sel_func == 0 else comput_gain_ratio
        self.m_attr_idx = None # The decision attribute if the node is a branch
        self.m_class = None # The decision class if the node is a leaf
        self.make_tree()

    def make_tree(self):
        if self.instances.num_instances == 0:
            # No any instance for this node
            self.m_class = '**MISSING**'
        else:
            gains = [self.gain_function(self.instances, i) for i in range(self.instances.num_attrs)]
            self.m_attr_idx = np.argmax(gains)
            if np.abs(gains[self.m_attr_idx]) < 1e-9:
                # A leaf to decide the decided class
                self.m_attr_idx = None
                ########## Please Fill Missing Lines Here ##########
                self.m_class = self.instances.label[0]
            else:
                # A branch
                split_data = self.instances.split(self.m_attr_idx)
                self.m_successors = {x: DecisionTree(split_data[x], self.sel_func) for x in split_data}
                for x in self.m_successors:
                    self.m_successors[x].make_tree()

    def classify(self, attrs):
        assert((self.m_attr_idx != None) or (self.m_class != None))  #uncomment this
        if self.m_attr_idx == None:
            return self.m_class
        else:
            return self.m_successors[attrs[self.m_attr_idx]].classify(attrs)
            
 

if __name__ == '__main__':
    if len(sys.argv) < 1 + 1:
        print('--usage python3 %s data [0/1, 0-Information Gain, 1-Gain Ratio, default: 0]' % sys.argv[0], file=sys.stderr)
        sys.exit(0)
    random.seed(27145)
    np.random.seed(27145)
    
    sel_func = int(sys.argv[2]) if len(sys.argv) > 1 + 1 else 0
    assert(0 <= sel_func <= 1)

    data = Instances().load_file(sys.argv[1])
    data = data.shuffle()


    
    # 5-Fold CV
    kf = KFold(n_splits=5)
    n_fold = 0
    accuracy = []
    for train_keys, test_keys in kf.split(range(data.num_instances)):
        train_data = data.get_subset(train_keys)
        test_data = data.get_subset(test_keys)
        n_fold += 1
        model = DecisionTree(train_data, sel_func)
        predictions = [model.classify(test_data.attrs[i]) for i in range(test_data.num_instances)]
        num_correct_predictions = sum([1 if predictions[i] == test_data.label[i] else 0 for i in range(test_data.num_instances)])
        nfold_acc = float(num_correct_predictions) / float(test_data.num_instances)
        accuracy.append(nfold_acc)
        print('Fold-{}: {}'.format(n_fold, nfold_acc))

    print('5-CV Accuracy = {}'.format(np.mean(accuracy)))

        
