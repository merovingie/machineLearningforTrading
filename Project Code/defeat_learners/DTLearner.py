""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  	  			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		  	  			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import warnings  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class DTLearner(object):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		  	  			  		 			     			  	 
    your own correct DTLearner from Project 3.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		  	  			  		 			     			  	 
    :type leaf_size: int  		  	   		  	  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  	  			  		 			     			  	 
    :type verbose: bool  		  	   		  	  			  		 			     			  	 
    """

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def verbose_print(self, *args):
        if self.verbose:
            for arg in args:
                print(arg)

    def corr_coef(self, a, b):
        return abs(np.corrcoef(a, b)[0, 1])

    def author(self):
        return 'rmikhael3'

    def object_info(self):
        self.verbose_print("the author of this code is ", self.author())
        if self.tree is None:
            self.verbose_print("The tree is empty!")
        else:
            self.verbose_print("The leaf size is ", self.leaf_size)
            self.verbose_print("The shape of tree is ", self.tree.shape)
            self.verbose_print("The tree is：")
            self.verbose_print(self.tree)

    # Correlation is a better way for this according to lecture -- no entropy or gini
    def best_feature(self, dataX, dataY):
        corrList = []
        for i in range(dataX.shape[1]):
            corrList.append(self.corr_coef(dataX[:, i], dataY))
        return np.argmax(corrList)

    def build_tree(self, dataX, dataY):
        if dataX.shape[0] <= self.leaf_size:
            leaf = np.array([[-1, np.mean(dataY), -1, -1]])
            return leaf
        elif np.std(dataY) == 0:
            return np.array([[-1, dataY[0], -1, -1]])

        bestFeature = self.best_feature(dataX, dataY)

        featureSplit = dataX[:, bestFeature]
        splitValue = np.median(featureSplit)

        if np.all(featureSplit <= splitValue):
            return np.array([[-1, np.mean(dataY), -1, -1]])

        split_left = featureSplit <= splitValue
        split_right = ~split_left

        left_dataX = dataX[split_left]
        left_dataY = dataY[split_left]
        right_dataX = dataX[split_right]
        right_dataY = dataY[split_right]

        left_branch = self.build_tree(left_dataX, left_dataY)
        right_branch = self.build_tree(right_dataX, right_dataY)

        root = np.array([[bestFeature, splitValue, 1, left_branch.shape[0] + 1]])

        return np.vstack((root, left_branch, right_branch))

    def add_evidence(self, dataX, dataY):
        self.tree = self.build_tree(dataX, dataY)
        self.object_info()

    def query(self, points):
        predY = [0]
        for point in points:
            keepSearching = True
            nodeIndex = 0
            while keepSearching:
                factor = int(self.tree[nodeIndex, 0])  # get factor to check

                if factor == -1:
                    predY.append(self.tree[nodeIndex, 1])
                    keepSearching = False

                else:  # if not compare the factor value with the points to determine which node to search
                    splitValue = self.tree[nodeIndex, 1]
                    if point[factor] <= splitValue:
                        nodeIndex = nodeIndex + int(self.tree[nodeIndex, 2])  # goes to the left branch
                    else:
                        nodeIndex = nodeIndex + int(self.tree[nodeIndex, -1])  # goes to the right branch

        del predY[0]

        return predY
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  	  			  		 			     			  	 
