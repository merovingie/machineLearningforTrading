import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return 'rmikheal3'

    def verbose_print(self, *args):
        if self.verbose:
            for arg in args:
                print(arg)

    def object_info(self):
        self.verbose_print("the author of this code is ", self.author())
        if self.tree is None:
            self.verbose_print("The tree is empty!")
        else:
            self.verbose_print("The leaf size is ", self.leaf_size)
            self.verbose_print("The shape of tree is ", self.tree.shape)
            self.verbose_print("The tree is：")
            self.verbose_print(self.tree)

    def build_tree(self, dataX, dataY):
        if dataX.shape[0] <= self.leaf_size:
            leaf = np.array([[-1, np.mean(dataY), -1, -1]])
            return leaf

        if np.all(dataY == dataY[0]):
            return np.array([[-1, dataY[0], -1, -1]])

        randFeature = np.random.randint(dataX.shape[1])

        featureSplit = dataX[:, randFeature]
        splitValue = np.median(featureSplit)


        if np.all(featureSplit <= splitValue):
            return np.array([[-1, np.mean(dataY), -1, -1]])

        # split the feature
        split_left = featureSplit <= splitValue
        split_right = ~split_left

        left_dataX = dataX[split_left]
        left_dataY = dataY[split_left]
        right_dataX = dataX[split_right]
        right_dataY = dataY[split_right]

        left_branch = self.build_tree(left_dataX,left_dataY)
        right_branch = self.build_tree(right_dataX,right_dataY)

        root = np.array([[randFeature, splitValue, 1, left_branch.shape[0] + 1]])

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
                factor = int(self.tree[nodeIndex,0])

                if factor == -1:
                    predY.append(self.tree[nodeIndex,1])
                    keepSearching = False

                else:
                    splitValue = self.tree[nodeIndex,1]
                    if point[factor] <= splitValue:
                        nodeIndex = nodeIndex + int(self.tree[nodeIndex, 2])
                    else:
                        nodeIndex = nodeIndex + int(self.tree[nodeIndex, -1])

        del predY[0]

        return predY


if __name__=="__main__":
    print("the secret clue is 'zzyzx'")