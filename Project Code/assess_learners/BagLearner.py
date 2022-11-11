import numpy as np

4
class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost = False, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.models = []

    def author(self):
        return 'rmikhael3'

    def add_evidence(self, dataX, dataY):

        for i in range(0, self.bags):
            learner = self.learner(**self.kwargs)

            # create a bag of data by sampling dataX and dataY
            choice = np.random.choice(a=dataX.shape[0], size=len(dataY),replace=True)
            bagDataX = dataX[choice]
            bagDataY = dataY[choice]

            # train the learners with different data
            learner.add_evidence(bagDataX, bagDataY)

            # save the learner model
            self.models.append(learner)

    def query(self, points):
        if not self.models:
            return np.nan

        predY = []
        for learner in self.models:
            predY.append(learner.query(points))
        return np.mean(predY, axis=0)  # average of the predicted value by all models is the result of the bagLearner


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")