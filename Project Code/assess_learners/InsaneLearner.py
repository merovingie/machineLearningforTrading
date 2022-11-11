import numpy as np
import BagLearner as bl
import LinRegLearner as lrl


class InsaneLearner(object):

    def __init__(self, verbose = False):
        self.verbose = verbose
        self.learners = []
        for i in range(20):
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=self.verbose))

    def author(self):
        return 'rmikhael3'

    def add_evidence(self, dataX, dataY):
        for learner in self.learners:
            learner.add_evidence(dataX, dataY)

    def query(self, points):
        predY = []
        for learner in self.learners:
            predY.append(learner.query(points))

        return np.mean(predY, axis=0)

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
