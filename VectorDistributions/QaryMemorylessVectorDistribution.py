import math

import numpy as np
from scipy.special import logsumexp

import VectorDistribution


class QaryMemorylessVectorDistribution(VectorDistribution.VectorDistribution):

    def __init__(self, q, length, use_log=False):
        assert (q > 1)
        self.q = q
        assert (length > 0)
        self.probs = np.empty((length, q),
                              dtype=np.float)  # so, probs[i][x] equals the probability of x transmitted or received at time i
        self.probs[:] = np.nan
        self.length = length

        self.use_log = use_log
        if self.use_log:
            self.default_marginal_probs = [-math.log(self.q)] * self.q
        else:
            self.default_marginal_probs = [1 / self.q] * self.q

    def minusTransform(self):
        assert (self.length % 2 == 0)
        halfLength = self.length // 2

        newVector = QaryMemorylessVectorDistribution(self.q, halfLength, use_log=self.use_log)
        if self.use_log:
            newVector.probs[:] = -math.inf
        else:
            newVector.probs[:] = 0.0

        for x1 in range(self.q):
            for x2 in range(self.q):
                u1 = (x1 + x2) % self.q
                if self.use_log:
                    newVector.probs[:, u1] = np.logaddexp(newVector.probs[:, u1], np.add(self.probs[::2, x1], self.probs[1::2, x2]))
                else:
                    newVector.probs[:, u1] += self.probs[::2, x1] * self.probs[1::2, x2]
        return newVector

    def plusTransform(self, uminusDecisions):
        assert (self.length % 2 == 0)
        halfLength = self.length // 2

        newVector = QaryMemorylessVectorDistribution(self.q, halfLength, use_log=self.use_log)
        if self.use_log:
            newVector.probs[:] = -math.inf
        else:
            newVector.probs[:] = 0.0

        u1 = np.array(uminusDecisions)
        for u2 in range(self.q):
            x1 = (u1 + u2) % self.q
            x2 = -u2 % self.q
            if self.use_log:
                newVector.probs[:, u2] = np.logaddexp(newVector.probs[:, u2], self.probs[np.arange(self.length, step=2), x1] + self.probs[1::2, x2])
            else:
                newVector.probs[:, u2] += self.probs[np.arange(self.length, step=2), x1] * self.probs[1::2, x2]

        return newVector

    def __len__(self):
        return self.length

    def calcMarginalizedProbabilities(self):
        assert (len(self) == 1)

        marginalizedProbs = np.empty(self.q, dtype=np.float)

        if self.use_log:
            s = logsumexp(self.probs[0])
        else:
            s = sum(self.probs[0])

        if self.use_log and s > -math.inf:
            for x in range(self.q):
                marginalizedProbs[x] = self.probs[0][x] - s
                # marginalizedProbs[x] = self.probs[0][x]
        elif (not self.use_log) and s > 0.0:
            for x in range(self.q):
                marginalizedProbs[x] = self.probs[0][x] / s
                # marginalizedProbs[x] = self.probs[0][x]
        else:
            marginalizedProbs = self.default_marginal_probs

        return marginalizedProbs

    def calcNormalizationVector(self):
        normalization = np.zeros(self.length)

        for i in range(self.length):
            # normalization[i] = self.probs[i].max(axis=0)
            if self.use_log:
                normalization[i] = logsumexp(self.probs[i])
            else:
                normalization[i] = sum(self.probs[i])

        return normalization

    def normalize(self, normalization=None):
        if normalization is None:
            normalization = self.calcNormalizationVector()
        for i in range(self.length):
            t = normalization[i]
            if not self.use_log:
                assert (t >= 0)
            if self.use_log:
                if t != -math.inf:
                    for x in range(self.q):
                        self.probs[i][x] -= t
            else:
                if t != 0:
                    for x in range(self.q):
                        self.probs[i][x] /= t
