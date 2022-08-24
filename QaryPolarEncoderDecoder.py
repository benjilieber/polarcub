import functools
import itertools
import random
import sys
from timeit import default_timer as timer
from enum import Enum
import math

import numpy as np

from ScalarDistributions import QaryMemorylessDistribution


class uIndexType(Enum):
    frozen = 0
    information = 1

class ProbResult(Enum):
    SuccessActualIsMax = 0
    SuccessActualSmallerThanMax = 1
    FailActualLargerThanMax = 2
    FailActualIsMax = 3
    FailActualWithinRange = 4
    FailActualSmallerThanMin = 5

class QaryPolarEncoderDecoder:
    def __init__(self, q, length, frozenSet, commonRandomnessSeed, use_log=False):
        """ If rngSeed is set to -1, then we freeze all frozen bits to zero.

        Args:
            q (int): the base alphabet size
            length (int): the length of the U vector
            frozenSet (set): the set of frozen indices
            commonRandomnessSeed (int): seed for the generating the frozen qudits, if -1 then all are set to 1.0
        """
        self.q = q
        self.commonRandomnessSeed = commonRandomnessSeed
        self.frozenSet = sorted(frozenSet)
        self.infoSet = sorted(set([i for i in range(length) if i not in self.frozenSet]))
        self.length = length
        self.k = length - len(self.frozenSet)
        self.frozenOrInformation = self.initFrozenOrInformation()
        self.randomlyGeneratedNumbers = self.initRandomlyGeneratedNumbers()

        self.prob_list = None
        self.actual_prob = None
        self.use_log = use_log

    def initFrozenOrInformation(self):
        frozenOrInformation = np.full(self.length, uIndexType.information, dtype=uIndexType)
        frozenOrInformation[list(self.frozenSet)] = uIndexType.frozen
        return frozenOrInformation

    def initRandomlyGeneratedNumbers(self):
        if self.commonRandomnessSeed != -1:
            commonRandomnessRNG = random.Random(self.commonRandomnessSeed)
            return np.array([commonRandomnessRNG.random() for _ in range(self.length)])
        else:
            return np.full(self.length, 1.0)

    def reinitRandomlyGeneratedNumbers(self, newSeed):
        self.commonRandomnessSeed = newSeed
        self.randomlyGeneratedNumbers = self.initRandomlyGeneratedNumbers()

    def encode(self, xVectorDistribution, information):
        """Encode k information bits according to a-priori input distribution.

        Args:
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-priori entries for P(X=j) for all j in F_q
            information (numpy array of Int64): the k information bits to encode.

        Returns:
            The encoded vector (reverse polar transform of the U)
        """

        uIndex = 0
        informationVectorIndex = 0
        assert (len(xVectorDistribution) == self.length)
        assert (len(information) == self.k)

        (encodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex,
                                                                                               informationVectorIndex,
                                                                                               xVectorDistribution)

        assert (next_uIndex == len(encodedVector) == len(xVectorDistribution) == self.length)
        assert (next_informationVectorIndex == len(information) == self.k)

        return encodedVector

    def decode(self, xVectorDistribution, xyVectorDistribution):
        """Decode k information bits according to a-priori input distribution and a-posteriori input distribution.

        Args:
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-priori entries for P(X=j) for each j in F_q.
            xyVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-posteriori entries for P(X=j) for each j in F_q. That is, entry i contains P(X=j,Y=y_i) for each j in F_q.

        Returns:
            (encodedVector, information): The encoded vector (reverse polar transform of the U) and the corresponding information bits.
        """

        uIndex = 0
        informationVectorIndex = 0

        information = np.full(self.k, -1, dtype=np.int64)

        assert (len(xVectorDistribution) == len(xyVectorDistribution) == self.length)

        (encodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex,
                                                                                               informationVectorIndex,
                                                                                               xVectorDistribution,
                                                                                               xyVectorDistribution)

        assert (next_uIndex == len(encodedVector) == self.length)
        assert (next_informationVectorIndex == len(information) == self.k)

        return information

    def listDecode(self, xyVectorDistribution, frozenValues, maxListSize, check_matrix, check_value, actualInformation=None, verbosity=0):
        """List-decode k information bits according to a-priori input distribution and a-posteriori input distribution

        Args:
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-priori entries for P(X=j) for each j in F_q.
            xyVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-posteriori entries for P(X=j) for each j in F_q. That is, entry i contains P(X=j,Y=y_i) for each j in F_q.
            maxListSize: maximum list size during list-decoding

        Returns:
            (encodedVector, information): The encoded vector (reverse polar transform of the U) and the corresponding information bits.
        """
        uIndex = 0
        informationVectorIndex = 0

        informationList = np.full((maxListSize * self.q, self.k), -1, dtype=np.int64)
        frozenValuesIterator = None
        if len(frozenValues):
            frozenValuesIterator = np.nditer(frozenValues, flags=['f_index'])

        assert (len(xyVectorDistribution) == self.length)

        self.actualInformation = actualInformation
        if self.use_log:
            self.actual_prob = 0.0
            self.prob_list = np.array([0.0])
        else:
            self.actual_prob = 1.0
            self.prob_list = np.array([1.0])

        self.info_time = 0
        self.transform_time = 0
        self.encoding_time = 0

        start = timer()
        (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, finalListSize,
         originalIndicesMap, actualEncoding) = self.recursiveListDecode(informationList, uIndex, informationVectorIndex,
                                                        [xyVectorDistribution], frozenValuesIterator, inListSize=1,
                                                        maxListSize=maxListSize, actualXyVectorDistribution=xyVectorDistribution)
        end = timer()
        # print("total time: " + str(end - start))
        # print("info time rate: " + str(self.info_time / (end - start)))
        # print("total transforms time: " + str(self.transform_time))
        # print("transforms time rate: " + str(self.transform_time / (end - start)))
        # print("encoding time rate: " + str(self.encoding_time / (end - start)))

        assert (1 <= finalListSize <= maxListSize)
        assert (len(encodedVectorList) == finalListSize)
        assert (next_uIndex == len(encodedVectorList[0]) == self.length)
        assert (next_informationVectorIndex == self.k)
        assert (len(originalIndicesMap) == finalListSize)
        assert (np.count_nonzero(originalIndicesMap) == 0)

        # print(actualInformation)
        # print(informationList)
        if actualInformation is not None:
            explicit_probs, normalization = normalize([self.calc_explicit_prob(information, frozenValues, xyVectorDistribution) for information
                              in informationList[:maxListSize]], use_log=self.use_log)
            actual_explicit_prob = self.calc_explicit_prob(actualInformation, frozenValues, xyVectorDistribution)
            if self.use_log:
                actual_explicit_prob = actual_explicit_prob - normalization
            else:
                actual_explicit_prob = actual_explicit_prob / normalization

            for i, information in enumerate(informationList[:maxListSize]):
                if np.array_equal(information, actualInformation):
                    # if probList[i] != self.actualProb:
                    #     print("Actual probability wrongly calculated: " + str(probList[i]) + " != " + str(self.actualProb))
                    maxProb = max(self.prob_list)
                    if self.prob_list[i] == maxProb:
                        probResult = ProbResult.SuccessActualIsMax
                    else:
                        probResult = ProbResult.SuccessActualSmallerThanMax
                    return information, probResult

            # if verbosity:
            #     print("actual information:" + str(actualInformation))
            #     print(informationList[:maxListSize])
            #     print("number of non-unique indices in list: " + str(sum([len(np.unique(informationList[:maxListSize, i])) > 1 for i, info in enumerate(informationList[0])])))
            #     identicalPrefixSizeList = min([i for i, info in enumerate(informationList[0]) if len(np.unique(informationList[:maxListSize, i])) > 1])
            #     print("identical prefix size in list: " + str(identicalPrefixSizeList))
            #     print("nearest neighbor distance from actual in list: " + str(min([sum(guess != actualInformation) for guess in informationList[:maxListSize]])))
            #     print("identical prefix size from actual in list: " + str(min([i for i in range(identicalPrefixSizeList) if informationList[0][i] != actualInformation[i]] or [0])))
            maxProb = max(self.prob_list)
            if self.actual_prob > maxProb:
                probResult = ProbResult.FailActualLargerThanMax
            elif self.actual_prob == maxProb:
                probResult = ProbResult.FailActualIsMax
            elif self.actual_prob >= min(self.prob_list):
                probResult = ProbResult.FailActualWithinRange
            else:
                probResult = ProbResult.FailActualSmallerThanMin
            # print(probResult)
            # print("maxProb: " + str(probList.max()))
            # print("minProb: " + str(probList.min()))
            # print("actualProb: " + str(self.actualProb))
            return informationList[0], probResult

        candidateList = np.array([np.array_equal(np.matmul(information, check_matrix) % self.q, check_value) for information in informationList[:maxListSize]])
        # if verbosity:
        #     print(candidateList)
        if True in candidateList:
            for i, val in enumerate(candidateList):
                if val:
                    return informationList[i], None

        # for information in informationList[:maxListSize]:
        #     cur_check = np.matmul(information, check_matrix) % self.q
        #     if np.array_equal(cur_check, check_value):
        #         return information
        return informationList[0], None

    def calc_explicit_prob(self, information, frozenValues, xyVectorDistribution):
        guess = polarTransformOfQudits(self.q, self.mergeInfoAndFrozen(information, frozenValues))
        probs_list = [xyProb[guess[i]] for i, xyProb in enumerate(xyVectorDistribution.probs)]
        if self.use_log:
            guess_prob = sum(probs_list)
        else:
            guess_prob = np.prod(probs_list)
        return guess_prob

    def mergeInfoAndFrozen(self, actualInformation, frozenValues):
        mergedVector = np.empty(self.length, dtype=np.int)
        mergedVector[list(self.infoSet)] = actualInformation
        mergedVector[list(self.frozenSet)] = frozenValues
        return mergedVector

    def genieSingleDecodeSimulation(self, xVectorDistribution, xyVectorDistribution, trustXYProbs):
        """Pick up statistics of a single decoding run
        Args:
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-priori entries for P(X=j) for each j in F_q
            xyVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-posteriori entries for P(X=j) for each j in F_q. That is, entry i contains P(X=j,Y=y_i) for each j in F_q.
            trustXYProbs (bool): Do we trust the probabilities of U_i=s for each s in F_q given past U and all Y (we usually should), or don't we (in case we have guard bands, which can be parsed wrong, and then result in garbage probs).

        Returns:
            (decodedVector, Pe, H): a triplet of arrays. The first array is the codeword we have produced. If we trust the probabilities, entry i of the second array is the probability of error, min{P(U_i=0|U_0^{i-1} = u_0^{i-1}, Y_0^{N-1} = y_0^{N-1}), P(U_i=1|U_0^{i-1} = u_0^{i-1}, Y_0^{N-1} = y_0^{N-1})}. Entry i of the third array is the entropy, -P(U_i=0|U_0^{i-1} = u_0^{i-1}, Y_0^{N-1} = y_0^{N-1}) * log_2(P(U_i=0|U_0^{i-1} = u_0^{i-1}, Y_0^{N-1} = y_0^{N-1})) - P(U_i=1|U_0^{i-1} = u_0^{i-1}, Y_0^{N-1} = y_0^{N-1}) * log_2(P(U_i=1|U_0^{i-1} = u_0^{i-1}, Y_0^{N-1} = y_0^{N-1})). If we don't trust the probabilities, entry i of the second array is 1.0, and the third array is empty.
        """

        marginalizedUProbs = []
        uIndex = 0
        informationVectorIndex = 0
        information = []

        assert (len(xVectorDistribution) == self.length)

        (decodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex,
                                                                                               informationVectorIndex,
                                                                                               xVectorDistribution,
                                                                                               xyVectorDistribution,
                                                                                               marginalizedUProbs)

        assert (next_uIndex == len(decodedVector) == len(xVectorDistribution))
        assert (next_informationVectorIndex == len(information) == 0)
        assert (len(marginalizedUProbs) == self.length)

        if trustXYProbs:
            Pevec = np.array([min(probTuple) for probTuple in marginalizedUProbs])
            Hvec = np.array([QaryMemorylessDistribution.eta_list(probTuple) for probTuple in marginalizedUProbs])
        else:
            print("NOT YET IMPLEMENTED")
            exit(1)

        return (decodedVector, Pevec, Hvec)

    def genieSingleEncodeSimulation(self, xVectorDistribution):
        """Pick up statistics of a single encoding run.

        Args:
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-priori entries for P(X=0) and P(X=1)

        Returns:
            (encodedVector, TVvec, H): a triplet of arrays. The first array is the codeword we have produced. Entry i of the second array is the the total variation |P(U_i=0|U_0^{i-1} = u_0^{i-1})-P(U_i=1|U_0^{i-1} = u_0^{i-1})|. Entry i of the third array is the entropy, -P(U_i=0|U_0^{i-1} = u_0^{i-1} * log_2(P(U_i=0|U_0^{i-1} = u_0^{i-1}) - P(U_i=1|U_0^{i-1} = u_0^{i-1} * log_2(P(U_i=1|U_0^{i-1} = u_0^{i-1})
        """

        assert (len(xVectorDistribution) == self.length)

        marginalizedUProbs = []
        uIndex = 0
        informationVectorIndex = 0
        information = []

        assert (len(xVectorDistribution) == self.length)

        (encodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex,
                                                                                               informationVectorIndex,
                                                                                               xVectorDistribution,
                                                                                               None, marginalizedUProbs)

        assert (next_uIndex == len(encodedVector) == len(xVectorDistribution))
        assert (next_informationVectorIndex == len(information) == 0)
        assert (len(marginalizedUProbs) == self.length)

        TVvec = []
        Hvec = []

        for probTuple in marginalizedUProbs:
            print("NOT IMPLEMENTED YET")
            exit(1)

        return (encodedVector, TVvec, Hvec)

    def recursiveEncodeDecode(self, information, uIndex, informationVectorIndex, xVectorDistribution,
                              xyVectorDistribution=None, marginalizedUProbs=None):
        """Encode/decode according to supplied vector distributions.

        Args:
            information (numpy array of Int64): an array of information bits to either read from when encoding or write to when decoding
            uIndex (int): the first relevant index in the polar transformed U vector of the *whole* codeword (non-recursive)
            informationVectorIndex (int): the first relevant index in the information vector associated with the *whole* codeword (non-recursive)
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector (whose length is a function of the recursion depth) with a-priori entries for P(X=j) for all j in F_q
            xyVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector (whose length is a function of the recursion depth) with a-posteriori entries for P(X=j) for all j in F_q. A None value means we are encoding.
            marginalizedUProbs (empty array, or None): If not None, we populate (return) this array so that if xyVectorDistribution is None (encoding), then marginalizedUProbs[i][x] = P(U_i=x|U_0^{i-1} = \hat{u}_0^{i-1}). Otherwise, marginalizedUProbs[i][x] = P(U_i=x|U_0^{i-1} = \hat{u}_0^{i-1}, Y_0^{N-1} = y_0^{N-1}). For genie decoding, we will have \hat{u}_i = u_i, as the frozen set contains all indices.

        Returns:
            (encodedVector, next_uIndex, next_informationVectorIndex): the recursive encoding of the relevant part of the information vector, as well as updated values for the parameters uIndex and informationVectorIndex
        """

        # By default, we assume encoding, and add small corrections for decoding.
        encodedVector = np.full(len(xVectorDistribution), -1, dtype=np.int64)
        decoding = xyVectorDistribution is not None

        if len(xVectorDistribution) == 1:
            if self.frozenOrInformation[uIndex] == uIndexType.information:
                if decoding:
                    marginalizedVector = xyVectorDistribution.calcMarginalizedProbabilities()
                    information[informationVectorIndex] = np.argmax(marginalizedVector)
                encodedVector[0] = information[informationVectorIndex]
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex + 1
            else:
                # marginalizedVector = xVectorDistribution.calcMarginalizedProbabilities()
                # encodedVector[0] = min(
                #     np.searchsorted(np.cumsum(marginalizedVector), self.randomlyGeneratedNumbers[uIndex]),
                #     self.q - 1)
                encodedVector[0] = 0
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex

            # should we return the marginalized probabilities?
            if marginalizedUProbs is not None:
                vectorDistribution = xyVectorDistribution or xVectorDistribution
                marginalizedVector = vectorDistribution.calcMarginalizedProbabilities()
                marginalizedUProbs.append(marginalizedVector)

            return (encodedVector, next_uIndex, next_informationVectorIndex)
        else:
            xMinusVectorDistribution = xVectorDistribution.minusTransform()
            xMinusVectorDistribution.normalize()
            if decoding:
                xyMinusVectorDistribution = xyVectorDistribution.minusTransform()
                xyMinusVectorDistribution.normalize()
            else:
                xyMinusVectorDistribution = None

            (minusEncodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information,
                                                                                                        uIndex,
                                                                                                        informationVectorIndex,
                                                                                                        xMinusVectorDistribution,
                                                                                                        xyMinusVectorDistribution,
                                                                                                        marginalizedUProbs)

            xPlusVectorDistribution = xVectorDistribution.plusTransform(minusEncodedVector)
            xPlusVectorDistribution.normalize()
            if decoding:
                xyPlusVectorDistribution = xyVectorDistribution.plusTransform(minusEncodedVector)
                xyPlusVectorDistribution.normalize()
            else:
                xyPlusVectorDistribution = None

            uIndex = next_uIndex
            informationVectorIndex = next_informationVectorIndex
            (plusEncodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information,
                                                                                                       uIndex,
                                                                                                       informationVectorIndex,
                                                                                                       xPlusVectorDistribution,
                                                                                                       xyPlusVectorDistribution,
                                                                                                       marginalizedUProbs)

            halfLength = len(xVectorDistribution) // 2

            for halfi in range(halfLength):
                encodedVector[2 * halfi] = (minusEncodedVector[halfi] + plusEncodedVector[halfi]) % self.q
                encodedVector[2 * halfi + 1] = (-plusEncodedVector[halfi] + self.q) % self.q

            return (encodedVector, next_uIndex, next_informationVectorIndex)

    def recursiveListDecode(self, informationList, uIndex, informationVectorIndex,
                            xyVectorDistributionList, frozenValuesIterator=None, marginalizedUProbs=None, inListSize=1, maxListSize=1,
                            actualXyVectorDistribution=None):
        """Encode/decode according to supplied vector distributions.

        Args:
            information (numpy array of Int64): an array of information bits to either read from when encoding or write to when decoding
            uIndex (int): the first relevant index in the polar transformed U vector of the *whole* codeword (non-recursive)
            informationVectorIndex (int): the first relevant index in the information vector associated with the *whole* codeword (non-recursive)
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector (whose length is a function of the recursion depth) with a-priori entries for P(X=j) for all j in F_q
            xyVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector (whose length is a function of the recursion depth) with a-posteriori entries for P(X=j) for all j in F_q. A None value means we are encoding.
            marginalizedUProbs (empty array, or None): If not None, we populate (return) this array so that if xyVectorDistribution is None (encoding), then marginalizedUProbs[i][x] = P(U_i=x|U_0^{i-1} = \hat{u}_0^{i-1}). Otherwise, marginalizedUProbs[i][x] = P(U_i=x|U_0^{i-1} = \hat{u}_0^{i-1}, Y_0^{N-1} = y_0^{N-1}). For genie decoding, we will have \hat{u}_i = u_i, as the frozen set contains all indices.

        Returns:
            (encodedVector, next_uIndex, next_informationVectorIndex): the recursive encoding of the relevant part of the information vector, as well as updated values for the parameters uIndex and informationVectorIndex
        """
        assert (inListSize <= maxListSize)
        assert (inListSize == len(self.prob_list))
        segmentSize = len(xyVectorDistributionList[0])

        if segmentSize == 1:
            if self.frozenOrInformation[uIndex] == uIndexType.information:
                encodedVectorList = np.full((inListSize * self.q, segmentSize), -1, dtype=np.int64)
                newProbList = np.empty(inListSize * self.q, dtype=np.float)
                for i in range(inListSize):
                    information = informationList[i]
                    marginalizedVector = xyVectorDistributionList[i].calcMarginalizedProbabilities()
                    start = timer()
                    for s in range(self.q):
                        if s > 0:
                            informationList[s * inListSize + i] = information  # branch the paths q times
                        informationList[s * inListSize + i][informationVectorIndex] = s
                        encodedVectorList[s * inListSize + i][0] = s
                        if self.use_log:
                            newProbList[s * inListSize + i] = self.prob_list[i] + marginalizedVector[s]
                        else:
                            newProbList[s * inListSize + i] = self.prob_list[i] * marginalizedVector[s]
                    end = timer()
                    self.info_time += end - start
                newListSize = inListSize * self.q
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex + 1

                if newListSize > maxListSize:
                    if self.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    original_indices_map = indices_to_keep % inListSize
                    start = timer()
                    informationList[0:newListSize] = informationList[indices_to_keep]
                    informationList[newListSize:, :] = -1
                    end = timer()
                    self.info_time += end - start
                    encodedVectorList = encodedVectorList[indices_to_keep]
                    newProbList = newProbList[indices_to_keep]
                else:
                    original_indices_map = np.tile(np.arange(inListSize), self.q)

                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.use_log)

                if actualXyVectorDistribution is not None:
                    actualEncodedVector = [self.actualInformation[informationVectorIndex]]
                    if self.use_log:
                        self.actual_prob += actualXyVectorDistribution.calcMarginalizedProbabilities()[actualEncodedVector[0]] - normalizationWeight
                    else:
                        self.actual_prob *= actualXyVectorDistribution.calcMarginalizedProbabilities()[actualEncodedVector[0]] / normalizationWeight
            else:
                newListSize = inListSize
                frozenValue = frozenValuesIterator[0]
                encodedVectorList = np.full((inListSize, segmentSize), frozenValue, dtype=np.int64)
                if actualXyVectorDistribution is not None:
                    actualEncodedVector = [frozenValue]
                frozenValuesIterator.iternext()
                # encodedVectorList = np.full((inListSize, segmentSize), 0, dtype=np.int64)
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex
                original_indices_map = np.arange(inListSize)
                if self.use_log:
                    self.prob_list, normalizationWeight = normalize([prob + xyVectorDistributionList[i].calcMarginalizedProbabilities()[frozenValue] for i, prob in enumerate(self.prob_list)], use_log=self.use_log)
                    self.actual_prob += actualXyVectorDistribution.calcMarginalizedProbabilities()[frozenValue] - normalizationWeight
                else:
                    self.prob_list, normalizationWeight = normalize([prob * xyVectorDistributionList[i].calcMarginalizedProbabilities()[frozenValue] for i, prob in enumerate(self.prob_list)], use_log=self.use_log)
                    self.actual_prob *= actualXyVectorDistribution.calcMarginalizedProbabilities()[frozenValue] / normalizationWeight

            return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                    original_indices_map, actualEncodedVector)
        else:
            numInfoBits = np.sum((self.frozenOrInformation[uIndex:uIndex + segmentSize] == uIndexType.information))

            # Rate-0 node
            if numInfoBits == 0:
                frozenVector = np.empty(segmentSize, dtype=np.int64)
                for i in range(segmentSize):
                    frozenVector[i] = frozenValuesIterator[0]
                    frozenValuesIterator.iternext()
                encodedVector = polarTransformOfQudits(self.q, frozenVector)
                encodedVectorList = [encodedVector] * inListSize
                if self.use_log:
                    newProbList = [prob + np.sum([xyVectorDistribution.probs[i, encodedVector[i]] for i in range(segmentSize)]) for xyVectorDistribution, prob in zip(xyVectorDistributionList, self.prob_list)]
                    self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.use_log)
                    self.actual_prob += np.sum([actualXyVectorDistribution.probs[i, encodedVector[i]] for i in range(segmentSize)]) - normalizationWeight
                else:
                    newProbList = [prob * np.product([xyVectorDistribution.probs[i, encodedVector[i]] for i in range(segmentSize)]) for xyVectorDistribution, prob in zip(xyVectorDistributionList, self.prob_list)]
                    self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.use_log)
                    self.actual_prob *= np.product([actualXyVectorDistribution.probs[i, encodedVector[i]] for i in
                                                range(segmentSize)]) / normalizationWeight

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex
                newListSize = inListSize
                originalIndicesMap = np.arange(inListSize)
                actualEncodedVector = encodedVector
                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                 originalIndicesMap, actualEncodedVector)

            # Rep node
            if numInfoBits == 1:
                k = np.where(self.frozenOrInformation[uIndex:uIndex + segmentSize] == uIndexType.information)[0][0]
                inputVectorSplits = np.empty((self.q, segmentSize), dtype=np.int64)
                for i in range(segmentSize):
                    if i != k:
                        inputVectorSplits[:, i] = frozenValuesIterator[0]
                        frozenValuesIterator.iternext()
                    else:
                        inputVectorSplits[:, i] = np.arange(self.q)
                encodedVectorSplits = np.array([polarTransformOfQudits(self.q, v) for v in inputVectorSplits])

                newProbList = np.empty(inListSize * self.q, dtype=np.float)
                for i in range(inListSize):
                    information = informationList[i]
                    start = timer()
                    for s in range(self.q):
                        if s > 0:
                            informationList[s * inListSize + i] = information  # branch the paths q times
                        informationList[s * inListSize + i][informationVectorIndex] = s
                        if self.use_log:
                            newProbList[s * inListSize + i] = self.prob_list[i] + np.sum([xyVectorDistributionList[i].probs[j, encodedVectorSplits[s, j]] for j in range(segmentSize)])
                        else:
                            newProbList[s * inListSize + i] = self.prob_list[i] * np.product([xyVectorDistributionList[i].probs[j, encodedVectorSplits[s, j]] for j in range(segmentSize)])
                    end = timer()
                    self.info_time += end - start
                newListSize = inListSize * self.q

                if newListSize > maxListSize:
                    if self.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    original_indices_map = indices_to_keep % inListSize
                    start = timer()
                    informationList[0:newListSize] = informationList[indices_to_keep]
                    informationList[newListSize:, :] = -1
                    end = timer()
                    self.info_time += end - start
                    encodedVectorList = encodedVectorSplits[indices_to_keep // inListSize]
                    newProbList = newProbList[indices_to_keep]
                else:
                    encodedVectorList = np.repeat(encodedVectorSplits, inListSize, axis=0)
                    original_indices_map = np.tile(np.arange(inListSize), self.q)

                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.use_log)

                if actualXyVectorDistribution is not None:
                    actualEncodedVector = encodedVectorSplits[self.actualInformation[informationVectorIndex]]
                    if self.use_log:
                        self.actual_prob += np.sum([actualXyVectorDistribution.probs[i, actualEncodedVector[i]] for i in range(segmentSize)]) - normalizationWeight
                    else:
                        self.actual_prob *= np.product([actualXyVectorDistribution.probs[i, actualEncodedVector[i]] for i in range(segmentSize)]) / normalizationWeight

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex + 1
                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                        original_indices_map, actualEncodedVector)

            # Rate-1 node
            if numInfoBits == segmentSize:
                fork_size = self.q ** 2
                encodedVectorList = np.empty((inListSize * fork_size, segmentSize), dtype=np.int64)
                newProbList = np.empty(inListSize * fork_size, dtype=np.float)
                for i in range(inListSize):
                    # Pick the 2 least reliable indices
                    [j1, j2] = self.pickLeastReliableIndices(xyVectorDistributionList[i].probs, 2)
                    # Fork there
                    encodedVectorList[fork_size*i : fork_size*(i + 1)], newProbList[fork_size*i : fork_size*(i + 1)] = self.forkIndices(self.prob_list[i], xyVectorDistributionList[i].probs, segmentSize, [j1, j2])
                newListSize = inListSize * fork_size

                # Prune
                if newListSize > maxListSize:
                    if self.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    encodedVectorList = encodedVectorList[indices_to_keep]
                    newProbList = newProbList[indices_to_keep]
                    original_indices_map = indices_to_keep // fork_size
                else:
                    original_indices_map = np.repeat(np.arange(inListSize), fork_size)

                # Normalize
                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.use_log)
                if actualXyVectorDistribution is not None:
                    actualEncodedVector = polarTransformOfQudits(self.q, self.actualInformation[informationVectorIndex:informationVectorIndex+segmentSize])
                    if self.use_log:
                        self.actual_prob += np.sum([actualXyVectorDistribution.probs[i, actualEncodedVector[i]] for i in
                                                    range(segmentSize)]) - normalizationWeight
                    else:
                        self.actual_prob *= np.product([actualXyVectorDistribution.probs[i, actualEncodedVector[i]] for i in
                                                    range(segmentSize)]) / normalizationWeight

                # Update informationList
                start = timer()
                informationList[0:newListSize] = informationList[original_indices_map]
                informationList[0:newListSize, informationVectorIndex:informationVectorIndex+segmentSize] = [polarTransformOfQudits(self.q, encodedVector) for encodedVector in encodedVectorList]
                informationList[newListSize:] = -1
                end = timer()
                self.info_time += end - start

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex + segmentSize

                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                        original_indices_map, actualEncodedVector)

            # SPC node
            if numInfoBits == segmentSize - 1:
                num_forked_indices = 3

                fork_size = self.q ** num_forked_indices
                encodedVectorList = np.empty((inListSize * fork_size, segmentSize), dtype=np.int64)
                newProbList = np.empty(inListSize * fork_size, dtype=np.float)
                frozenValue = frozenValuesIterator[0]
                for i in range(inListSize):
                    # Pick the least reliable indices
                    leastReliableIndices = self.pickLeastReliableIndices(xyVectorDistributionList[i].probs, num_forked_indices + 1)
                    # Fork there
                    encodedVectorList[fork_size*i: fork_size*(i + 1)], newProbList[fork_size*i: fork_size*(i + 1)] = self.forkIndicesSpc(self.prob_list[i], xyVectorDistributionList[i].probs, segmentSize, leastReliableIndices, frozenValue)
                frozenValuesIterator.iternext()
                newListSize = inListSize * fork_size

                # Prune
                if newListSize > maxListSize:
                    if self.use_log:
                        newListSize = min(newListSize - np.isneginf(newProbList).sum(), maxListSize)
                    else:
                        newListSize = min(np.count_nonzero(newProbList), maxListSize)
                    indices_to_keep = np.argpartition(newProbList, -newListSize)[-newListSize:]
                    encodedVectorList = encodedVectorList[indices_to_keep]
                    newProbList = newProbList[indices_to_keep]
                    original_indices_map = indices_to_keep // fork_size
                else:
                    original_indices_map = np.repeat(np.arange(inListSize), fork_size)

                # Normalize
                self.prob_list, normalizationWeight = normalize(newProbList, use_log=self.use_log)
                if actualXyVectorDistribution is not None:
                    actualEncodedVector = polarTransformOfQudits(self.q, np.concatenate(([frozenValue], self.actualInformation[informationVectorIndex:informationVectorIndex+segmentSize-1]), axis=None))
                    if self.use_log:
                        self.actual_prob += np.sum([actualXyVectorDistribution.probs[i, actualEncodedVector[i]] for i in
                                                    range(segmentSize)]) - normalizationWeight
                    else:
                        self.actual_prob *= np.product([actualXyVectorDistribution.probs[i, actualEncodedVector[i]] for i in
                                                    range(segmentSize)]) / normalizationWeight

                # Update informationList
                start = timer()
                informationList[0:newListSize] = informationList[original_indices_map]
                informationList[0:newListSize, informationVectorIndex:informationVectorIndex+segmentSize-1] = [polarTransformOfQudits(self.q, encodedVector)[1:] for encodedVector in encodedVectorList]
                informationList[newListSize:] = -1
                end = timer()
                self.info_time += end - start

                next_uIndex = uIndex + segmentSize
                next_informationVectorIndex = informationVectorIndex + segmentSize - 1

                return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                        original_indices_map, actualEncodedVector)

            start = timer()
            xyMinusVectorDistributionList = []
            for i in range(inListSize):
                xyMinusVectorDistribution = xyVectorDistributionList[i].minusTransform()
                xyMinusVectorDistribution.normalize()
                xyMinusVectorDistributionList.append(xyMinusVectorDistribution)
            # xyMinusVectorDistributionList, xyMinusNormalizationVector = normalizeDistList(xyMinusVectorDistributionList)
            end = timer()
            self.transform_time += end-start
            if actualXyVectorDistribution is not None:
                actualXyMinusVectorDistribution = actualXyVectorDistribution.minusTransform()
                actualXyMinusVectorDistribution.normalize()
                # actualXyMinusVectorDistribution.normalize(xyMinusNormalizationVector)

            (minusInformationList, minusEncodedVectorList, next_uIndex, next_informationVectorIndex, minusListSize,
             minusOriginalIndicesMap, minusActualEncodedVector) = self.recursiveListDecode(informationList, uIndex, informationVectorIndex,
                                                                 xyMinusVectorDistributionList, frozenValuesIterator, marginalizedUProbs,
                                                                 inListSize, maxListSize, actualXyVectorDistribution=actualXyMinusVectorDistribution)

            start = timer()
            xyPlusVectorDistributionList = []
            for i in range(minusListSize):
                origI = minusOriginalIndicesMap[i]

                xyPlusVectorDistribution = xyVectorDistributionList[origI].plusTransform(minusEncodedVectorList[i])
                xyPlusVectorDistribution.normalize()
                xyPlusVectorDistributionList.append(xyPlusVectorDistribution)
            # xyPlusVectorDistributionList, xyPlusNormalizationVector = normalizeDistList(xyPlusVectorDistributionList)
            end = timer()
            self.transform_time += end-start
            if actualXyVectorDistribution is not None:
                actualXyPlusVectorDistribution = actualXyVectorDistribution.plusTransform(minusActualEncodedVector)
                actualXyPlusVectorDistribution.normalize()
                # actualXyPlusVectorDistribution.normalize(xyPlusNormalizationVector)

            uIndex = next_uIndex
            informationVectorIndex = next_informationVectorIndex
            (plusInformationList, plusEncodedVectorList, next_uIndex, next_informationVectorIndex, plusListSize,
             plusOriginalIndicesMap, plusActualEncodedVector) = self.recursiveListDecode(minusInformationList, uIndex, informationVectorIndex,
                                                                xyPlusVectorDistributionList, frozenValuesIterator, marginalizedUProbs,
                                                                minusListSize, maxListSize, actualXyVectorDistribution=actualXyPlusVectorDistribution)

            newListSize = plusListSize

            encodedVectorList = np.full((newListSize, segmentSize), -1, dtype=np.int64)
            # halfLength = segmentSize // 2

            start = timer()
            for i in range(newListSize):
                minusI = plusOriginalIndicesMap[i]
                encodedVectorList[i][::2] = minusEncodedVectorList[minusI] + plusEncodedVectorList[i]
                encodedVectorList[i][1::2] = -plusEncodedVectorList[i]
                encodedVectorList[i] %= self.q
                # for halfi in range(halfLength):
                #     encodedVectorList[i][2 * halfi] = (minusEncodedVectorList[minusI][halfi] + plusEncodedVectorList[i][
                #         halfi]) % self.q
                #     encodedVectorList[i][2 * halfi + 1] = (-plusEncodedVectorList[i][halfi] + self.q) % self.q
            end = timer()
            self.encoding_time += end-start

            if actualXyVectorDistribution is not None:
                actualEncodedVector = np.full(segmentSize, -1, dtype=np.int64)
                actualEncodedVector[::2] = np.add(minusActualEncodedVector, plusActualEncodedVector)
                actualEncodedVector[1::2] = np.array(plusActualEncodedVector) * (-1)
                actualEncodedVector %= self.q
                # for halfi in range(halfLength):
                #     actualEncodedVector[2 * halfi] = (minusActualEncodedVector[halfi] + plusActualEncodedVector[
                #         halfi]) % self.q
                #     actualEncodedVector[2 * halfi + 1] = (-plusActualEncodedVector[halfi] + self.q) % self.q

            originalIndicesMap = minusOriginalIndicesMap[plusOriginalIndicesMap]

            return (plusInformationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize,
                    originalIndicesMap, actualEncodedVector)

    def pickLeastReliableIndices(self, xyDist, numUnreliableIndices):
        scores = [self.reliability(cur_probs) for cur_probs in xyDist]
        return np.argpartition(scores, range(-numUnreliableIndices, 1))[-numUnreliableIndices:]

    def reliability(self, probs):
        max_probs = np.partition(probs, -2)[-2:]
        if self.use_log:
            return max_probs[0] - max_probs[1]
        else:
            return max_probs[0] / max_probs[1]

    def forkIndices(self, cur_prob, xyDist, segmentSize, indicesToFork):
        numIndicesToFork = len(indicesToFork)
        if numIndicesToFork != 2:
            raise "No support for number of indices to fork: " + str(numIndicesToFork)

        num_forks = self.q ** numIndicesToFork
        forks = np.empty((num_forks, segmentSize), dtype=np.int64)
        constant_indices_mask = np.ones(segmentSize, dtype=bool)
        constant_indices_mask[indicesToFork] = False

        forks[:, constant_indices_mask] = np.argmax(xyDist[constant_indices_mask], axis=1)
        forks[:, indicesToFork] = list(itertools.product(np.arange(self.q), repeat=numIndicesToFork))

        fork_prob_combinations = list(itertools.product(xyDist[indicesToFork[0]], xyDist[indicesToFork[1]]))
        if self.use_log:
            base_prob = cur_prob + sum(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.sum(fork_prob_combinations, axis=1) + base_prob
        else:
            base_prob = cur_prob * np.product(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.product(fork_prob_combinations, axis=1) * base_prob

        return forks, prob_list

    def forkIndicesSpc(self, cur_prob, xyDist, segmentSize, leastReliableIndices, frozenValue):
        numIndicesToFork = len(leastReliableIndices) - 1
        dependentIndex = leastReliableIndices[-1]
        indicesToFork = leastReliableIndices[:-1]
        # dependentIndex = leastReliableIndices[0]
        # indicesToFork = leastReliableIndices[1:]

        num_forks = self.q ** numIndicesToFork
        forks = np.empty((num_forks, segmentSize), dtype=np.int64)
        constant_indices_mask = np.ones(segmentSize, dtype=bool)
        constant_indices_mask[leastReliableIndices] = False

        forked_values = np.argmax(xyDist[constant_indices_mask], axis=1)
        forks[:, constant_indices_mask] = forked_values
        base_frozen_delta = (frozenValue - sum(forked_values)) % self.q
        forks[:, indicesToFork] = list(itertools.product(np.arange(self.q), repeat=numIndicesToFork))
        forks[:, dependentIndex] = np.mod(
            (base_frozen_delta - np.sum(forks[:, indicesToFork], axis=1)), self.q)

        fork_prob_combinations = np.array([[xyDist[i, fork[i]] for i in leastReliableIndices] for fork in forks])
        if self.use_log:
            base_prob = cur_prob + sum(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.sum(fork_prob_combinations, axis=1) + base_prob
        else:
            base_prob = cur_prob * np.product(np.max(xyDist[constant_indices_mask], axis=1))
            prob_list = np.product(fork_prob_combinations, axis=1) * base_prob

        return forks, prob_list

    def calculate_syndrome_and_complement(self, u_message):
        y = polarTransformOfQudits(self.q, u_message)

        w = np.copy(y)
        w[list(self.infoSet)] = 0
        w[list(self.frozenSet)] *= self.q - 1
        w[list(self.frozenSet)] %= self.q

        u = y
        u[list(self.frozenSet)] = 0

        return w, u

    def get_message_info_bits(self, u_message):
        return u_message[list(self.infoSet)]

    def get_message_frozen_bits(self, u_message):
        return u_message[list(self.frozenSet)]

    def ir(self,  a, b, make_xyVectorDistribution, list_size=1, check_size=0, verbosity=0):
        w, u = self.calculate_syndrome_and_complement(a)
        a_key = self.get_message_info_bits(u)
        x_b = b
        # x_b = np.mod(np.add(b, polarTransformOfQudits(self.q, w)), self.q)
        frozen_values = (self.get_message_frozen_bits(w) * (self.q-1)) % self.q
        # print(w)
        # print(self.frozenSet)
        # print(frozen_values)

        # if list_size == 1:
        #     b_key = self.decode(xVectorDistribution, make_xyVectorDistribution(x_b))
        # else:
        check_matrix = np.random.choice(range(self.q), (self.k, check_size))
        check_value = np.matmul(a_key, check_matrix) % self.q
        b_key, prob_result = self.listDecode(make_xyVectorDistribution(x_b), frozenValues=frozen_values, maxListSize=list_size, check_matrix=check_matrix, check_value=check_value, actualInformation=a_key, verbosity=verbosity)

        return a_key, b_key, prob_result

    def ir2(self, a, b, xVectorDistribution, make_xyVectorDistribution, maxListSize, check_size, verbosity=0):
        a_key = polarTransformOfQudits(self.q, a)[np.array(list(self.infoSet))]
        check_matrix = np.random.choice(range(self.q), (self.k, check_size))
        check_value = np.matmul(a_key, check_matrix) % self.q
        b_key, prob_result = self.listDecode(make_xyVectorDistribution(b), maxListSize, check_matrix, check_value, verbosity=verbosity)
        return a_key, b_key, prob_result

def normalize(prob_list, use_log=False):
    maxProb = np.max(prob_list)
    if use_log:
        return prob_list - maxProb, maxProb
    else:
        return prob_list / maxProb, maxProb

def calcNormalizationVector(dist_list):
    segment_size = len(dist_list[0].probs)
    normalization = np.zeros(segment_size)
    for i in range(segment_size):
        normalization[i] = max([dist.probs[i].max(axis=0) for dist in dist_list])
    return normalization

def normalizeDistList(dist_list):
    normalization_vector = calcNormalizationVector(dist_list)
    for dist in dist_list:
        dist.normalize(normalization_vector)
    return dist_list, normalization_vector

def irSimulation(q, length, simulateChannel, make_xyVectorDistribution, numberOfTrials,
                  frozenSet, maxListSize=1, checkSize=0, commonRandomnessSeed=1, randomInformationSeed=1, use_log=False, verbosity=0, ir_version=1):
    badKeys = 0
    badSymbols = 0
    encDec = QaryPolarEncoderDecoder(q, length, frozenSet, commonRandomnessSeed, use_log=use_log)
    informationRNG = random.Random(randomInformationSeed)
    probResultList = []

    # Note that we set a random seed, which is in charge of both setting the information bits as well as the channel output.
    for t in range(numberOfTrials):
        a = informationRNG.choices(range(0, q), k=encDec.length)
        b = simulateChannel(a)

        if ir_version == 1:
            a_key, b_key, prob_result = encDec.ir(a, b, make_xyVectorDistribution, maxListSize, checkSize, verbosity=verbosity)
        if ir_version == 2:
            a_key, b_key, prob_result = encDec.ir2(a, b, make_xyVectorDistribution, maxListSize, checkSize, verbosity=verbosity)

        probResultList.append(prob_result)

        if not np.array_equal(a_key, b_key):
            badKeys += 1
            badSymbols += hamming(a_key, b_key)
            # if verbosity > 0:
            #     s = "Bad key\n"
            #     s += "Alice's key:\t" + str(a_key)
            #     s += "\nBob's key:  \t" + str(b_key)
            #     print(s)
        # else:
        #     if verbosity > 0:
        #         print("Good key")

    assert(len(a_key) == length - len(frozenSet))
    # rate = (math.log(q, 2)*len(a_key) - math.log(maxListSize, 2))/length
    rate = (math.log2(q)*len(a_key) - math.log2(maxListSize))/length
    frame_error_prob = badKeys / numberOfTrials
    symbol_error_prob = badSymbols / (numberOfTrials * encDec.length)

    if verbosity:
        print("Rate: ", rate)
        print("Frame error probability = ", badKeys, "/", numberOfTrials, " = ", frame_error_prob)
        print("Symbol error probability = ", badSymbols, "/ (", numberOfTrials, " * ", encDec.length, ") = ", symbol_error_prob)

    return frame_error_prob, symbol_error_prob, rate, probResultList

def hamming(x, y):
    return sum(x != y)

def encodeDecodeSimulation(q, length, make_xVectorDistribution, make_codeword, simulateChannel,
                           make_xyVectorDistribution, numberOfTrials, frozenSet, commonRandomnessSeed=1,
                           randomInformationSeed=1, verbosity=0):
    """Run a polar encoder and a corresponding decoder (SC, not SCL).

    Args:
       q: the base alphabet size
       length (int): the number of indices in the polar transformed vector
       make_xVectorDistribution (function): return xVectorDistribution, and takes no arguments
       make_codeword (function): make a codeword out of the encodedVector (for example, by doing nothing, or by adding guard bands)
       simulateChannel (function): transforms a codeword to a received word, using the current state of the random number generator
       make_xyVectorDistribution (function): return xyVectorDistribution, as a function of the received word
       frozenSet (set): the set of (dynamically) frozen indices
       commonRandomnessSeed (int): the seed used for defining the encoder/decoder common randomness
       randomInformationSeed (int): the seed used to create the random information to be encoded
    """

    misdecodedWords = 0

    xVectorDistribution = make_xVectorDistribution()

    encDec = QaryPolarEncoderDecoder(q, length, frozenSet, commonRandomnessSeed)

    informationRNG = random.Random(randomInformationSeed)

    # Note that we set a random seed, which is in charge of both setting the information bits as well as the channel output.
    for t in range(numberOfTrials):
        information = informationRNG.choices(range(0, q), k=encDec.k)
        encodedVector = encDec.encode(xVectorDistribution, information)

        codeword = make_codeword(encodedVector)

        receivedWord = simulateChannel(codeword)
        xyVectorDistribution = make_xyVectorDistribution(receivedWord)

        decodedInformation = encDec.decode(xVectorDistribution, xyVectorDistribution)

        if not np.array_equal(information, decodedInformation):
            misdecodedWords += 1
            # if verbosity > 0:
            #     s = str(t) + ") error, transmitted information:\n" + str(information)
            #     s += "\ndecoded information:\n" + str(decodedInformation)
            #     s += "\nencoded vector before guard bands added:\n" + str(encodedVector)
            #     s += "\ncodeword:\n" + str(codeword)
            #     s += "\nreceived word:\n" + str(receivedWord)
            #     print(s)

    print("Error probability = ", misdecodedWords, "/", numberOfTrials, " = ", misdecodedWords / numberOfTrials)


def encodeListDecodeSimulation(q, length, make_xVectorDistribution, make_codeword, simulateChannel,
                               make_xyVectorDistribution, numberOfTrials, frozenSet, maxListSize, checkSize,
                               commonRandomnessSeed=1, randomInformationSeed=1, verbosity=0):
    """Run a polar encoder and a corresponding decoder (SC, not SCL)

    Args:
       q: the base alphabet size
       length (int): the number of indices in the polar transformed vector
       make_xVectorDistribution (function): return xVectorDistribution, and takes no arguments
       make_codeword (function): make a codeword out of the encodedVector (for example, by doing nothing, or by adding guard bands)
       simulateChannel (function): transforms a codeword to a received word, using the current state of the random number generator
       make_xyVectorDistribution (function): return xyVectorDistribution, as a function of the received word
       frozenSet (set): the set of (dynamically) frozen indices
       commonRandomnessSeed (int): the seed used for defining the encoder/decoder common randomness
       randomInformationSeed (int): the seed used to create the random information to be encoded
    """

    misdecodedWords = 0

    xVectorDistribution = make_xVectorDistribution()

    encDec = QaryPolarEncoderDecoder(q, length, frozenSet, commonRandomnessSeed)

    informationRNG = random.Random(randomInformationSeed)

    # Note that we set a random seed, which is in charge of both setting the information bits as well as the channel output.
    for t in range(numberOfTrials):
        information = informationRNG.choices(range(0, q), k=encDec.k)
        encodedVector = encDec.encode(xVectorDistribution, information)

        codeword = make_codeword(encodedVector)

        receivedWord = simulateChannel(codeword)
        xyVectorDistribution = make_xyVectorDistribution(receivedWord)

        check_matrix = np.random.choice(range(q), (encDec.k, checkSize))
        check_value = np.matmul(information, check_matrix) % q

        decodedInformation = encDec.listDecode(xyVectorDistribution, maxListSize, check_matrix, check_value, information, verbosity=verbosity)

        if not np.array_equal(information, decodedInformation):
            misdecodedWords += 1
            # if verbosity > 0:
            #     s = str(t) + ") error, transmitted information:\n" + str(information)
            #     s += "\ndecoded information:\n" + str(decodedInformation)
            #     s += "\nencoded vector before guard bands added:\n" + str(encodedVector)
            #     s += "\ncodeword:\n" + str(codeword)
            #     s += "\nreceived word:\n" + str(receivedWord)
            #     print(s)

    print("Error probability = ", misdecodedWords, "/", numberOfTrials, " = ", misdecodedWords / numberOfTrials)


def genieEncodeDecodeSimulation(length, make_xVectorDistribution, make_codeword, simulateChannel,
                                make_xyVectorDistribution, numberOfTrials, errorUpperBoundForFrozenSet, genieSeed,
                                trustXYProbs=True, filename=None):
    """Run a genie encoder and corresponding decoder, and return frozen set.

    Args:
       length (int): the number of indices in the polar transformed vector
       make_xVectorDistribution (function): return xVectorDistribution, and takes no arguments
       make_codeword (function): make a codeword out of the encodedVector (for example, by doing nothing, or by adding guard bands)
       simulateChannel (function): transforms a codeword to a received word, using the current state of the random number generator
       make_xyVectorDistribution (function): return xyVectorDistribution, as a function of the received word
       numberOfTrials (int): number of Monte-Carlo simulations
       errorUpperBoundForFrozenSet (float): choose a frozen set that will result in decoding error not more than this variable
       genieSeed (int): the seed used by the genie to have different encoding/decoding common randomness in each run
       trustXYProbs (bool): Do we trust the probabilities of U_i=j for all j in F_q given past U and all Y (we usually should), or don't we (in case we have guard bands, which can be parsed wrong, and then result in garbage probs).
    """

    xVectorDistribution = make_xVectorDistribution()

    TVvec = None
    HEncvec = None
    HDecvec = None

    encDec = QaryPolarEncoderDecoder(length, set(range(length)), 0)
    genieSingleRunSeedRNG = random.Random(genieSeed)

    for trialNumber in range(numberOfTrials):
        genieSingleRunSeed = genieSingleRunSeedRNG.randint(1, 1000000)
        encDec.reinitRandomlyGeneratedNumbers(genieSingleRunSeed)

        encodedVector, TVvecTemp, HencvecTemp = encDec.genieSingleEncodeSimulation(xVectorDistribution)

        codeword = make_codeword(encodedVector)

        receivedWord = simulateChannel(codeword)

        xyVectorDistribution = make_xyVectorDistribution(receivedWord)

        (decodedVector, PevecTemp, HdecvecTemp) = encDec.genieSingleDecodeSimulation(xVectorDistribution,
                                                                                     xyVectorDistribution, trustXYProbs)

        if TVvec is None:
            TVvec = TVvecTemp
            Pevec = PevecTemp
            HEncvec = HencvecTemp
            HDecvec = HdecvecTemp
        else:
            assert (len(TVvec) == len(TVvecTemp))
            TVvec = np.add(TVvec, TVvecTemp)
            Pevec = np.add(Pevec, PevecTemp)
            HEncvec = np.add(HEncvec, HencvecTemp)
            if trustXYProbs:
                HDecvec = np.add(HDecvec, HdecvecTemp)

    TVvec /= numberOfTrials
    Pevec /= numberOfTrials
    HEncvec /= numberOfTrials
    HEncsum = np.sum(HEncvec)
    if trustXYProbs:
        HDecvec /= numberOfTrials
        HDecsum = np.sum(HDecvec)

    print("TVVec = ", TVvec)
    print("pevec = ", Pevec)
    print("HEncvec = ", HEncvec)
    if trustXYProbs:
        print("HDecvec = ", HDecvec)
    print("Normalized HEncsum = ", HEncsum / len(HEncvec))
    if trustXYProbs:
        print("Normalized HDecsum = ", HDecsum / len(HDecvec))

    frozenSet = frozenSetFromTVAndPe(TVvec, Pevec, errorUpperBoundForFrozenSet)
    print("code rate = ", (len(TVvec) - len(frozenSet)) / len(codeword))
    print("codeword length = ", len(codeword))

    if filename is not None:
        f = open(filename, "w")
        s = "* " + ' '.join(sys.argv[:]) + "\n"
        f.write(s)

        for i in frozenSet:
            f.write(str(i))
            f.write("\n")

        s = "** number of trials = " + str(numberOfTrials) + "\n"
        f.write(s)
        s = "* (TotalVariation+errorProbability) * (number of trials)" + "\n"
        f.write(s)

        for i in range(len(TVvec)):
            s = "*** " + str(i) + " " + str((TVvec[i] + Pevec[i]) * numberOfTrials) + "\n"
            f.write(s)

        f.close()

    return frozenSet


def polarTransformOfQudits(q, xvec):
    # print("xvec =", xvec)
    if len(xvec) == 1:
        return xvec
    else:
        if len(xvec) % 2 != 0:
            print(xvec)
        assert (len(xvec) % 2 == 0)

        vfirst = []
        vsecond = []
        for i in range((len(xvec) // 2)):
            vfirst.append((xvec[2 * i] + xvec[2 * i + 1]) % q)
            vsecond.append((q - xvec[2 * i + 1]) % q)

        ufirst = polarTransformOfQudits(q, vfirst)
        usecond = polarTransformOfQudits(q, vsecond)

        return np.concatenate((ufirst, usecond))


def frozenSetFromTVAndPe(TVvec, Pevec, errorUpperBoundForFrozenSet=None, numInfoIndices=None, verbosity=False):
    TVPlusPeVec = np.add(TVvec, Pevec)
    N = len(TVPlusPeVec)
    sortedIndices = sorted(range(N), key=lambda k: TVPlusPeVec[k])

    if numInfoIndices is None:
        errorSum = 0.0
        indexInSortedIndicesArray = -1
        while errorSum < errorUpperBoundForFrozenSet and indexInSortedIndicesArray + 1 < len(TVPlusPeVec):
            i = sortedIndices[indexInSortedIndicesArray + 1]
            if TVPlusPeVec[i] + errorSum <= errorUpperBoundForFrozenSet:
                errorSum += TVPlusPeVec[i]
                indexInSortedIndicesArray += 1
            else:
                break
    else:
        indexInSortedIndicesArray = numInfoIndices

    frozen_mask = np.zeros(len(TVPlusPeVec), dtype=bool)
    frozen_mask[sortedIndices[indexInSortedIndicesArray + 1:]] = True
    segmentSize = N
    while segmentSize > 1:
        for i in range(N // segmentSize):
            if sum(frozen_mask[i * segmentSize: (i + 1) * segmentSize]) == 1 and frozen_mask[i * segmentSize] is False:
                frozen_mask[i * segmentSize] = True
                frozen_mask[i * segmentSize + 1: (i + 1) * segmentSize] = False
        segmentSize //= 2
    frozenSet = set([i for i, is_frozen in enumerate(frozen_mask) if is_frozen])

    if verbosity:
        print("frozen set =", frozenSet)
        if numInfoIndices is None:
            print("fraction of info indices =", 1.0 - len(frozenSet) / len(TVPlusPeVec))

    return frozenSet

def make_cmp_function(TVPlusPeVec):
    def cmp_function(a, b):
        a_prefix, a_suffix_len = prefix(a)
        b_prefix, b_suffix_len = prefix(b)
        if a_suffix_len > b_suffix_len and b >> (a_suffix_len - b_suffix_len) == a_prefix:
            print("a")
            return 1
        if a_suffix_len < b_suffix_len and a >> (b_suffix_len - a_suffix_len) == b_prefix:
            print("b")
            return -1
        else:
            print("s")
            return math.sign(TVPlusPeVec[a], TVPlusPeVec[b])
    return cmp_function

def prefix(x):
    i = 0
    while x % 2 == 0:
        x >> 2
        i+=1
    return x, i