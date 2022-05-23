import random
import sys
from enum import Enum

import numpy as np

from ScalarDistributions import QaryMemorylessDistribution


class uIndexType(Enum):
    frozen = 0
    information = 1


class QaryPolarEncoderDecoder:
    def __init__(self, q, length, frozenSet, commonRandomnessSeed):
        """ If rngSeed is set to -1, then we freeze all frozen bits to zero.
        Args:
            q (int): the base alphabet size
            length (int): the length of the U vector
            frozenSet (set): the set of frozen indices
            commonRandomnessSeed (int): seed for the generating the frozen qudits, if -1 then all are set to 1.0
        """
        self.q = q
        self.commonRandomnessSeed = commonRandomnessSeed
        self.frozenSet = frozenSet
        self.infoSet = set([i for i in range(length) if i not in self.frozenSet])
        self.length = length

        self.k = length - len(self.frozenSet)
        self.frozenOrInformation = self.initFrozenOrInformation()
        self.randomlyGeneratedNumbers = self.initRandomlyGeneratedNumbers()

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
        """Encode k information bits according to a-priori input distribution

        Args:
            xVectorDistribution (VectorDistribution): in a memoryless setting, this is essentially a vector with a-priori entries for P(X=j) for all j in F_q

            information (numpy array of Int64): the k information bits to encode

        Returns:
            The encoded vector (reverse polar transform of the U)
        """

        uIndex = 0
        informationVectorIndex = 0
        assert (len(xVectorDistribution) == self.length)
        assert (len(information) == self.k)


        (encodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex, informationVectorIndex, xVectorDistribution)

        assert( next_uIndex == len(encodedVector) == len(xVectorDistribution)  == self.length )
        assert( next_informationVectorIndex == len(information)  == self.k )

        return encodedVector

    def decode(self, xVectorDistribution, xyVectorDistribution):
        """Decode k information bits according to a-priori input distribution and a-posteriori input distribution

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

        (encodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex, informationVectorIndex, xVectorDistribution, xyVectorDistribution)

        assert( next_uIndex == len(encodedVector) == self.length )
        assert( next_informationVectorIndex == len(information) == self.k )

        return information

    def listDecode(self, xVectorDistribution, xyVectorDistribution, maxListSize, check_matrix, check):
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

        assert (len(xVectorDistribution) == len(xyVectorDistribution) == self.length)

        (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, finalListSize, originalIndicesMap) = self.recursiveListDecode(informationList, uIndex, informationVectorIndex, [xVectorDistribution], [xyVectorDistribution], inListSize=1, maxListSize=maxListSize)

        assert(1 <= finalListSize <= maxListSize)
        assert(len(encodedVectorList) == finalListSize)
        assert( next_uIndex == len(encodedVectorList[0]) == self.length )
        assert( next_informationVectorIndex == self.k )
        assert(len(originalIndicesMap) == finalListSize)
        assert(np.count_nonzero(originalIndicesMap) == 0)

        for information, encodedVector in zip(informationList, encodedVectorList):
            cur_check = np.matmul(information, check_matrix) % self.q
            if np.array_equal(cur_check, check):
                return information, encodedVector
        return informationList[0], encodedVectorList[0]

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

        assert( len(xVectorDistribution) == self.length )

        # print(xyVectorDistribution)

        (decodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex, informationVectorIndex, xVectorDistribution, xyVectorDistribution, marginalizedUProbs)

        # print( decodedVector, marginalizedUProbs )
        assert( next_uIndex == len(decodedVector) == len(xVectorDistribution) )
        assert( next_informationVectorIndex == len(information) == 0 )
        assert( len(marginalizedUProbs) ==  self.length )

        if trustXYProbs:
            Pevec = np.array([min(probTuple) for probTuple in marginalizedUProbs])
            Hvec = np.array([QaryMemorylessDistribution.eta_list(probTuple) for probTuple in marginalizedUProbs])
        else:
            print("NOT YET IMPLEMENTED")
            exit(1)

        return (decodedVector, Pevec, Hvec)

    def genieSingleEncodeSimulation(self, xVectorDistribution):
        """Pick up statistics of a single encoding run
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

        assert( len(xVectorDistribution) == self.length )

        (encodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex, informationVectorIndex, xVectorDistribution, None, marginalizedUProbs)

        # print( encodedVector, marginalizedUProbs )
        assert( next_uIndex == len(encodedVector) == len(xVectorDistribution) )
        assert( next_informationVectorIndex == len(information) == 0 )
        assert( len(marginalizedUProbs) ==  self.length )

        TVvec = []
        Hvec = []

        for probTuple in marginalizedUProbs:
            print("NOT IMPLEMENTED YET")
            exit(1)

        return (encodedVector, TVvec, Hvec)

    def recursiveEncodeDecode(self, information, uIndex, informationVectorIndex, xVectorDistribution, xyVectorDistribution=None, marginalizedUProbs=None):
        """Encode/decode according to supplied vector distributions

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
                marginalizedVector = xVectorDistribution.calcMarginalizedProbabilities()
                encodedVector[0] = min(
                    np.searchsorted(np.cumsum(marginalizedVector), self.randomlyGeneratedNumbers[uIndex]),
                    self.q - 1)
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

            (minusEncodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex, informationVectorIndex, xMinusVectorDistribution, xyMinusVectorDistribution, marginalizedUProbs)

            xPlusVectorDistribution = xVectorDistribution.plusTransform(minusEncodedVector)
            xPlusVectorDistribution.normalize()
            if decoding:
                xyPlusVectorDistribution = xyVectorDistribution.plusTransform(minusEncodedVector)
                xyPlusVectorDistribution.normalize()
            else:
                xyPlusVectorDistribution = None

            uIndex = next_uIndex
            informationVectorIndex = next_informationVectorIndex
            (plusEncodedVector, next_uIndex, next_informationVectorIndex) = self.recursiveEncodeDecode(information, uIndex, informationVectorIndex, xPlusVectorDistribution, xyPlusVectorDistribution, marginalizedUProbs)

            halfLength = len(xVectorDistribution) // 2

            for halfi in range(halfLength):
                encodedVector[2*halfi] = (minusEncodedVector[halfi] + plusEncodedVector[halfi]) % self.q
                encodedVector[2*halfi + 1] = (-plusEncodedVector[halfi] + self.q) % self.q

            return (encodedVector, next_uIndex, next_informationVectorIndex)

    def recursiveListDecode(self, informationList, uIndex, informationVectorIndex, xVectorDistributionList, xyVectorDistributionList=None, marginalizedUProbs=None, inListSize=1, maxListSize=1):
        """Encode/decode according to supplied vector distributions

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
        assert(inListSize <= maxListSize)
        segmentSize = len(xVectorDistributionList[0])

        if segmentSize == 1:
            if self.frozenOrInformation[uIndex] == uIndexType.information:
                encodedVectorList = np.full((inListSize * self.q, segmentSize), -1, dtype=np.int64)
                for i in range(inListSize):
                    information = informationList[i]
                    for s in range(self.q):
                        if s > 0:
                            informationList[s*inListSize + i] = information  # branch the paths q times
                        informationList[s*inListSize + i][informationVectorIndex] = s
                        encodedVectorList[s*inListSize + i][0] = s
                # marginalizedVector = xyVectorDistribution.calcMarginalizedProbabilities()
                # information[informationVectorIndex] = np.argmax(marginalizedVector)
                # encodedVector[0] = information[informationVectorIndex]
                newListSize = inListSize * self.q
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex + 1

                if newListSize > maxListSize:
                    probs = np.empty(newListSize, dtype=np.float)
                    for i in range(inListSize):
                        marginalizedVector = xyVectorDistributionList[i].calcMarginalizedProbabilities()
                        for s in range(self.q):
                            probs[s*inListSize + i] = marginalizedVector[s]
                    newListSize = min(np.count_nonzero(probs), maxListSize)
                    indices_to_keep = np.argpartition(probs, -newListSize)[-newListSize:]
                    original_indices_map = indices_to_keep % inListSize
                    informationList[0:newListSize] = informationList[indices_to_keep]
                    informationList[newListSize:, :] = -1
                    encodedVectorList = encodedVectorList[indices_to_keep]
                else:
                    original_indices_map = np.repeat(np.arange(inListSize), self.q)
            else:
                newListSize = inListSize
                encodedVectorList = np.full((inListSize, segmentSize), -1, dtype=np.int64)
                # Calculate the frozen value for each branch and append it to the corresponding path
                for i in range(inListSize):
                    marginalizedVector = xVectorDistributionList[i].calcMarginalizedProbabilities()
                    encodedVectorList[i][0] = min(
                        np.searchsorted(np.cumsum(marginalizedVector), self.randomlyGeneratedNumbers[uIndex]),
                        self.q - 1)
                next_uIndex = uIndex + 1
                next_informationVectorIndex = informationVectorIndex
                original_indices_map = np.arange(inListSize)

            # should we return the marginalized probabilities?
            # TODO adapt this to list decoding
            # if marginalizedUProbs is not None:
            #     vectorDistribution = xyVectorDistribution or xVectorDistribution
            #     marginalizedVector = vectorDistribution.calcMarginalizedProbabilities()
            #     marginalizedUProbs.append(marginalizedVector)

            return (informationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize, original_indices_map)
        else:
            xMinusVectorDistributionList = []
            xyMinusVectorDistributionList = []
            for i in range(inListSize):
                xMinusVectorDistribution = xVectorDistributionList[i].minusTransform()
                xMinusVectorDistribution.normalize()
                xMinusVectorDistributionList.append(xMinusVectorDistribution)

                xyMinusVectorDistribution = xyVectorDistributionList[i].minusTransform()
                xyMinusVectorDistribution.normalize()
                xyMinusVectorDistributionList.append(xyMinusVectorDistribution)

            (minusInformationList, minusEncodedVectorList, next_uIndex, next_informationVectorIndex, minusListSize, minusOriginalIndicesMap) = self.recursiveListDecode(informationList, uIndex, informationVectorIndex, xMinusVectorDistributionList, xyMinusVectorDistributionList, marginalizedUProbs, inListSize, maxListSize)

            xPlusVectorDistributionList = []
            xyPlusVectorDistributionList = []
            for i in range(minusListSize):
                origI = minusOriginalIndicesMap[i]

                xPlusVectorDistribution = xVectorDistributionList[origI].plusTransform(minusEncodedVectorList[i])
                xPlusVectorDistribution.normalize()
                xPlusVectorDistributionList.append(xPlusVectorDistribution)

                xyPlusVectorDistribution = xyVectorDistributionList[origI].plusTransform(minusEncodedVectorList[i])
                xyPlusVectorDistribution.normalize()
                xyPlusVectorDistributionList.append(xyPlusVectorDistribution)

            uIndex = next_uIndex
            informationVectorIndex = next_informationVectorIndex
            (plusInformationList, plusEncodedVectorList, next_uIndex, next_informationVectorIndex, plusListSize, plusOriginalIndicesMap) = self.recursiveListDecode(minusInformationList, uIndex, informationVectorIndex, xPlusVectorDistributionList, xyPlusVectorDistributionList, marginalizedUProbs, minusListSize, maxListSize)

            newListSize = plusListSize

            encodedVectorList = np.full((newListSize, segmentSize), -1, dtype=np.int64)
            halfLength = segmentSize // 2

            for i in range(newListSize):
                minusI = plusOriginalIndicesMap[i]
                for halfi in range(halfLength):
                    encodedVectorList[i][2*halfi] = (minusEncodedVectorList[minusI][halfi] + plusEncodedVectorList[i][halfi]) % self.q
                    encodedVectorList[i][2*halfi + 1] = (-plusEncodedVectorList[i][halfi] + self.q) % self.q

            originalIndicesMap = minusOriginalIndicesMap[plusOriginalIndicesMap]

            return (plusInformationList, encodedVectorList, next_uIndex, next_informationVectorIndex, newListSize, originalIndicesMap)

    def calculate_syndrome_and_complement(self, u_message):
        x_message = polarTransformOfQudits(self.q, u_message)

        w = np.copy(x_message)
        w[np.array(self.infoSet)] = 0
        w[np.array(self.frozenSet)] *= self._q - 1
        w[np.array(self.frozenSet)] %= self._q

        u = x_message
        u[np.array(self.frozenSet)] = 0

        return w, u

    def get_message_info_bits(self, u_message):
        return u_message[np.array(self.infoSet)]

    def get_message_frozen_bits(self, u_message):
        return u_message[np.array(self.frozenSet)]

    def ir(self, enc_dec, a, b, list_size, check_size):
        w, u = self.calculate_syndrome_and_complement(a)
        a_key = self.get_message_info_bits(u)

        check_matrix = np.random.choice(range(self.q), (enc_dec.k, check_size))
        check_value = np.matmul(a_key, check_matrix) % self.q

        # information = informationRNG.choices(range(0, q), k=encDec.k)
        # encodedVector = enc_dec.encode(xVectorDistribution, information)
        #
        # codeword = make_codeword(encodedVector)
        #
        # receivedWord = simulateChannel(codeword)
        # xyVectorDistribution = make_xyVectorDistribution(receivedWord)
        #
        # check_matrix = np.random.choice(range(q), (encDec.k, checkSize))
        # check_value = np.matmul(information, check_matrix) % q
        #
        # (decodedInformation, decodedVector) = enc_dec.listDecode(xVectorDistribution, xyVectorDistribution, maxListSize, check_matrix, check_value)


        x_b = np.mod(np.sum(b, polarTransformOfQudits(self.q, w)), self.q)
        u_b = self.decode(x_b, list_size=list_size, check_value=check_value, encoding_matrix=check_matrix)
        b_key = self.get_message_info_bits(u_b)

        return a_key, b_key

    def ir2(self, a, b, xVectorDistribution, make_xyVectorDistribution):
        a_key = polarTransformOfQudits(self.q, a)[np.array(list(self.infoSet))]
        b_key = self.decode(xVectorDistribution, make_xyVectorDistribution(b))
        return a_key, b_key

def ir2Simulation(q, length, make_xVectorDistribution, simulateChannel, make_xyVectorDistribution, numberOfTrials, frozenSet, commonRandomnessSeed=1, randomInformationSeed=1, verbosity=0):
    badKeys = 0
    xVectorDistribution = make_xVectorDistribution()
    encDec = QaryPolarEncoderDecoder(q, length, frozenSet, commonRandomnessSeed)
    informationRNG = random.Random(randomInformationSeed)

    # Note that we set a random seed, which is in charge of both setting the information bits as well as the channel output.
    for t in range(numberOfTrials):
        a = informationRNG.choices(range(0, q), k=encDec.length)
        b = simulateChannel(a)

        a_key, b_key = encDec.ir2(a, b, xVectorDistribution, make_xyVectorDistribution)

        if not np.array_equal(a_key, b_key):
            badKeys += 1
            if verbosity > 0:
                s = "Bad key"
                s += "Alice's key:\n" + str(a_key)
                s += "\nBob's key:\n" + str(b_key)
                print(s)
        else:
            if verbosity > 0:
                print("Good key")

    print("Error probability = ", badKeys, "/", numberOfTrials, " = ", badKeys / numberOfTrials)

def encodeDecodeSimulation(q, length, make_xVectorDistribution, make_codeword, simulateChannel, make_xyVectorDistribution, numberOfTrials, frozenSet, commonRandomnessSeed=1, randomInformationSeed=1, verbosity=0):
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

        (decodedVector, decodedInformation) = encDec.decode(xVectorDistribution, xyVectorDistribution)

        if not np.array_equal(information, decodedInformation):
            misdecodedWords += 1
            if verbosity > 0:
                s = str(t) + ") error, transmitted information:\n" + str(information)
                s += "\ndecoded information:\n" + str(decodedInformation)
                s += "\nencoded vector before guard bands added:\n" + str(encodedVector)
                s += "\ncodeword:\n" + str(codeword)
                s += "\nreceived word:\n" + str(receivedWord)
                print(s)

    print( "Error probability = ", misdecodedWords, "/", numberOfTrials, " = ", misdecodedWords/numberOfTrials )

def encodeListDecodeSimulation(q, length, make_xVectorDistribution, make_codeword, simulateChannel, make_xyVectorDistribution, numberOfTrials, frozenSet, maxListSize, checkSize, commonRandomnessSeed=1, randomInformationSeed=1, verbosity=0):
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

        (decodedInformation, decodedVector) = encDec.listDecode(xVectorDistribution, xyVectorDistribution, maxListSize, check_matrix, check_value)

        if not np.array_equal(information, decodedInformation):
            misdecodedWords += 1
            if verbosity > 0:
                s = str(t) + ") error, transmitted information:\n" + str(information)
                s += "\ndecoded information:\n" + str(decodedInformation)
                s += "\nencoded vector before guard bands added:\n" + str(encodedVector)
                s += "\ncodeword:\n" + str(codeword)
                s += "\nreceived word:\n" + str(receivedWord)
                print(s)

    print( "Error probability = ", misdecodedWords, "/", numberOfTrials, " = ", misdecodedWords/numberOfTrials )

def genieEncodeDecodeSimulation(length, make_xVectorDistribution, make_codeword, simulateChannel, make_xyVectorDistribution, numberOfTrials, errorUpperBoundForFrozenSet, genieSeed, trustXYProbs=True, filename=None):
    """Run a genie encoder and corresponding decoder, and return frozen set

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

        # print("codeword = ", codeword, ", receivedWord = ", receivedWord)

        xyVectorDistribution = make_xyVectorDistribution(receivedWord)

        (decodedVector, PevecTemp, HdecvecTemp) = encDec.genieSingleDecodeSimulation(xVectorDistribution, xyVectorDistribution, trustXYProbs)

        if  TVvec is None:
            TVvec = TVvecTemp
            Pevec = PevecTemp
            HEncvec = HencvecTemp
            HDecvec = HdecvecTemp
        else:
            assert( len(TVvec) == len(TVvecTemp) )
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

    print( "TVVec = ", TVvec )
    print( "pevec = ", Pevec )
    print( "HEncvec = ", HEncvec )
    if trustXYProbs:
        print( "HDecvec = ", HDecvec )
    print( "Normalized HEncsum = ",  HEncsum /len(HEncvec) )
    if trustXYProbs:
        print( "Normalized HDecsum = ", HDecsum /len(HDecvec) )

    frozenSet = frozenSetFromTVAndPe(TVvec, Pevec, errorUpperBoundForFrozenSet)
    print( "code rate = ", (len(TVvec) - len(frozenSet)) /  len(codeword) )
    print( "codeword length = ", len(codeword) )

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
            s = "*** " + str(i) + " " + str((TVvec[i] + Pevec[i]) * numberOfTrials) +  "\n"
            f.write(s)

        f.close()

    return frozenSet

def polarTransformOfQudits( q, xvec ):
    # print("xvec =", xvec)
    if len(xvec) == 1:
        return xvec
    else:
        assert( len(xvec) % 2 == 0 )

        vfirst = []
        vsecond = []
        for i in range((len(xvec) // 2)):
            vfirst.append( (xvec[2*i] + xvec[2*i+1]) % q )
            vsecond.append( (q - xvec[2*i+1]) % q )

        ufirst = polarTransformOfQudits(q, vfirst)
        usecond = polarTransformOfQudits(q, vsecond)

        return np.concatenate((ufirst, usecond))


def frozenSetFromTVAndPe(TVvec, Pevec, errorUpperBoundForFrozenSet):
    TVPlusPeVec = np.add(TVvec, Pevec)
    sortedIndices = sorted(range(len(TVPlusPeVec)), key=lambda k: TVPlusPeVec[k])

    # print( sortedIndices )

    errorSum = 0.0
    indexInSortedIndicesArray = -1
    frozenSet = set()

    while errorSum < errorUpperBoundForFrozenSet and indexInSortedIndicesArray + 1 < len(TVPlusPeVec):
        i = sortedIndices[indexInSortedIndicesArray + 1]
        if TVPlusPeVec[i] + errorSum <= errorUpperBoundForFrozenSet:
            errorSum += TVPlusPeVec[i]
            indexInSortedIndicesArray += 1
        else:
            break

    for j in range(indexInSortedIndicesArray+1,  len(TVPlusPeVec)):
        i = sortedIndices[j]
        frozenSet.add(i)

    print( "frozen set =", frozenSet )
    print( "fraction of non-frozen indices =", 1.0 - len(frozenSet) / len(TVPlusPeVec) )

    return frozenSet
