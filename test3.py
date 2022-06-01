#! /usr/bin/env python3

import random
import os.path
import numpy as np

import QaryPolarEncoderDecoder
from ScalarDistributions import QaryMemorylessDistribution


def make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, length):
    def make_xVectorDistribution():
        xDistribution = QaryMemorylessDistribution.QaryMemorylessDistribution(q)
        xDistribution.probs = [xyDistribution.calcXMarginals()]
        xVectorDistribution = xDistribution.makeQaryMemorylessVectorDistribution(length, None)
        return xVectorDistribution

    return make_xVectorDistribution


def make_codeword_noprocessing(encodedVector):
    return encodedVector


def simulateChannel_fromQaryMemorylessDistribution(xyDistribution):
    def simulateChannel(codeword):
        receivedWord = []
        length = len(codeword)

        for j in range(length):
            x = codeword[j]

            rand = random.random()
            probSum = 0.0

            for y in range(len(xyDistribution.probs)):
                if probSum + xyDistribution.probXGivenY(x, y) >= rand:
                    receivedWord.append(y)
                    break
                else:
                    probSum += xyDistribution.probXGivenY(x, y)

        return receivedWord

    return simulateChannel


def make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution):
    def make_xyVectrorDistribution(receivedWord):
        length = len(receivedWord)
        useTrellis = False

        if useTrellis:
            xyVectorDistribution = xyDistribution.makeQaryTrellisDistribution(length, receivedWord)
        else:
            xyVectorDistribution = xyDistribution.makeQaryMemorylessVectorDistribution(length, receivedWord)

        return xyVectorDistribution

    return make_xyVectrorDistribution

def get_construction_path(q, N, QER):
    """
    Returns the path to a file containing a construction of the code (i.e. indices of bits in codeword
    sorted in the descending order of their "quality". The path depends on codeword length and the
    chosen construction method. All construction are stored in the package folder.
    :return: A string with absolute path to the code construction.
    """
    construction_path = os.path.dirname(os.path.abspath(__file__))
    construction_path += '/polar_codes_constructions/'
    construction_path += 'q={}/'.format(q)
    construction_path += 'N={}/'.format(N)
    construction_path += '{}/'.format('QER={}'.format(QER))

    return construction_path

def getFrozenSet(q, N, n, L, upperBoundOnErrorProbability, xDistribution, xyDistribution, QER):
    """
    Constructs the code, i.e. defines which bits are informational and which are frozen.
    Two behaviours are possible:
    1) If there is previously saved data with the sorted indices of channels for given N, QBER
    and construction method, it loads this data and uses it to define sets of informational and frozen bits.
    2) Otherwise, it calls the preferred method from the dict of methods to construct the code. Then, it
    saves sorted indices of channels and finally defines sets of informational and frozen bits.
    :param construction_method: A string defining which of the construction method to use;
    :return: void.
    """
    # Define the name where the dumped data should be stored
    construction_path = get_construction_path(q, N, QER)
    construction_name = construction_path + '{}.npy'.format("DegradingUpgrading_L=" + str(L) + "_upperBoundOnErrorProbability=" + str(upperBoundOnErrorProbability))

    print(construction_name)
    # If the file with data exists, load ordered_channels
    if os.path.isfile(construction_name):
        frozenSet = set(np.load(construction_name))
    # Otherwise, obtain construction and save it to the file
    else:
        frozenSet = QaryMemorylessDistribution.calcFrozenSet_degradingUpgrading(n, L, upperBoundOnErrorProbability,
                                                                            xDistribution, xyDistribution)

        if not os.path.exists(construction_path):
            os.makedirs(construction_path)
        np.save(construction_name, list(frozenSet))

    return frozenSet

def test(q, listDecode=False, maxListSize=None, checkSize=None):
    print("q = " + str(q))

    p = 0.99
    L = 100
    n = 5
    N = 2 ** n

    upperBoundOnErrorProbability = 0.1

    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, p)

    frozenSet = getFrozenSet(q, N, n, L, upperBoundOnErrorProbability, xDistribution, xyDistribution, p)

    # print("Rate = ", N - len(frozenSet), "/", N, " = ", (N - len(frozenSet)) / N)

    numberOfTrials = 200

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N)
    make_codeword = make_codeword_noprocessing
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution)

    if not listDecode:
        QaryPolarEncoderDecoder.encodeDecodeSimulation(q, N, make_xVectorDistribution, make_codeword, simulateChannel,
                                                       make_xyVectorDistribution, numberOfTrials, frozenSet,
                                                       verbosity=0)
    else:
        QaryPolarEncoderDecoder.encodeListDecodeSimulation(q, N, make_xVectorDistribution, make_codeword,
                                                           simulateChannel,
                                                           make_xyVectorDistribution, numberOfTrials, frozenSet,
                                                           maxListSize, checkSize, verbosity=0)

    # # trustXYProbs = False
    # trustXYProbs = True
    # PolarEncoderDecoder.genieEncodeDecodeSimulation(N, make_xVectorDistribuiton, make_codeword, simulateChannel, make_xyVectorDistribution, numberOfTrials, upperBoundOnErrorProbability, trustXYProbs)


def test_ir(q, maxListSize=1, checkSize=0, ir_version=1):
    p = 0.99
    L = 100
    n = 6
    N = 2 ** n

    upperBoundOnErrorProbability = 0.1

    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, p)

    frozenSet = getFrozenSet(q, N, n, L, upperBoundOnErrorProbability, xDistribution, xyDistribution, p)


    # print("Rate = ", N - len(frozenSet), "/", N, " = ", (N - len(frozenSet)) / N)

    numberOfTrials = 200

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N)
    make_codeword = make_codeword_noprocessing
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution)

    QaryPolarEncoderDecoder.irSimulation(q, N, make_xVectorDistribution, simulateChannel,
                                          make_xyVectorDistribution, numberOfTrials, frozenSet, maxListSize, checkSize,
                                          verbosity=1, ir_version=ir_version)


# test(2)
test(3)
test(3, listDecode=True, maxListSize=1, checkSize=1)
test(3, listDecode=True, maxListSize=9, checkSize=4)
# test_ir(2)
# test_ir(2, 100, 20)
# test_ir(3, 20, 4)
# test_ir(3)
# test_ir(3, ir_version=2)
