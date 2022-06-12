#! /usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

import csv
import math
import random
from timeit import default_timer as timer
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

def getFrozenSet(q, N, n, L, xDistribution, xyDistribution, QER, upperBoundOnErrorProbability=None, numInfoIndices=None, verbosity=False):
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
    frozenSet = QaryMemorylessDistribution.calcFrozenSet_degradingUpgrading(n, L, xDistribution, xyDistribution, construction_path, upperBoundOnErrorProbability, numInfoIndices, verbosity)
    return frozenSet

def test(q, listDecode=False, maxListSize=None, checkSize=None, numInfoIndices=None, verbosity=False):
    print("q = " + str(q))

    p = 0.99
    L = 100
    n = 8
    N = 2 ** n

    upperBoundOnErrorProbability = 0.1

    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, p)

    frozenSet = getFrozenSet(q, N, n, L, xDistribution, xyDistribution, p, upperBoundOnErrorProbability, numInfoIndices, verbosity=verbosity)

    # print("Rate = ", N - len(frozenSet), "/", N, " = ", (N - len(frozenSet)) / N)

    numberOfTrials = 200

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N)
    make_codeword = make_codeword_noprocessing
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution)

    if not listDecode:
        QaryPolarEncoderDecoder.encodeDecodeSimulation(q, N, make_xVectorDistribution, make_codeword, simulateChannel,
                                                       make_xyVectorDistribution, numberOfTrials, frozenSet,
                                                       verbosity=verbosity)
    else:
        QaryPolarEncoderDecoder.encodeListDecodeSimulation(q, N, make_xVectorDistribution, make_codeword,
                                                           simulateChannel,
                                                           make_xyVectorDistribution, numberOfTrials, frozenSet,
                                                           maxListSize, checkSize, verbosity=verbosity)

    # # trustXYProbs = False
    # trustXYProbs = True
    # PolarEncoderDecoder.genieEncodeDecodeSimulation(N, make_xVectorDistribuiton, make_codeword, simulateChannel, make_xyVectorDistribution, numberOfTrials, upperBoundOnErrorProbability, trustXYProbs)


def test_ir(q, maxListSize=None, checkSize=0, ir_version=1, numInfoIndices=None, verbosity=False):
    p = 0.98
    L = 100
    n = 6
    N = 2 ** n
    upperBoundOnErrorProbability = 0.1

    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, p)

    frozenSet = getFrozenSet(q, N, n, L, xDistribution, xyDistribution, p, upperBoundOnErrorProbability, numInfoIndices, verbosity=verbosity)


    # print("Rate = ", N - len(frozenSet), "/", N, " = ", (N - len(frozenSet)) / N)

    numberOfTrials = 200

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N)
    make_codeword = make_codeword_noprocessing
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution)

    if maxListSize is None:
        mixingFactor = max(frozenSet)+1 - len(frozenSet)
        maxListSize = mixingFactor ** q
        print(maxListSize)
    QaryPolarEncoderDecoder.irSimulation(q, N, make_xVectorDistribution, simulateChannel,
                                          make_xyVectorDistribution, numberOfTrials, frozenSet, maxListSize, checkSize,
                                          verbosity=verbosity, ir_version=ir_version)

def test_ir_per_config(q, qer, L, n, maxListSize, numInfoIndices, numTrials, verbosity=False):
    N = 2 ** n
    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, qer)

    frozenSet = getFrozenSet(q, N, n, L, xDistribution, xyDistribution, qer, None, numInfoIndices, verbosity=verbosity)

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N)
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution)

    start = timer()
    error_prob, key_rate = QaryPolarEncoderDecoder.irSimulation(q, N, make_xVectorDistribution, simulateChannel,
                                         make_xyVectorDistribution, numTrials, frozenSet, maxListSize, verbosity=verbosity)
    end = timer()
    time_rate = (end - start) / (numTrials * N)
    return error_prob, key_rate, time_rate

def test_ir_and_record_range(file_name, q_range, qer_range, L, n_range, numTrials, verbosity=False):
    write_header(file_name)
    for q in q_range:
        for qer in qer_range:
            theoretic_key_rate = calc_theoretic_key_rate(q, qer)
            for n in n_range:
                N = 2**n
                theoretic_num_info_qudits = math.ceil(N*theoretic_key_rate*math.log(2, q))
                for numInfoQudits in range(max(0, theoretic_num_info_qudits - 10), theoretic_num_info_qudits):
                    maxListSize = 1
                    cur_error_prob = None
                    prev_error_prob = None
                    while prev_error_prob is None or cur_error_prob < prev_error_prob:
                        error_prob, key_rate, time_rate = test_ir_per_config(q, qer, L, n, maxListSize, numInfoQudits, numTrials, verbosity=verbosity)
                        write_result(file_name, q, qer, theoretic_key_rate, n, N, L, "degradingUpgrading", numInfoQudits, maxListSize, error_prob, key_rate, time_rate, numTrials, verbosity=verbosity)
                        prev_error_prob = cur_error_prob
                        cur_error_prob = error_prob
                        maxListSize += 1

def write_header(file_name):
    header = ["q", "qer", "theoreticKeyRate", "n", "N", "L", "frozenBitsAlgorithm", "numInfoQudits", "maxListSize", "errorProb", "keyRate", "timeRate", "numTrials"]
    try:
        with open(file_name, 'r') as f:
            for row in f:
                assert(row.rstrip('\n').split(",") == header)
                return
    except FileNotFoundError:
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    except AssertionError:
        raise AssertionError(f"Header of {file_name} is bad.")

def write_result(file_name, q, qer, theoretic_key_rate, n, N, L, frozenBitsAlgorithm, numInfoQudits, maxListSize, error_prob, key_rate, time_rate, numTrials, verbosity=False):
    if verbosity:
        print("writing results")
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([q, qer, theoretic_key_rate, n, N, L, frozenBitsAlgorithm, numInfoQudits, maxListSize, error_prob, key_rate, time_rate, numTrials])

def calc_theoretic_key_rate(q, qer):
    if qer in [0.0, 1.0]:
        return math.log(q/(q-1), 2)
    return math.log(q, 2) + (1-qer) * math.log(1-qer, 2) + qer * math.log(qer/(q - 1), 2)

def diff_frozen(q, n, QER, L1, L2):
    N = 2**n
    directory_name = get_construction_path(q, N, QER)
    print(directory_name)

    tv_construction_name1 = directory_name + '{}.npy'.format("DegradingUpgrading_L=" + str(L1) + "_tv")
    pe_construction_name1 = directory_name + '{}.npy'.format("DegradingUpgrading_L=" + str(L1) + "_pe")
    tv_construction_name2 = directory_name + '{}.npy'.format("DegradingUpgrading_L=" + str(L2) + "_tv")
    pe_construction_name2 = directory_name + '{}.npy'.format("DegradingUpgrading_L=" + str(L2) + "_pe")
    # If the files with data exist, load them
    if os.path.isfile(tv_construction_name1) and os.path.isfile(pe_construction_name1) and os.path.isfile(tv_construction_name2) and os.path.isfile(pe_construction_name2):
        tv1 = np.load(tv_construction_name1)
        pe1 = np.load(pe_construction_name1)
        tv2 = np.load(tv_construction_name2)
        pe2 = np.load(pe_construction_name2)
        TvPlusPe1 = np.add(tv1, pe1)
        sortedIndices1 = sorted(range(len(TvPlusPe1)), key=lambda k: TvPlusPe1[k])
        TvPlusPe2 = np.add(tv2, pe2)
        sortedIndices2 = sorted(range(len(TvPlusPe2)), key=lambda k: TvPlusPe2[k])
        if sortedIndices1 != sortedIndices2:
            print(sortedIndices1)
            print(sortedIndices2)
            print("diff!")
        else:
            print("identical")

def diff_frozen_range(q_range, n_range, QER_range, L1, L2):
    for q in q_range:
        for n in n_range:
            for qer in QER_range:
                diff_frozen(q, n, qer, L1, L2)

def calcFrozenSetOnly(q, qer, L, n, verbosity=False):
    print("q = " + str(q))
    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, qer)
    frozenSet = getFrozenSet(q, 2**n, n, L, xDistribution, xyDistribution, qer, upperBoundOnErrorProbability=None,
                             numInfoIndices=0, verbosity=verbosity)

def calcFrozenSetsOnly(q_range, qer_range, L, n_range, verbosity=False):
    for q in q_range:
        for n in n_range:
            for qer in qer_range:
                calcFrozenSetOnly(q, qer, L, n, verbosity=verbosity)

# test(2)
# test(3)
# test(3, listDecode=True, maxListSize=1, checkSize=1)
# test(3, listDecode=True, maxListSize=3, checkSize=4)
# test_ir(2)
# test_ir(2, 100, 20)
# test_ir(3, 3, 1, numInfoIndices=18)
# test_ir(3, 1, 1)
# test_ir(3)
# test_ir(3, ir_version=2)

# test_ir_and_record_range("results.csv", [2, 3], [0.98, 0.99, 1.0], 150, [6, 7, 8, 9, 10], 200, verbosity=True)
# calcFrozenSetsOnly([3], [0.98, 0.99, 1.0], 150, [4, 5, 6, 7, 8, 9, 10], verbosity=True)
# test_ir_and_record_range("results.csv", [2], [0.98], 150, [4], 200, verbosity=False)
# import plot
# plot.plot_results2("results.csv")
diff_frozen_range([2, 3], [4, 5, 6, 7, 8, 9, 10], [0.98, 0.99, 1.0], 100, 150)
