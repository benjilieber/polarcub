#! /usr/bin/env python3

import sys
import os
from collections import Counter

from scipy.stats import norm

sys.path.append(os.getcwd())

import csv
import math
import random
from timeit import default_timer as timer
import numpy as np

import QaryPolarEncoderDecoder
from ScalarDistributions import QaryMemorylessDistribution


def make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, length, use_log=False):
    def make_xVectorDistribution():
        xDistribution = QaryMemorylessDistribution.QaryMemorylessDistribution(q)
        xDistribution.probs = [xyDistribution.calcXMarginals()]
        xVectorDistribution = xDistribution.makeQaryMemorylessVectorDistribution(length, None, use_log=use_log)
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


def make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution, use_log=False):
    def make_xyVectorDistribution(receivedWord):
        length = len(receivedWord)
        useTrellis = False

        if useTrellis:
            xyVectorDistribution = xyDistribution.makeQaryTrellisDistribution(length, receivedWord)
        else:
            xyVectorDistribution = xyDistribution.makeQaryMemorylessVectorDistribution(length, receivedWord, use_log)

        return xyVectorDistribution

    return make_xyVectorDistribution

def get_construction_path(q, N, channel_type="QSC", QER=None, SNR=None, rate=None):
    """
    Returns the path to a file containing a construction of the code (i.e. indices of bits in codeword
    sorted in the descending order of their "quality". The path depends on codeword length and the
    chosen construction method. All construction are stored in the package folder.
    :return: A string with absolute path to the code construction.
    """
    assert(channel_type in ["QSC", "AWGN"])
    construction_path = os.path.dirname(os.path.abspath(__file__))
    construction_path += '/polar_codes_constructions/'
    construction_path += 'q={}/'.format(q)
    construction_path += 'N={}/'.format(N)
    if channel_type == "QSC":
        assert (QER is not None)
        construction_path += '{}/'.format('QER={}'.format(QER))
    elif channel_type == "AWGN":
        assert (SNR and rate)
        construction_path += '{}/'.format('SNR={}'.format(SNR))
        construction_path += '{}/'.format('rate={}'.format(rate))

    return construction_path

def getFrozenSet(q, N, n, L, channel_type, xDistribution, xyDistribution, qer, snr=None, rate=None, upperBoundOnErrorProbability=None, numInfoIndices=None, verbosity=False):
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
    if channel_type == "QSC":
        construction_path = get_construction_path(q, N, QER=qer)
    elif channel_type == "AWGN":
        construction_path = get_construction_path(q, N, SNR=snr, rate=rate)
    else:
        raise "Unsupported channel type: " + channel_type

    if channel_type != "QSC":
        raise "TODO: Implement this for AWGN channel"
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

    frozenSet = getFrozenSet(q, N, n, L, "QSC", xDistribution, xyDistribution, p, upperBoundOnErrorProbability, numInfoIndices, verbosity=verbosity)

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


def test_ir(q, channel_type="QSC", maxListSize=None, checkSize=0, numberOfTrials=200, ir_version=1, numInfoIndices=None, use_log=False, verbosity=False):
    p = 0.98
    L = 100
    n = 6
    N = 2 ** n
    upperBoundOnErrorProbability = 0.1

    xDistribution = None
    xyDistribution = QaryMemorylessDistribution.makeQSC(q, p)

    if channel_type != "QSC":
        raise "TODO: Implement this for AWGN channel"
    frozenSet = getFrozenSet(q, N, n, L, channel_type, xDistribution, xyDistribution, p, upperBoundOnErrorProbability, numInfoIndices, verbosity=verbosity)

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N, use_log)
    make_codeword = make_codeword_noprocessing
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution, use_log)

    if maxListSize is None:
        mixingFactor = max(frozenSet)+1 - len(frozenSet)
        maxListSize = mixingFactor ** q
        print(maxListSize)
    QaryPolarEncoderDecoder.irSimulation(q, N, simulateChannel,
                                          make_xyVectorDistribution, numberOfTrials, frozenSet, maxListSize, checkSize, use_log=use_log,
                                          verbosity=verbosity, ir_version=ir_version)

def test_ir_per_config(q, L, n, maxListSize, numTrials, channel_type="QSC", qer=None, snr=None, rate=None, numInfoIndices=None, frozenSet=None, use_log=False, verbosity=False, file_name=None):
    N = 2 ** n
    assert(channel_type in ["QSC", "AWGN"])

    if numInfoIndices is None:
        assert(rate is not None)
        numInfoIndices = math.floor(rate * N)

    xDistribution = None
    if channel_type == "QSC":
        xyDistribution = QaryMemorylessDistribution.makeQSC(q, qer)
    elif channel_type == "AWGN":
        xyDistribution = QaryMemorylessDistribution.makeAWGN(q, snr, rate)

    if frozenSet is None:
        frozenSet = getFrozenSet(q, N, n, L, channel_type, xDistribution, xyDistribution, qer=qer, snr=snr, rate=rate, numInfoIndices=numInfoIndices, verbosity=verbosity)

    if maxListSize is None:
        mixingFactor = max(frozenSet) + 1 - len(frozenSet)
        maxListSize = mixingFactor ** q
        print(maxListSize)

    if verbosity:
        print("q=" + str(q)
              + ", channelType=" + str(channel_type)
              + ", qer=" + str(qer)
              + ", snr=" + str(snr)
              + ", rate=" + str(rate)
              + ", n=" + str(n)
              + ", L=" + str(L)
              + ", numInfoQudits=" + str(numInfoIndices)
              + ", maxListSize=" + str(maxListSize)
              + ", numTrials=" + str(numTrials))

    if channel_type == "AWGN":
        raise "TODO: Implement this for AWGN channel"

    make_xVectorDistribution = make_xVectorDistribution_fromQaryMemorylessDistribution(q, xyDistribution, N, use_log)
    simulateChannel = simulateChannel_fromQaryMemorylessDistribution(xyDistribution)
    make_xyVectorDistribution = make_xyVectorDistribution_fromQaryMemorylessDistribution(xyDistribution, use_log)

    start = timer()
    frame_error_prob, symbol_error_prob, key_rate, prob_result_list = QaryPolarEncoderDecoder.irSimulation(q, N, simulateChannel,
                                         make_xyVectorDistribution, numTrials, frozenSet, maxListSize, use_log=use_log, verbosity=verbosity)
    end = timer()
    time_rate = (end - start) / (numTrials * N)
    if file_name is not None:
        write_header(file_name)
        write_result(file_name, q, qer, snr, calc_theoretic_key_rate(q, channel_type, qer=qer, snr=snr, rate=rate), n, N, L, "degradingUpgrading",
                     numInfoIndices, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate,
                     numTrials, prob_result_list,
                     verbosity=verbosity)
    return frame_error_prob, symbol_error_prob, key_rate, time_rate, maxListSize, prob_result_list

def test_ir_and_record_range(file_name, q_range, L, n_range, numTrials, channel_type="QSC", qer_range=None, snr_range=None, maxListRange=None, rate_range=None, numInfoQuditsDelta=0, numInfoQuditsRadius=10, maxListSizeCompleteRange=False, explicitNumInfoQuditsRange=None, largeMaxListSize=False, use_log=False, verbosity=False):
    write_header(file_name)
    assert(channel_type in ["QSC", "AWGN"])
    if channel_type != "QSC":
        raise "TODO: Implement this for AWGN channel"
    for q in q_range:
        if channel_type == "QSC":
            snr_range = [None] * len(qer_range)
        elif channel_type == "AWGN":
            qer_range = [None] * len(snr_range)
        for qer, snr in zip(qer_range, snr_range):
            for n in n_range:
                N = 2**n
                if rate_range is not None:
                    numInfoQuditsRange = [math.floor(rate * N) for rate in rate_range]
                else:
                    assert (explicitNumInfoQuditsRange is not None)
                    numInfoQuditsRange = explicitNumInfoQuditsRange
                    rate_range = [None] * len(numInfoQuditsRange)
                for numInfoQudits, rate in zip(numInfoQuditsRange, rate_range):
                    theoretic_key_rate = calc_theoretic_key_rate(q, channel_type, qer=qer, snr=snr, rate=rate)
                    # theoretic_num_info_qudits = math.ceil(N*theoretic_key_rate*math.log(2, q))
                    if largeMaxListSize:
                        frame_error_prob, symbol_error_prob, key_rate, time_rate, maxListSize, prob_result_list = test_ir_per_config(q=q, channel_type=channel_type, qer=qer, snr=snr, rate=rate, L=L, n=n, maxListSize=None, numInfoIndices=numInfoQudits,
                                                                             numTrials=numTrials, use_log=use_log, verbosity=verbosity)
                        write_result(file_name, q, qer, snr, theoretic_key_rate, n, N, L, "degradingUpgrading",
                                     numInfoQudits, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate, numTrials, prob_result_list,
                                     verbosity=verbosity)
                        continue

                    if maxListRange is None:
                        maxListRange = list(range(1, 21))
                    cur_frame_error_prob = None
                    prev_frame_error_prob = None
                    for maxListSize in maxListRange:
                        if not maxListSizeCompleteRange and prev_frame_error_prob is not None and (cur_frame_error_prob >= prev_frame_error_prob + 0.1):
                            break
                        if math.log2(q)*numInfoQudits < math.log2(maxListSize):
                            break
                        frame_error_prob, symbol_error_prob, key_rate, time_rate, maxListSize, prob_result_list = test_ir_per_config(q=q, channel_type=channel_type, qer=qer, snr=snr, rate=rate, L=L, n=n, maxListSize=maxListSize, numInfoIndices=numInfoQudits, numTrials=numTrials, use_log=use_log, verbosity=verbosity)
                        write_result(file_name, q, qer, snr, theoretic_key_rate, n, N, L, "degradingUpgrading", numInfoQudits, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate, numTrials, prob_result_list, verbosity=verbosity)
                        prev_frame_error_prob = cur_frame_error_prob
                        cur_frame_error_prob = frame_error_prob

def write_header(file_name):
    prob_result_names = [probResult.name for probResult in QaryPolarEncoderDecoder.ProbResult]
    header = ["q", "qer", "snr", "theoreticKeyRate", "n", "N", "L", "frozenBitsAlgorithm", "numInfoQudits", "rate", "maxListSize", "frameErrorProb", "symbolErrorProb", "keyRate", "yield", "efficiency", "timeRate", "numTrials"] + prob_result_names
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

def write_result(file_name, q, qer, snr, theoretic_key_rate, n, N, L, frozenBitsAlgorithm, numInfoQudits, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, time_rate, numTrials, prob_result_list, verbosity=False):
    if verbosity:
        print("writing results")
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        if q == 2:
            yld = (1 - frame_error_prob) * numInfoQudits * math.log(q, 2)
            efficiency = numInfoQudits * math.log(q, 2) / (-qer*math.log(qer, 2) - (1-qer) *math.log(1-qer, 2))
        else:
            yld = None
            efficiency = None
        probResultCounter = Counter(prob_result_list)
        if verbosity:
            print(probResultCounter)
        prob_result_stats = [probResultCounter[probResult]/numTrials for probResult in QaryPolarEncoderDecoder.ProbResult]
        writer.writerow([q, qer, snr, theoretic_key_rate, n, N, L, frozenBitsAlgorithm, numInfoQudits, rate, maxListSize, frame_error_prob, symbol_error_prob, key_rate, yld, efficiency, time_rate, numTrials] + prob_result_stats)

def calc_theoretic_key_rate(q, channel_type="QSC", qer=None, snr=None, rate=None):
    if channel_type == "QSC":
        if qer == 0.0:
            return math.log(q, 2)
        if qer == 1.0:
            return math.log(q/(q-1), 2)
        return math.log(q, 2) + (1-qer) * math.log(1-qer, 2) + qer * math.log(qer/(q - 1), 2)
    else:
        raise "TODO: Implement for channel type: " + channel_type

def calc_theoretic_key_qrate(q, qer):
    if qer == 0.0:
        return 1.0
    if qer == 1.0:
        return math.log(q/(q-1), q)
    return 1.0 + (1-qer) * math.log(1-qer, q) + qer * math.log(qer/(q - 1), q)


def diff_frozen(q, n, L1, L2, QER=None, SNR=None, rate=None):
    N = 2**n
    if QER is not None:
        directory_name = get_construction_path(q, N, QER=QER)
    else:
        assert(SNR is not None)
        directory_name = get_construction_path(q, N, SNR=SNR, rate=rate)
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

def diff_frozen_range(q_range, n_range, L1, L2, QER_range=None, SNR_range=None, rate_range=None):
    for q in q_range:
        for n in n_range:
            if QER_range is not None:
                for qer in QER_range:
                    diff_frozen(q, n, L1, L2, qer=qer)
            else:
                assert(SNR_range is not None)
                for snr in SNR_range:
                    for rate in rate_range:
                        diff_frozen(q, n, L1, L2, snr=snr, rate=rate)

def calcFrozenSetOnly(q, L, n, channel_type="QSC", qer=None, snr=None, rate=None, verbosity=False):
    print("q = " + str(q))
    assert(qer or (snr and rate))

    xDistribution = None
    if channel_type == "QSC":
        xyDistribution = QaryMemorylessDistribution.makeQSC(q, qer)
        getFrozenSet(q, 2**n, n, L, channel_type, xDistribution, xyDistribution, qer=qer, upperBoundOnErrorProbability=None, numInfoIndices=0, verbosity=verbosity)
    elif channel_type == "AWGN":
        xyDistribution = QaryMemorylessDistribution.makeAWGN(q, snr, rate)
        getFrozenSet(q, 2**n, n, L, channel_type, xDistribution, xyDistribution, snr=snr, rate=rate, upperBoundOnErrorProbability=None, numInfoIndices=0, verbosity=verbosity)
    else:
        raise "Unsupported channel type: " + channel_type

def calcFrozenSetsOnly(q_range, L, n_range, channel_type="QSC", qer_range=None, snr_range=None, rate_range=None, verbosity=False):
    assert(qer_range or (snr_range and rate_range))
    for q in q_range:
        for n in n_range:
            if channel_type == "QSC":
                for qer in qer_range:
                    calcFrozenSetOnly(q, L, n, channel_type=channel_type, qer=qer, verbosity=verbosity)
            elif channel_type == "AWGN":
                for snr in snr_range:
                    for rate in rate_range:
                        calcFrozenSetOnly(q, L, n, channel_type=channel_type, snr=snr, rate=rate, verbosity=verbosity)
            else:
                raise "Unsupported channel type: " + channel_type

def snr_to_qer(q, snr, rate):
    assert (q == 2)
    sigma_inv = math.sqrt(2*rate*10**(snr/10))
    return 1 - norm.cdf(sigma_inv)

def check_SPC_property(q, L, n, channel_type="QSC", qer=None, rate=None, verbosity=False):
    xDistribution = None
    if channel_type == "QSC":
        N = 2**n
        construction_path = get_construction_path(q, N, QER=qer)
        xyDistribution = QaryMemorylessDistribution.makeQSC(q, qer)
        TVvec, Pevec = QaryMemorylessDistribution.calcTVAndPe_degradingUpgrading(n, L, xDistribution, xyDistribution, construction_path, verbosity=verbosity)
        TVPlusPeVec = np.add(TVvec, Pevec)
        if rate:
            numInfoIndices = math.floor(N * rate)
            frozen_mask = np.zeros(N, dtype=bool)
            frozen_mask[list(QaryMemorylessDistribution.calcFrozenSet_degradingUpgrading(n, L, xDistribution, xyDistribution, construction_path, numInfoIndices=numInfoIndices, verbosity=verbosity))] = True
        print("min: " + str(min(TVPlusPeVec)) + ", max: " + str(max(TVPlusPeVec)))
        segmentSize = N
        violation = False
        while segmentSize > 1:
            for i in range(N // segmentSize):
                if rate:
                    if sum(frozen_mask[i * segmentSize: (i+1) * segmentSize]) == 1 and frozen_mask[i * segmentSize] is False:
                        violation = True
                        print("violation!")
                        print(TVPlusPeVec[i * segmentSize: (i + 1) * segmentSize])
                # if TVPlusPeVec[i * segmentSize] != max(TVPlusPeVec[i * segmentSize: (i+1) * segmentSize]):
                #     violation = True
                #     print("violation!")
                #     print(TVPlusPeVec[i * segmentSize: (i+1) * segmentSize])
            segmentSize //= 2
    else:
        raise "Unsupported channel type: " + channel_type

    if violation is False:
        print("yay!")
    else:
        print("violation!")

# test(2)
# test(3)
# test(3, listDecode=True, maxListSize=1, checkSize=1)
# test(3, listDecode=True, maxListSize=3, checkSize=4)
# test_ir(2)
# test_ir(2, 100, 20)
# test_ir(3, 10, 1, numberOfTrials=1, numInfoIndices=5, verbosity=True)
# test_ir(3)
# test_ir(3, ir_version=2)
# print(calc_theoretic_key_qrate(3, qer=0.99))
# print(calc_theoretic_key_qrate(3, qer=0.95))

maxListRange = [2187, 6561, 19683, 59049]  # [2, 4, 8, 16, 32, 64, 729, 2187, 6561, 19683]
n_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
use_log = False

# test_ir_and_record_range(file_name="results_nospc.csv", q_range=[3], qer_range=[0.01], L=100, n_range=n_range, numTrials=100, rate_range=[0.94, 0.95], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_nospc.csv", q_range=[3], qer_range=[0.02], L=100, n_range=n_range, numTrials=100, rate_range=[0.9, 0.92], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_nospc.csv", q_range=[3], qer_range=[0.05], L=100, n_range=n_range, numTrials=100, rate_range=[0.79, 0.81], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)

# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[0.95], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.24, 0.26], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[0.98], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.27, 0.28], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[0.99], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.31, 0.32], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)
# test_ir_and_record_range(file_name="results_fastsscl.csv", q_range=[3], qer_range=[1.0], L=100, n_range=[11, 12, 13, 14], numTrials=100, rate_range=[0.36, 0.37], maxListRange=maxListRange, maxListSizeCompleteRange=True, use_log=use_log, verbosity=True)

# calcFrozenSetsOnly(q_range=[3], L=100, n_range=[12], qer_range=[1.0], verbosity=True)

# import plot
# q_list = [3]
# qer_list = [0.99]
# rate_list = None  # [0.94, 0.95]
# max_list_size_list = None  # [729, 2187]  # [2, 4, 8, 16, 32, 64, 128]
# n_list = [11, 12, 13, 14]
# plot.plot_results3("results_fastsscl.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list, rate_filter=rate_list, max_list_size_filter=max_list_size_list, errorType="ml_lower_bound", adjustedKeyRate=True)
# plot.plot_results3("results_spc.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list, rate_filter=rate_list, max_list_size_filter=max_list_size_list, errorType="frame", adjustedKeyRate=True)
# plot.plot_results1("results11.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list, errorType="ml_lower_bound", adjustedKeyRate=False) # log probs
# plot.plot_results2("results9.csv", q_filter=q_list, qer_filter=qer_list, n_filter=n_list)
# diff_frozen_range([2, 3], [4, 5, 6, 7, 8, 9, 10], 100, 150, qer_range=[0.98, 0.99, 1.0])
# diff_frozen_range([3], [6], 100, 200, qer_range=[0.98])
# test_ir_per_config(q=3, qer=0.01, L=100, n=8, maxListSize=10, numInfoIndices=230, numTrials=100, use_log=True, verbosity=True)
#
# test_ir_per_config(q=2, snr=1.0, L=100, n=11, maxListSize=1, rate=0.5, numTrials=100, use_log=True, verbosity=True)
# calcFrozenSetsOnly(q_range=[3], L=100, n_range=[13, 14], qer_range=[1.0], verbosity=True)

# snr_list = list(np.arange(1.0, 2.25, 0.25))
# snr_list = list(np.arange(2.25, 3.25, 0.25))
# for snr in snr_list:
#     # print("snr=" + str(snr))
#     # qer = snr_to_qer(2, snr, 0.5)
#     # print("qer=" + str(qer))
#     # calcFrozenSetsOnly(q_range=[2], L=100, n_range=[11], snr_range=[snr], rate_range=[0.5], verbosity=True)

# print(calc_theoretic_key_qrate(2, 0.02))  # 0.8585594574581793
# print(calc_theoretic_key_qrate(2, 0.02)*2**20)  # 900264.8416636678
# for maxListSize in [1, 2, 4, 8, 16, 32, 64]:
#     for numInfoIndices in [90_000, 90_100, 90_200, 90_300]:
#         test_ir_per_config(q=2, qer=0.02, L=20, n=20, maxListSize=maxListSize, numInfoIndices=numInfoIndices, numTrials=100, use_log=True, verbosity=True, file_name="results11.csv")
# check_SPC_property(q=3, L=100, n=8, channel_type="QSC", qer=1.0, rate=0.36, verbosity=True)
# test_ir_per_config(q=3, qer=1.0, L=100, n=10, maxListSize=30, numInfoIndices=600, numTrials=100, use_log=False, verbosity=True)