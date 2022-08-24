import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_results1(file_name, q_filter=None, qer_filter=None, snr_filter=None, n_filter=None, errorType="frame", adjustedKeyRate=False):
    df = pd.read_csv(file_name)

    if n_filter is not None:
        df = df[df.n.isin(n_filter)]

    if not adjustedKeyRate:
        df["origKeyRate"] = np.log2(df.q)*df.numInfoQudits/df.N
        plt.xlabel('key rate')
    else:
        plt.xlabel('key rate, adjusted by list size')

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        if q_filter is not None and q not in q_filter:
            continue
        cur_q_group = grouped_by_q.get_group(q)

        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            if qer_filter is not None and qer not in qer_filter:
                continue
            cur_qer_group = grouped_by_qer.get_group(qer)
            theoretic_key_rate = cur_qer_group.theoreticKeyRate.iloc[0]
            plt.axvline(x=theoretic_key_rate)

            grouped_by_N = cur_qer_group.groupby("N")
            for key in grouped_by_N.groups.keys():
                cur_group = grouped_by_N.get_group(key)

                if not adjustedKeyRate:
                    keyRate = cur_group.origKeyRate
                else:
                    keyRate = cur_group.keyRate

                if errorType == "frame":
                    errorProb = cur_group.frameErrorProb
                    plt.ylabel('frame error probability')
                elif errorType == "symbol":
                    errorProb = cur_group.symbolErrorProb
                    plt.ylabel('symbol error probability')
                elif errorType == "ml_lower_bound":
                    errorProb = cur_group.FailActualSmallerThanMin
                    plt.ylabel('ML lower bound error probability')
                else:
                    raise "Unknown error type: " + errorType

                plt.scatter(keyRate, errorProb, label=str(key), s=cur_group.maxListSize)

            plt.legend()
            plt.title('q=' + str(q) + ', qer=' + str(qer))
            plt.show()

def plot_results2(file_name, q_filter=None, qer_filter=None, n_filter=None):
    df = pd.read_csv(file_name)

    if n_filter is not None:
        df = df[df.n.isin(n_filter)]

    df['finalRate'] = df.keyRate * (1 - df.frameErrorProb)
    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        if q_filter is not None and q not in q_filter:
            continue
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            if qer_filter is not None and qer not in qer_filter:
                continue
            cur_qer_group = grouped_by_qer.get_group(qer)
            theoretic_key_rate = cur_qer_group.theoreticKeyRate.iloc[0]
            plt.axhline(y=theoretic_key_rate, linewidth=4, label="TheoreticKeyRate")
            if q == 3:
                hagai_key_rate1 = qer / 2
                plt.axhline(y=hagai_key_rate1, color='r', label="Hagai1Rate")
                hagai_key_rate2 = math.log(3, 2) * qer / 4
                plt.axhline(y=hagai_key_rate2, color='y', label="Hagai2Rate")

            grouped_by_n = cur_qer_group.groupby("n")
            for key in grouped_by_n.groups.keys():
                cur_group = grouped_by_n.get_group(key)
                plt.scatter(cur_group.maxListSize, cur_group.finalRate, label=str(key))

            plt.legend()
            plt.title('q=' + str(q) + ', qer=' + str(qer))
            plt.xlabel('maxListSize')
            plt.ylabel('final rate')
            plt.show()


def plot_results3(file_name, q_filter=None, qer_filter=None, snr_filter=None, n_filter=None, rate_filter=None, max_list_size_filter=None, errorType="frame", adjustedKeyRate=False):
    df = pd.read_csv(file_name)

    if n_filter is not None:
        df = df[df.n.isin(n_filter)]
    if rate_filter is not None:
        df = df[df.rate.isin(rate_filter)]
    if max_list_size_filter is not None:
        df = df[df.maxListSize.isin(max_list_size_filter)]

    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        if q_filter is not None and q not in q_filter:
            continue
        cur_q_group = grouped_by_q.get_group(q)

        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            if qer_filter is not None and qer not in qer_filter:
                continue
            cur_qer_group = grouped_by_qer.get_group(qer)

            if not adjustedKeyRate:
                cur_qer_group["origKeyRate"] = np.log2(cur_qer_group.q)*cur_qer_group.numInfoQudits/cur_qer_group.N
                minRate = np.min(cur_qer_group["origKeyRate"])
                maxRate = np.max(cur_qer_group["origKeyRate"])
            else:
                minRate = np.min(cur_qer_group["keyRate"])
                maxRate = np.max(cur_qer_group["keyRate"])

            theoretic_key_rate = cur_qer_group.theoreticKeyRate.iloc[0]
            if errorType == "frame":
                maxProb = np.max(cur_qer_group.frameErrorProb)
            elif errorType == "symbol":
                maxProb = np.max(cur_qer_group.symbolErrorProb)
            elif errorType == "ml_lower_bound":
                maxProb = np.max(cur_qer_group.FailActualSmallerThanMin)
            elif errorType == "frame_ml":
                maxProb = np.max(1.0-cur_qer_group.SuccessActualIsMax)

            grouped_by_N = cur_qer_group.groupby("N")
            for N in grouped_by_N.groups.keys():
                plt.axvline(x=theoretic_key_rate)
                cur_N_group = grouped_by_N.get_group(N)
                grouped_by_rate = cur_N_group.groupby("rate")
                for rate in grouped_by_rate.groups.keys():
                    cur_group = grouped_by_rate.get_group(rate)

                    if not adjustedKeyRate:
                        keyRate = cur_group.origKeyRate
                        plt.xlabel('key rate')
                    else:
                        keyRate = cur_group.keyRate
                        plt.xlabel('key rate, adjusted by list size')

                    if errorType == "frame":
                        errorProb = cur_group.frameErrorProb
                        plt.ylabel('frame error probability')
                    elif errorType == "symbol":
                        errorProb = cur_group.symbolErrorProb
                        plt.ylabel('symbol error probability')
                    elif errorType == "ml_lower_bound":
                        errorProb = cur_group.FailActualSmallerThanMin
                        plt.ylabel('ML lower bound error probability')
                    elif errorType == "frame_ml":
                        errorProb = 1.0-cur_group.SuccessActualIsMax
                        plt.ylabel('frame list-decode ML error probability')
                    else:
                        raise "Unknown error type: " + errorType

                    plt.scatter(keyRate, errorProb, label=str(round(rate*math.log2(q)/theoretic_key_rate, 2)) + ", maxListSize = " + str(np.max(cur_group.maxListSize)), s=10*np.log2(cur_group.maxListSize))
                    plt.plot(keyRate, errorProb)

                plt.legend()
                plt.title('q=' + str(q) + ', qer=' + str(qer) + ', N=' + str(N))
                plt.xlim([max(0, minRate-0.1), maxRate+0.1])
                plt.ylim([0, maxProb])
                plt.show()