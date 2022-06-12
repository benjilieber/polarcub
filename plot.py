import matplotlib.pyplot as plt
import pandas as pd

def plot_results1(file_name):
    df = pd.read_csv(file_name)
    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)

        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            cur_qer_group = grouped_by_qer.get_group(qer)
            theoretic_key_rate = cur_qer_group.theoreticKeyRate.iloc[0]
            plt.axvline(x=theoretic_key_rate)

            grouped_by_N = cur_qer_group.groupby("N")
            for key in grouped_by_N.groups.keys():
                cur_group = grouped_by_N.get_group(key)
                plt.scatter(cur_group.keyRate, cur_group.errorProb, label=str(key), s=cur_group.maxListSize)

            plt.legend()
            plt.title('q=' + str(q) + ', qer=' + str(qer))
            plt.xlabel('rate')
            plt.ylabel('error probability')
            plt.show()

def plot_results2(file_name):
    df = pd.read_csv(file_name)
    df['finalRate'] = df.keyRate * (1 - df.errorProb)
    grouped_by_q = df.groupby("q")
    for q in grouped_by_q.groups.keys():
        cur_q_group = grouped_by_q.get_group(q)
        grouped_by_qer = cur_q_group.groupby("qer")
        for qer in grouped_by_qer.groups.keys():
            cur_qer_group = grouped_by_qer.get_group(qer)
            theoretic_key_rate = cur_qer_group.theoreticKeyRate.iloc[0]
            plt.axhline(y=theoretic_key_rate)

            grouped_by_n = cur_qer_group.groupby("n")
            for key in grouped_by_n.groups.keys():
                cur_group = grouped_by_n.get_group(key)
                plt.scatter(cur_group.maxListSize, cur_group.finalRate, label=str(key))

            plt.legend()
            plt.title('q=' + str(q) + ', qer=' + str(qer))
            plt.xlabel('n (= log N)')
            plt.ylabel('final rate')
            plt.show()