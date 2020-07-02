import matplotlib.pyplot as plt
import numpy as np


def plot_all(players, demands, schedules, dates):
    plot_aggregated_demand(players, demands, schedules, dates)
    plot_schedules(players, schedules, dates)
    plot_storage(players, schedules, dates)


def plot_aggregated_demand(p, d, s, t):
    # compute aggregated demand
    agg = np.zeros(len(t))
    for player in p:
        for i in range(len(t)):
            agg[i] += d[player][i] + s[player][i]

    ref = np.zeros(len(t))
    for player in p:
        for i in range(len(t)):
            ref[i] += d[player][i]

    # plot
    fig, ax = plt.subplots()
    ax.bar(t, ref)
    ax.bar(t, agg)
    plt.xticks(rotation=45, ha='right')
    plt.legend(['reference', 'game'])
    plt.title("aggregated demand")
    plt.show()


def plot_schedules(p, s, t):
    # plot
    fig, ax = plt.subplots()
    for player in p:
        ax.plot(t, s[player])
    plt.xticks(rotation=45, ha='right')
    plt.title("schedules")
    plt.legend(p)
    plt.show()


def plot_storage(p, s, t):
    # empty storage
    storage = dict()
    for player in p:
        temp = np.zeros(len(t), dtype=np.int32)
        storage[player] = temp

    # fill storage with values from s
    for player in p:
        for i in range(1, len(t)):
            storage[player][i] = storage[player][i-1] + s[player][i]

    # plot
    fig, ax = plt.subplots()
    for player in p:
        ax.plot(t, storage[player])
    plt.xticks(rotation=45, ha='right')
    plt.title("stored PPE")
    plt.legend(p)
    plt.show()


def calc_savings(costs, ref, p):
    savings = dict()
    for player in p:
        savings[player] = (1 - (costs[player] / ref[player]))*100
    return savings
