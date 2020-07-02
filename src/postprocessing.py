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
            agg[i] += d[player][i]

    # plot
    fig, ax = plt.subplots()
    ax.bar(t, agg)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_schedules(p, s, t):
    pass


def plot_storage(p, s, t):
    pass
