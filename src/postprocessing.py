import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(game, path):
    players = list(game.get_players().keys())
    demands = game.get_demand()
    schedules = game.get_schedules()
    dates = game.get_dates()
    fc_demands = game.get_fc_demand()

    plot_aggregated_demand(players, demands, schedules, dates, path)
    plot_schedules(players, schedules, dates, path)
    plot_storage(players, schedules, dates, path)
    plot_forecasted_values(players, demands, fc_demands, dates, path)


def plot_aggregated_demand(p, d, s, t, path):
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
    plt.savefig(path + '/aggregated_demand.png')
    plt.show()


def plot_schedules(p, s, t, path):
    # plot
    fig, ax = plt.subplots()
    for player in p:
        ax.plot(t, s[player])
    plt.xticks(rotation=45, ha='right')
    plt.title("schedules")
    plt.legend(p)
    plt.savefig(path + '/schedules.png')
    plt.show()


def plot_storage(p, s, t, path):
    # adding final element to t
    time = t.copy()
    time.append('final')

    # empty storage
    storage = dict()
    for player in p:
        temp = np.zeros(len(time), dtype=np.int32)
        storage[player] = temp

    # fill storage with values from s
    for player in p:
        for i in range(1, len(time)):
            storage[player][i] = storage[player][i-1] + s[player][i-1]

    # plot
    fig, ax = plt.subplots()
    for player in p:
        ax.plot(time, storage[player])
    plt.xticks(rotation=45, ha='right')
    plt.title("stored PPE")
    plt.legend(p)
    plt.savefig(path + '/storage.png')
    plt.show()


def plot_forecasted_values(p, d, fc_d, t, path):
    # plot
    fig, ax = plt.subplots()
    for player in p:
        ax.plot(t, d[player])
        ax.plot(t, fc_d[player])
    plt.xticks(rotation=45, ha='right')
    plt.title("forecasted demand")
    plt.legend([p, p])
    plt.savefig(path + '/forecasts.png')
    plt.show()


def calc_savings(costs, ref, p):
    savings = dict()
    for player in p:
        if ref[player] == 0:
            savings[player] = 0
        else:
            savings[player] = (1 - (costs[player] / ref[player]))*100
    return savings


def output_files(game, loc, repeat_counter=0):
    t = game.get_dates()
    p = game.get_players()
    d = game.get_demand()
    s = game.get_schedules()
    fc_d = game.get_fc_demand()
    loc = loc
    repeat_counter = repeat_counter

    agg = np.zeros(len(t))
    for player in p:
        for i in range(len(t)):
            agg[i] += d[player][i] + s[player][i]

    ref = np.zeros(len(t))
    for player in p:
        for i in range(len(t)):
            ref[i] += d[player][i]

    if repeat_counter == 0:
        demand_agg = [t, ref, agg]
        df = pd.DataFrame(demand_agg).T
        df.columns = ["time", "reference", "game"]
        df.to_csv(loc + 'load_agg.csv')

        fc_demands = fc_d
        df = pd.DataFrame(fc_demands)
        df.to_csv(loc + 'forecasted_demand.csv')

        schedules = s
        df = pd.DataFrame(schedules)
        df.to_csv(loc + 'schedules.csv')
    else:
        demand_agg = [t, ref, agg]
        df = pd.DataFrame(demand_agg).T
        df.columns = ["time", "reference", "game"]
        df.to_csv(loc + 'load_agg.csv', mode='a', header=False)

        fc_demands = fc_d
        df = pd.DataFrame(fc_demands)
        df.to_csv(loc + 'forecasted_demand.csv', mode='a', header=False)

        schedules = s
        df = pd.DataFrame(schedules)
        df.to_csv(loc + 'schedules.csv', mode='a', header=False)

