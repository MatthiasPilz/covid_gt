import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_cur_path_from_parameters(date, demand, storage):
    start_date = datetime.datetime.strptime(date, '%d/%m/%Y')
    result = ""
    result += start_date.strftime('%d%b%Y') + "_"
    result += demand[-12:-4] + "_"
    result += "storage" + str(storage) + "/"
    return result


def main():
    start_dates = [
        "31/01/2020",
        "07/02/2020",
        "28/02/2020",
    #   "11/03/2020"
    ]
    start_dates_alt = [
        "31Jan2020",
        "07Feb2020",
        "28Feb2020",
    #   "11Mar2020"
    ]
    # khaled:
    # start_dates = ["20/03/2020"]

    demands = [
        "./data/Demand_Oct_Wave.csv",
        "./data/Demand_Nov_Wave.csv",
        "./data/Demand_Dec_Wave.csv",
        "./data/Demand_Jan_Wave.csv",
        "./data/Demand_Feb_Wave.csv"
    ]
    end_dates = [
        "20/01/2021",
        "20/02/2021",
        "20/03/2021",
        "20/04/2021",
            "20/05/2021"
    ]

    storages = [0.0, 4.0, 9.0, 14.0, 19.0]

    players = ["east", "london", "midlands", "north_east", "north_west", "south_east", "south_west"]

    for date, date_alt in zip(start_dates, start_dates_alt):

        index = [x[-12:-9] for x in demands]
        columns = [str(x+1) for x in storages]

        savings = pd.DataFrame(index=index, columns=columns, dtype=np.float)
        savings = savings.fillna(0.0)
        # savings = [[0 for i in range(len(demands))] for j in range(len(storages))]

        for i, (demand, *_) in enumerate(zip(demands, end_dates)):
            for j, storage in enumerate(storages):
                cur_path = "../results/" + get_cur_path_from_parameters(date, demand, storage)

                costs_file = cur_path + "costs.csv"
                costs_ref_file = cur_path + "costs_ref.csv"

                df_costs = pd.read_csv(costs_file)
                df_costs.drop("time", axis=1, inplace=True)
                df_costs_ref = pd.read_csv(costs_ref_file)
                df_costs_ref.drop("time", axis=1, inplace=True)

                # costs per player
                costs = df_costs.aggregate('sum')
                costs_ref = df_costs_ref.aggregate('sum')

                total_costs = 0.0
                total_costs_ref = 0.0
                for p in players:
                    total_costs += costs[p]
                    total_costs_ref += costs_ref[p]

                savings.at[demand[-12:-9], str(storage+1)] = 100.0 * (1.0 - (total_costs_ref / total_costs))
                # savings[i][j] = 1.0 - (total_costs_ref / total_costs)

        sns_plot = sns.heatmap(savings)
        fig = sns_plot.get_figure()
        plt.xlabel("amount of storage (multiplier)")
        plt.ylabel("peak of second wave")
        plt.suptitle("Cost savings in %")
        plt.title("start date: " + date_alt)
        # plt.show()
        fig.savefig("../analysis/savings_startdate" + date_alt + ".png")
        plt.close(fig)
        # save data in new location
        # plot image
        # save plot


if __name__ == '__main__':
    main()
    exit(0)



