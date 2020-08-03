import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m


def _heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    plt.xlabel("amount of storage (multiplier)")
    plt.ylabel("peak of second wave")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, valfmt="{x:.2f}", textcolors=["white", "black"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = m.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def pd_heatmap(dataframe, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    col_labels = list(dataframe.columns)
    row_labels = list(dataframe.index.values)
    data = dataframe.to_numpy()

    im, cbar = _heatmap(data,
                        row_labels,
                        col_labels,
                        ax=ax,
                        cbar_kw=cbar_kw,
                        cbarlabel=cbarlabel,
                        **kwargs)
    return im, cbar


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
    #   "11/03/2020",
        "20/03/2020"
    ]
    start_dates_alt = [
        "31Jan2020",
        "07Feb2020",
        "28Feb2020",
    #   "11Mar2020",
        "20Mar2020"
    ]

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

    savings_dict = {}

    for date, date_alt in zip(start_dates, start_dates_alt):

        index = [x[-12:-9] for x in demands]
        columns = [str(x+1) for x in storages]

        savings = pd.DataFrame(index=index, columns=columns, dtype=np.float)
        savings = savings.fillna(0.0)
        savings_dict[date_alt] = savings

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

                savings_dict[date_alt].at[demand[-12:-9], str(storage+1)] = -100.0 * (1.0 - (total_costs_ref / total_costs))

    # calculate min and max for consistent axis
    minimum = np.float('inf')
    maximum = np.float('-inf')
    for saving in savings_dict:
        cur_min = min(savings_dict[saving].min())
        cur_max = max(savings_dict[saving].max())

        if cur_min < minimum:
            minimum = cur_min
        if cur_max > maximum:
            maximum = cur_max
    print('maximum savings overall: ', minimum)
    print('minimum savings overall: ', maximum)

    for date, date_alt in zip(start_dates, start_dates_alt):
        fig, _ = pd_heatmap(savings_dict[date_alt], cmap="YlGn", vmin=minimum, vmax=maximum)
        annotate_heatmap(fig, valfmt="{x:.1f} %", textcolors=["black", "white"], threshold=(maximum+minimum)//2)
        # plt.show()
        plt.title("start date: " + date_alt)
        plt.savefig("../analysis/savings_startdate" + date_alt + ".png")
        plt.close()


if __name__ == '__main__':
    main()
    exit(0)



