import datetime
import pandas as pd
import numpy as np


class Player:
    def __init__(self, name, demand, storage, start_date, game_length):
        self.name = name
        self.demand = self.read_demand(demand)

        self.storage_rate, \
            self.usage_rate, \
            self.max_capacity, \
            self.min_capacity, \
            self.initial_storage = self.read_storage(storage)

        self.game_start_date = start_date
        self.game_length = game_length
        self.game_start_index = self.compute_start_index()

    def read_demand(self, demand):
        dateparse = lambda x: datetime.datetime.strptime(x, '%d/%m/%Y')
        all_demand = pd.read_csv(demand,
                                 sep=',',
                                 skipinitialspace=True,
                                 parse_dates=['date'],
                                 date_parser=dateparse)
        temp = pd.concat([all_demand['date'], all_demand[self.name]], axis=1)
        temp = temp.rename(columns={"date": "date", self.name: "demand"})
        return temp

    def read_storage(self, storage):
        all_storage = pd.read_csv(storage,
                                  sep=',',
                                  skipinitialspace=True,
                                  index_col=0)

        storage_rate = all_storage[self.name]['storage-rate']
        usage_rate = all_storage[self.name]['usage-rate']
        max_capacity = all_storage[self.name]['max-capacity']
        min_capacity = all_storage[self.name]['min-capacity']
        initial_storage = all_storage[self.name]['initial-storage']

        return storage_rate, usage_rate, max_capacity, min_capacity, initial_storage

    def compute_start_index(self):
        temp = self.demand['date'].isin([self.game_start_date]).to_list()
        return temp.index(True)

    def display(self):
        """Display Configuration values."""
        print("\ngame class variables:")
        for a in dir(self):
            if not callable(getattr(self, a)):
                if "__" not in a:
                    print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def set_new_initial_storage(self, value):
        self.initial_storage = value

    def get_game_demand(self):
        temp = np.empty(self.game_length)
        for i in range(self.game_length):
            temp[i] = self.demand['demand'][self.game_start_index + i]
        return temp

    def get_initial_storage(self):
        return self.initial_storage

    def get_storage_rate(self):
        return self.storage_rate



