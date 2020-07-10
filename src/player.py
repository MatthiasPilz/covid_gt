import datetime
import pandas as pd


class Player:
    def __init__(self, name, demand, storage):
        self.name = name
        self.demand = self.read_demand(demand)

        self.storage_rate, \
            self.usage_rate, \
            self.max_capacity, \
            self.min_capacity, \
            self.initial_storage = self.read_storage(storage)

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

    def display(self):
        """Display Configuration values."""
        print("\ngame class variables:")
        for a in dir(self):
            if not callable(getattr(self, a)):
                if "__" not in a:
                    print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


