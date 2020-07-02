import datetime
import yaml
import pandas as pd
import numpy as np


class Game:
    def __init__(self, config_file):
        params = self.read_parameters(config_file)
        # general
        self._debug_flag = params['debug-flag']
        self._num_players = int(params['num-players'])
        self._schedule_length = int(params['schedule-length'])
        self._start_date = datetime.datetime.strptime(params['start-date'], '%d/%m/%Y')
        self._end_date = self._start_date + datetime.timedelta(days=self._schedule_length)

        dateparse = lambda x: datetime.datetime.strptime(x, '%d/%m/%Y')
        self.__demand = pd.read_csv(params['demand-file'],
                                    sep = ',',
                                    skipinitialspace = True,
                                    dtype={'east': np.int32,
                                           'london': np.int32,
                                           'midlands': np.int32,
                                           'north_east': np.int32,
                                           'north_west': np.int32,
                                           'south_east': np.int32,
                                           'south_west': np.int32, },
                                    parse_dates=['date'],
                                    date_parser=dateparse)

        # pricing
        self._pricing_parameter = []
        self._pricing_parameter.append(params['pricing-coeff'][0])
        self._pricing_parameter.append(params['pricing-coeff'][1])
        self._pricing_parameter.append(params['pricing-coeff'][2])

        # storage
        self._storage_rate = params['storage-rate']
        self._usage_rate = params['usage-rate']
        self._initial_state = params['initial-state']
        self._max_capacity = params['max-capacity']
        self._min_capacity = params['min-capacity']

        if self.debug_state():
            self.check_initialisation()

        # more fields for eventual computations
        self._players = self.get_players()
        self.__schedules = dict()
        for p in self._players:
            s = np.zeros(self._schedule_length, dtype=np.int32)
            self.__schedules[p] = s

        # derive index from dates..
        self._start_index = self.get_start_index()
        self._end_index = self._start_index + self._schedule_length
        assert self.__demand['date'][self._end_index] == self._end_date, "end date and index don't match"

    @staticmethod
    def read_parameters(config_file):
        params = {}
        try:
            with open(config_file, 'r') as file:
                # all the parameters from the config file
                params = yaml.load(file, Loader=yaml.SafeLoader)
        except Exception as e:
            print('*** Error reading the config file - ' + str(e))
        return params

    def debug_state(self):
        return self._debug_flag

    def check_initialisation(self):
        assert self._num_players == self.__demand.shape[1]-1, "number of players should match number of columns in file!"
        flag_pricing_parameter = (self._pricing_parameter[1] != 0.0 or self._pricing_parameter[2] != 0.0)
        assert flag_pricing_parameter, "at least one of the cost-coefficients needs to be different from zero!"
        assert (self._start_date == self.__demand['date']).any(), "start date needs to be within data range"
        assert (self._end_date == self.__demand['date']).any(), "end date needs to be within data range"
        assert self._storage_rate > 0, "storage rate needs to be larger than zero"
        assert self._usage_rate < 0, "usage rate needs to be smaller than zero"
        assert self._max_capacity >= 0, "max capacity needs to be larger than zero"
        assert self._min_capacity >= 0, "min capacity needs to be non-negative"
        flag_initial_state = self._min_capacity <= self._initial_state <= self._max_capacity
        assert flag_initial_state, "initial state not within the storage limits"

    def display(self):
        """Display Configuration values."""
        print("\ngame class variables:")
        for a in dir(self):
            if not callable(getattr(self, a)):
                if "__" not in a:
                    print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def get_players(self):
        players = list(self.__demand.columns)
        players.pop(0)
        return players

    def get_start_index(self):
        temp = self.__demand['date'].isin([self._start_date]).to_list()
        return temp.index(True)
