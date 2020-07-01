import datetime
import pandas as pd
import numpy as np


class Game:
    def __init__(self, params):
        self.num_players = int(params['num-players'])
        self.schedule_length = int(params['schedule-length'])
        self.start_date = datetime.datetime.strptime(params['start-date'], '%d/%m/%Y')
        self.end_date = self.start_date + datetime.timedelta(days=self.schedule_length)

        self.pricing_parameter = []
        self.pricing_parameter.append(params['pricing-coeff'][0])
        self.pricing_parameter.append(params['pricing-coeff'][1])
        self.pricing_parameter.append(params['pricing-coeff'][2])

        dateparse = lambda x: datetime.datetime.strptime(x, '%d/%m/%Y')
        self.demand = pd.read_csv(params['demand-file'],
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

    def display(self):
        """Display Configuration values."""
        print("\ngame class variables:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
