import yaml
import datetime


class Config:
    def __init__(self, file_path):
        params = self.read_parameters(file_path)
        self.debug_flag = params['debug-flag']
        self.num_players = int(params['num-players'])
        self.schedule_length = int(params['schedule-length'])
        self.start_date = datetime.datetime.strptime(params['start-date'], '%d/%m/%Y')
        self.end_date = self.start_date + datetime.timedelta(days=self.schedule_length)
        self.forecast_error = float(params['forecast-error'])

        # pricing
        self.pricing_parameter = []
        self.pricing_parameter.append(float(params['pricing-coeff'][0]))
        self.pricing_parameter.append(float(params['pricing-coeff'][1]))
        self.pricing_parameter.append(float(params['pricing-coeff'][2]))

        # iteration
        self.eps = float(params['eps'])
        self.max_iter_game = int(params['max-iter-game'])
        self.max_iter_opt = int(params['max-iter-opt'])
        self.max_time = int(params['max-time'])

        self.demand_file = params['demand-file']
        self.storage_file = params['storage-file']

        # players
        self.player_names = []
        for i in range(self.num_players):
            self.player_names.append(params['players'][i])

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

    def get_player_names(self):
        return self.player_names

    def get_demand_file(self):
        return self.demand_file

    def get_storage_file(self):
        return self.storage_file
