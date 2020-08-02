import yaml
import datetime
import os


class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        params = self.read_parameters(file_path)
        self.repeat = int(params['repeat'])
        self.debug_flag = params['debug-flag']
        self.plot_flag = params['plot-flag']
        self.num_players = int(params['num-players'])
        self.start_date = datetime.datetime.strptime(params['start-date'], '%d/%m/%Y')
        self.end_date = datetime.datetime.strptime(params['end-date'], '%d/%m/%Y')
        # self.schedule_length = int(params['schedule-length'])
        self.schedule_length = self.schedule_length = (self.end_date.date() - self.start_date.date()).days
        self.forecast_error = float(params['forecast-error'])

        # pricing
        self.pricing_parameters = []
        self.pricing_parameters.append(float(params['pricing-coeff'][0]))
        self.pricing_parameters.append(float(params['pricing-coeff'][1]))
        self.pricing_parameters.append(float(params['pricing-coeff'][2]))
        self.pricing_parameters.append(float(params['pricing-coeff'][3]))

        # iteration
        self.eps = float(params['eps'])
        self.max_iter_game = int(params['max-iter-game'])
        self.max_iter_opt = int(params['max-iter-opt'])
        self.max_time = int(params['max-time'])

        self.demand_file = params['demand-file']
        self.storage_file = params['storage-file']
        self.additional_storage = params['additional-storage']

        # players
        self.player_names = []
        for i in range(self.num_players):
            self.player_names.append(params['players'][i])

        self.output_dir = params['output-dir']
        # self.output_name = params['output-name']
        self.output_name = self.create_output_name_from_parameters()
        # self.output_path = self.create_output_dir()
        # self.parameter_copy_file = self.copy_config_to_output_dir()

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

    def create_output_dir(self):
        output_location = os.path.join(self.output_dir, self.output_name)
        try:
            os.makedirs(output_location)
            print("Directory ", output_location, " Created ")
        except FileExistsError:
            print("Directory ", output_location, " already exists")
        return output_location

    def copy_config_to_output_dir(self):
        with open(self.file_path) as f:
            list_doc = yaml.load(f, Loader=yaml.SafeLoader)

        file_name = os.path.join(self.output_path, "parameter_file.yaml")
        with open(file_name, "w") as f:
            yaml.dump(list_doc, f, Dumper=yaml.SafeDumper, default_flow_style=False)
        return file_name

    def create_output_name_from_parameters(self):
        result = ""
        result += self.start_date.strftime('%d%b%Y') + "_"
        result += self.demand_file[-12:-4] + "_"
        result += "storage" + str(self.additional_storage) + "/"
        return result

    def reset_for_new_configuration(self, new_start_date, new_demand_file, new_end_date, new_additional_storage):
        self.start_date = datetime.datetime.strptime(new_start_date, '%d/%m/%Y')
        self.end_date = datetime.datetime.strptime(new_end_date, '%d/%m/%Y')
        self.demand_file = new_demand_file
        self.additional_storage = new_additional_storage

        self.schedule_length = (self.end_date.date() - self.start_date.date()).days

        self.output_name = self.create_output_name_from_parameters()
        self.output_path = self.create_output_dir()
        self.parameter_copy_file = self.copy_config_to_output_dir()

    def set_start_date(self, val):
        self.start_date = val

    def get_player_names(self):
        return self.player_names

    def get_demand_file(self):
        return self.demand_file

    def get_storage_file(self):
        return self.storage_file

    def get_forecast_error(self):
        return self.forecast_error

    def get_schedule_length(self):
        return self.schedule_length

    def get_start_date(self):
        return self.start_date

    def get_max_game_time(self):
        return self.max_time

    def get_max_game_iter(self):
        return self.max_iter_game

    def get_pricing_parameters(self):
        return self.pricing_parameters

    def get_debug_flag(self):
        return self.debug_flag

    def get_iter_opt(self):
        return self.max_iter_opt

    def get_eps(self):
        return self.eps

    def get_repeat(self):
        return self.repeat

    def get_output_path(self):
        return self.output_path

    def get_plot_flag(self):
        return self.plot_flag

    def get_additional_storage(self):
        return self.additional_storage
