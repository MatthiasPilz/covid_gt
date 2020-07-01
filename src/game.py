import datetime


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

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
