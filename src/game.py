import time
import numpy as np
from .player import Player
from scipy.optimize import minimize, LinearConstraint, Bounds
from datetime import timedelta


class Game:
    def __init__(self, config):
        self.config = config

        player_names = self.config.get_player_names()
        demand_file = self.config.get_demand_file()
        storage_file = self.config.get_storage_file()
        start_date = self.config.get_start_date()
        game_length = self.config.get_schedule_length()
        self.players = dict()
        for name in player_names:
            self.players[name] = Player(name, demand_file, storage_file, start_date, game_length)

        # derived variables:
        self.schedules = self.create_empty_int_arrays()
        self.L = self.create_empty_int_arrays()

        forecast_error = self.config.get_forecast_error()
        self.forecast_demand = self.compute_forecast_demand(forecast_error)
        self.cur_player = list(self.players.keys())[0]

    def display(self):
        """Display Configuration values."""
        print("\ngame class variables:")
        for a in dir(self):
            if not callable(getattr(self, a)):
                if "__" not in a:
                    print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

        for name in self.config.get_player_names():
            self.players[name].display()

    def create_empty_int_arrays(self):
        temp = dict()
        for p in self.players.keys():
            s = np.zeros(self.config.get_schedule_length(), dtype=np.int32)
            temp[p] = s
        return temp

    def compute_forecast_demand(self, forecast_error):
        temp = self.create_empty_int_arrays()

        for p in self.players.keys():
            # the chosen factor for the stdv is completely arbitrary
            mean = np.mean(self.players[p].get_game_demand())
            error_mean = forecast_error * mean
            error_stdv = error_mean * 0.33

            for i in range(self.config.get_schedule_length()):
                temp[p][i] = self.players[p].get_game_demand()[i] - int(np.random.normal(error_mean, error_stdv))
            # set negative values to 0
            temp[p] = np.where(temp[p] < 0, 0, temp[p])
        return temp

    def solve_game(self):
        start_time = time.time()
        end_time = start_time + self.config.get_max_game_time()

        num_iterations = 0
        flag_convergence = False
        flag_time_is_up = False

        while num_iterations < self.config.get_max_game_iter() and not flag_convergence and not flag_time_is_up:
            solution_profile = self.copy_schedules_to_solution()

            for p in self.players.keys():
                self.cur_player = p
                self.update_load_other_players()
                self.schedules[p] = self.find_optimal_response()

            flag_convergence, achieved_eps = self.check_for_convergence(solution_profile)
            flag_time_is_up = True if time.time() > end_time else False
            num_iterations += 1

            print("*** finished iteration number {} with eps = {}".format(num_iterations, achieved_eps))
            print("###############################################")
            print("")

        print("total number of iterations: {}".format(num_iterations))
        print("execution time for solver: {:.3f}s".format(time.time()-start_time))

        final_storage = self.adjust_schedules_for_real_demand()
        return final_storage

    def update_load_other_players(self):
        p = self.cur_player
        self.L[p] = np.zeros(self.config.get_schedule_length())
        for other in self.players.keys():
            if other == p:
                continue
            for i in range(self.config.get_schedule_length()):
                self.L[p][i] = self.players[other].get_game_demand()[i] + self.schedules[other][i]

    def copy_schedules_to_solution(self):
        temp = self.create_empty_int_arrays()
        for p in self.players.keys():
            for i in range(self.config.get_schedule_length()):
                temp[p][i] = self.schedules[p][i]
        return temp

    def objective(self, x):
        costs = 0
        a = self.config.get_pricing_parameters()[2]
        b = self.config.get_pricing_parameters()[1]

        for i in range(self.config.get_schedule_length()):
            d = self.forecast_demand[self.cur_player][i]
            L = self.L[self.cur_player][i]

            costs_i = a*d**3 + 2*a*d**2*L + 3*a*d**2*x[i] + a*d*L**2 + 4*a*d*L*x[i] + 3*a*d*x[i]**2 + \
                a*L**2*x[i] + 2*a*L*x[i]**2 + a*x[i]**3 + b*d**2 + b*d*L + 2*b*d*x[i] + b*L*x[i] + b*x[i]**2

            costs += costs_i
        return costs

    def objective_der(self, x):
        a = self.config.get_pricing_parameters()[2]
        b = self.config.get_pricing_parameters()[1]
        der = np.zeros_like(x)
        for i in range(self.config.get_schedule_length()):
            d = self.forecast_demand[self.cur_player][i]
            L = self.L[self.cur_player][i]

            der[i] = a*((d+x[i])+L)**2 + 2*a*(d+x[i])*(d+x[i]+L) + b*(d+L+x[i]) + b*(d+x[i])
        return der

    def objective_hess(self, x):
        a = self.config.get_pricing_parameters()[2]
        b = self.config.get_pricing_parameters()[1]
        s = self.config.get_schedule_length()
        hess = np.zeros((s, s))
        for i in range(s):
            d = self.forecast_demand[self.cur_player][i]
            L = self.L[self.cur_player][i]

            hess[i][i] = 2*(a*(3*d + 2*L + 3*x[i]) + b)
        return hess

    def find_optimal_response(self):
        p = self.cur_player
        x_0 = self.schedules[p]

        lb = np.empty(self.config.get_schedule_length())
        lb.fill(-self.players[p].get_initial_storage())
        A = 1.0 * np.eye(self.config.get_schedule_length())
        for i in range(self.config.get_schedule_length()):
            for j in range(self.config.get_schedule_length()):
                if j < i:
                    A[i][j] = 1

        ub = np.empty(self.config.get_schedule_length())
        ub.fill(np.inf)
        # cons = {'type': 'ineq', 'fun': lambda x: -(A @ x - lb)}
        linear_constraint = LinearConstraint(A, lb, ub)

        bounds = Bounds(-1.0*self.forecast_demand[p], np.ones(self.config.get_schedule_length())*self.players[p].get_storage_rate())

        if self.config.get_debug_flag():
            options = {'disp': True, 'xtol': 0.000001}
        else:
            options = {'xtol': 0.000001}
        res = minimize(fun = self.objective,
                       x0 = x_0,
                       method = 'trust-constr',
                       jac = self.objective_der,
                       hess = self.objective_hess,
                       constraints = linear_constraint,
                       bounds = bounds,
                       options = options)
        return res.x

    def check_for_convergence(self, solution):
        val = 0
        for p in self.players.keys():
            for i in range(self.config.get_schedule_length()):
                val += (self.schedules[p][i] - solution[p][i])**2

        val = np.sqrt(val)
        val /= self.config.get_schedule_length()

        result_flag = True if val < self.config.get_eps() else False
        return result_flag, val

    def adjust_schedules_for_real_demand(self):
        # keep track of storage
        storage = self.create_empty_int_arrays()
        new_schedules = self.create_empty_int_arrays()
        final_storage = self.create_empty_int_arrays()

        for p in self.players.keys():
            # sort out initial state and extend by one to evaluate final storage state
            storage[p][0] = self.players[p].get_initial_storage()
            storage[p] = np.append(storage[p], 0)

            d = self.players[p].get_game_demand()
            for i in range(self.config.get_schedule_length()):
                # adding to storage is currently not restricted
                # check only when using the stored PPE
                cur_decision = self.schedules[p][i]
                if cur_decision < 0:
                    temp = np.minimum(storage[p][i], d[i])
                    if abs(cur_decision) > temp:
                        cur_decision = -temp

                storage[p][i+1] += storage[p][i] + cur_decision
                new_schedules[p][i] = cur_decision
            final_storage[p] = storage[p][-1]

        self.schedules = new_schedules
        return final_storage

    def reset_for_repetition(self, new_initial_storage):
        # reset dates
        self.config.set_start_date(self.config.get_start_date() + timedelta(days=self.config.get_schedule_length()))

        # update initial storage for each player
        for p in self.players.keys():
            self.players[p].set_initial_storage(new_initial_storage[p])
            self.players[p].set_start_date(self.config.get_start_date())

        self.forecast_demand = self.compute_forecast_demand(self.config.get_forecast_error())

    # getter
    def get_players(self):
        return self.players

    def get_schedules(self):
        return self.schedules

    def get_fc_demand(self):
        return self.forecast_demand

    def get_demand(self):
        temp = self.create_empty_int_arrays()
        for p in self.players.keys():
            d = self.players[p].get_game_demand()
            for i in range(self.config.get_schedule_length()):
                temp[p][i] = d[i]
        return temp

    def get_dates(self):
        dates = []
        for i in range(self.config.get_schedule_length()):
            dates.append((self.config.get_start_date() + timedelta(days=i)).strftime("%d/%m"))
        return dates
