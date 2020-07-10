import time
import numpy as np
import random
from .player import Player
from .config import Config
from scipy.optimize import minimize, LinearConstraint, Bounds


class Game:
    def __init__(self, config_file):
        self.config = Config(config_file)

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
                temp[p] = self.players[p].get_game_demand()[i] - int(np.random.normal(error_mean, error_stdv))
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
                self.update_load_other_players(p)
                self.schedules[p] = self.find_optimal_response(p)

            flag_convergence, achieved_eps = self.check_for_convergence(solution_profile)
            flag_time_is_up = True if time.time() > end_time else False
            num_iterations += 1

            print("*** finished iteration number {} with eps = {}".format(num_iterations, achieved_eps))
            print("###############################################")
            print("")

        print("total number of iterations: {}".format(num_iterations))
        print("execution time for solver: {:.3f}s".format(time.time()-start_time))

        self.adjust_schedules_for_real_demand()

        return flag_convergence

    def update_load_other_players(self, p):
        self.L[p] = np.zeros(self.config.get_schedule_length())
        for other in self.players.keys():
            if other == p:
                continue
            for i in range(self.config.get_schedule_length()):
                self.L[p][i] = self.players[p].get_game_demand()[i] + self.schedules[other][i]

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
            d = self.forecast_demand[i]
            L = self._load_other_players[i]

            costs_i = a*d**3 + 2*a*d**2*L + 3*a*d**2*x[i] + a*d*L**2 + 4*a*d*L*x[i] + 3*a*d*x[i]**2 + \
                a*L**2*x[i] + 2*a*L*x[i]**2 + a*x[i]**3 + b*d**2 + b*d*L + 2*b*d*x[i] + b*L*x[i] + b*x[i]**2

            costs += costs_i
        return costs

    def objective_der(self, x):
        a = self._pricing_parameter[2]
        b = self._pricing_parameter[1]
        der = np.zeros_like(x)
        for i in range(self._schedule_length):
            d = self._fc_demand_current_player[i]
            L = self._load_other_players[i]

            der[i] = a*((d+x[i])+L)**2 + 2*a*(d+x[i])*(d+x[i]+L) + b*(d+L+x[i]) + b*(d+x[i])
        return der

    def objective_hess(self, x):
        a = self._pricing_parameter[2]
        b = self._pricing_parameter[1]
        hess = np.zeros((self._schedule_length, self._schedule_length))
        for i in range(self._schedule_length):
            d = self._fc_demand_current_player[i]
            L = self._load_other_players[i]

            hess[i][i] = 2*(a*(3*d + 2*L + 3*x[i]) + b)
        return hess

    def find_optimal_response(self, p):

        x_0 = self.__schedules[p]

        lb = np.empty(self._schedule_length)
        lb.fill(-self._initial_state_current_player)
        A = 1.0 * np.eye(self._schedule_length)
        for i in range(self._schedule_length):
            for j in range(self._schedule_length):
                if j < i:
                    A[i][j] = 1

        ub = np.empty(self._schedule_length)
        ub.fill(np.inf)
        # cons = {'type': 'ineq', 'fun': lambda x: -(A @ x - lb)}
        linear_constraint = LinearConstraint(A, lb, ub)

        bounds = Bounds(-1.0*self._fc_demand_current_player, np.ones(self._schedule_length)*self._storage_rate)

        if self._debug_flag:
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
        for p in self._players:
            for i in range(self._schedule_length):
                val += (self.__schedules[p][i] - solution[p][i])**2

        val = np.sqrt(val)
        val /= self._schedule_length

        result_flag = True if val < self._eps else False
        return result_flag, val

    def set_random_schedule(self):
        for p in self._players:
            for i in range(self._schedule_length):
                self.__schedules[p][i] = random.randint(-10000, 10000)

    def adjust_schedules_for_real_demand(self):
        # keep track of storage
        storage = self.create_empty_arrays()
        new_schedules = self.create_empty_arrays()

        for k, p in enumerate(self._players):
            # sort out initial state and extend by one to evaluate final storage state
            storage[p][0] = self._initial_states[k]
            storage[p] = np.append(storage[p], 0)

            for i in range(self._schedule_length):
                # adding to storage is currently not restricted
                # check only when using the stored PPE
                cur_decision = self.__schedules[p][i]
                if cur_decision < 0:
                    temp = np.minimum(storage[p][i], self.__demand[p][i+self._start_index])
                    if abs(cur_decision) > temp:
                        cur_decision = -temp

                storage[p][i+1] += storage[p][i] + cur_decision
                new_schedules[p][i] = cur_decision

        self.__schedules = new_schedules

    # getter
    def get_players(self):
        return self._players

    def get_schedules(self):
        return self.__schedules

    def get_fc_demand(self):
        fc_demand = self.create_empty_arrays()
        for p in self._players:
            for i in range(self._schedule_length):
                fc_demand[p][i] = self.__fc_demand[p][i + self._start_index]
        return fc_demand

    def get_demand(self):
        demand = self.create_empty_arrays()
        for p in self._players:
            for i in range(self._schedule_length):
                demand[p][i] = self.__demand[p][i + self._start_index]
        return demand

    def get_dates(self):
        dates = []
        for i in range(self._schedule_length):
            dates.append(self.__demand['date'][i + self._start_index].strftime("%d/%m"))
        return dates

    def get_initial_state(self, p):
        for (i, player) in enumerate(self._players):
            if p == player:
                return self._initial_states[i]

    '''
    def check_initialisation(self):
        assert self._num_players == self.__demand.shape[1] - 1, "number of players should match number of columns in file!"
        flag_pricing_parameter = (self._pricing_parameter[1] != 0.0 or self._pricing_parameter[2] != 0.0)
        assert flag_pricing_parameter, "at least one of the cost-coefficients needs to be different from zero!"
        assert self._pricing_parameter[0] == 0, "current implementation assumes zero constant term!"
        assert (self._start_date == self.__demand['date']).any(), "start date needs to be within data range"
        assert (self._end_date == self.__demand['date']).any(), "end date needs to be within data range"
        assert self._storage_rate > 0, "storage rate needs to be larger than zero"
        assert self._usage_rate < 0, "usage rate needs to be smaller than zero"
        assert self._max_capacity >= 0, "max capacity needs to be larger than zero"
        assert self._min_capacity >= 0, "min capacity needs to be non-negative"
        flag_initial_state = True
        for i in range(self._num_players):
            flag_cur = self._min_capacity <= self._initial_states[i] <= self._max_capacity
            flag_initial_state = flag_initial_state and flag_cur
        assert flag_initial_state, "initial state not within the storage limits"
        assert self._max_time > 0, "iteration time needs to be larger than zero"
        assert self._max_iter >= 1, "need to perform at least one iteration"

    def calc_costs_for_all(self):
        L = self.create_empty_arrays()
        for p in self._players:
            L[p] = self.calc_current_load_of_others(p)

        L_ref = self.create_empty_arrays()
        for p in self._players:
            L_ref[p] = self.calc_reference_load_of_others(p)

        costs = dict()
        costs_ref = dict()
        for p in self._players:
            costs[p] = 0
            costs_ref[p] = 0
            for i in range(self._schedule_length):
                total_load = self.__demand[p][i+self._start_index] + L[p][i] + self.__schedules[p][i]
                total_load_ref = self.__demand[p][i+self._start_index] + L_ref[p][i]

                total_price = self._pricing_parameter[2] * total_load**2 + \
                              self._pricing_parameter[1] * total_load
                total_price_ref = self._pricing_parameter[2] * total_load_ref**2 + \
                                  self._pricing_parameter[1] * total_load_ref

                costs[p] += (self.__demand[p][i+self._start_index] + self.__schedules[p][i])*total_price
                costs_ref[p] += self.__demand[p][i+self._start_index] * total_price_ref
            costs[p] = int(costs[p])
            costs_ref[p] = int(costs_ref[p])

        return costs, costs_ref
    '''