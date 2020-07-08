import argparse
import pprint
import os
import yaml
import datetime

from src.game import Game
from src.postprocessing import plot_all, calc_savings, output_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      "-c", required=True, help="config file location")
    parser.add_argument("--output", "-o", required=True, help="output location within results folder")
    parser.add_argument("--plot", "-p", help="flag for instant plotting", action='store_true')
    parser.add_argument("--write", "-w", help="flag for writing output to files", action='store_true')
    parser.add_argument("--repeat", "-r", help="the number of repetitions of the game", default=1)

    return parser.parse_args()


def create_output_dir(loc):
    output_location = "./results/" + loc + "/"
    try:
        os.makedirs(output_location)
        print("Directory ", output_location, " Created ")
    except FileExistsError:
        print("Directory ", output_location, " already exists")


def copy_config_to_output_dir(config_file, output_loc):
    with open(config_file) as f:
        list_doc = yaml.load(f, Loader=yaml.SafeLoader)

    file_name = output_loc + "parameter_file_initial.yaml"
    with open(file_name, "w") as f:
        yaml.dump(list_doc, f, Dumper=yaml.SafeDumper, default_flow_style=False)

    return file_name


def update_config_file(config_file, output_loc, repeat_counter, schedules, players):
    with open(config_file) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    # for the moment only update time variables (and storage later)
    old_start_date = datetime.datetime.strptime(params['start-date'], '%d/%m/%Y')
    shift = params['schedule-length']

    new_start_date = old_start_date + datetime.timedelta(days=shift)
    params['start-date'] = new_start_date.strftime('%d/%m/%Y')

    params = update_initial_storage(schedules, players, params)

    file_name = output_loc + "parameter_file" + str(repeat_counter) + ".yaml"
    with open(file_name, "w") as f:
        yaml.dump(params, f, Dumper=yaml.SafeDumper, default_flow_style=False)

    return file_name


def update_initial_storage(schedules, players, params):
    for (i, p) in enumerate(players):
        initial_state = params['initial-state'][i]
        for j in range(len(schedules[p])):
            initial_state += schedules[p][j]

        params['initial-state'][i] = initial_state
    return params


def main():
    args = parse_args()
    create_output_dir(args.output)
    output_loc = "./results/" + args.output + "/"
    config_file = copy_config_to_output_dir(args.config, output_loc)

    for i in range(int(args.repeat)):
        game = Game(config_file)
        if game.debug_state():
            game.display()

        flag_success = game.solve_game()

        if args.plot:
            # plotting
            plot_all(game.get_players(), game.get_demand(), game.get_schedules(), game.get_dates(), game.get_fc_demand())

        if args.write:
            # output some stats to terminal
            costs, costs_ref = game.calc_costs_for_all()
            print("reference costs:")
            pprint.pprint(costs_ref)
            print("costs with game:")
            pprint.pprint(costs)
            print("resulting savings")
            pprint.pprint(calc_savings(costs, costs_ref, game.get_players()))

            # output to file
            output_files(game.get_dates(),
                         game.get_players(),
                         game.get_demand(),
                         game.get_schedules(),
                         game.get_fc_demand(),
                         output_loc,
                         i)

            if i < int(args.repeat)-1:
                config_file = update_config_file(config_file,
                                                 output_loc,
                                                 i+1,
                                                 game.get_schedules(),
                                                 game.get_players())


if __name__ == '__main__':
    main()
    exit(0)
