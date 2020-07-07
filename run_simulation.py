import time
import argparse
import pprint

from src.game import Game
from src.postprocessing import plot_all, calc_savings, output_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      "-c", required=True, help="config file location")
    parser.add_argument("--plot", "-p", help="flag for instant plotting", default=True)
    parser.add_argument("--write", "-w", help="flag for writing output to files", default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    game = Game(args.config)
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
        output_files(game.get_dates(), game.get_players(), game.get_demand(), game.get_schedules(), game.get_fc_demand())


if __name__ == '__main__':
    main()
    exit(0)
