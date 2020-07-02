import time
import argparse

from src.game import Game
from src.postprocessing import plot_all


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
        plot_all(game.get_players(), game.get_demand(), game.get_schedules(), game.get_dates())

    if args.write:
        # output to files
        pass


if __name__ == '__main__':
    main()

    exit(0)
