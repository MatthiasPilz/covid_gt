import time
import argparse

from src.game import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      "-c", required=True, help="config file location")

    return parser.parse_args()


def main():
    config_file = parse_args().config

    game = Game(config_file)
    if game.debug_state():
        game.display()

    if game.solve_game():
        print("Iteration converged!")
    else:
        print("Iteration did not converge!")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Simulation took {:.3f} seconds.".format(end - start))

    exit(0)
