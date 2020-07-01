import time
import argparse

from src.game import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      "-c", required=True, help="config file location")

    return parser.parse_args()


def main():
    start = time.time()

    args = parse_args()
    config_file = args.config

    game = Game(config_file)
    if game.debug_state():
        game.display()

    end = time.time()
    print("Simulation took {:.3f} seconds.".format(end - start))


if __name__ == '__main__':
    main()
    exit(0)
