import argparse
from src.game import Game


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     "-c", required=True, help="config file location")

    return parser.parse_args()


def main():
    args = parse_args()
    config_file = args.config

    game = Game(config_file)
    game.display()


if __name__ == '__main__':
    main()
    exit(0)
