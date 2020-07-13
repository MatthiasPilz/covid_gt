import argparse
from src.game import Game
from src.postprocessing import plot_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     "-c", required=True, help="config file location")

    return parser.parse_args()


def main():
    args = parse_args()
    config_file = args.config

    game = Game(config_file)
    game.display()

    game.solve_game()

    plot_all(list(game.get_players().keys()), game.get_demand(), game.get_schedules(), game.get_dates(), game.get_fc_demand())


if __name__ == '__main__':
    main()
    exit(0)