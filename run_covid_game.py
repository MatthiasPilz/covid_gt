import argparse
from src.game import Game
from src.config import Config
from src.postprocessing import plot_all, output_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     "-c", required=True, help="config file location")

    return parser.parse_args()


def main():
    args = parse_args()
    config_file = args.config
    config = Config(config_file)

    dates = ["10/04/2020", "23/03/2020"]
    lengths = [10, 20]
    demands = ["./data/covid_allDemand.csv", "./data/second_peak_Dec01.csv"]
    storages = [0.0, 10.0]

    for date, length in zip(dates, lengths):
        for demand in demands:
            for storage in storages:
                config.reset_for_new_configuration(date, demand, storage, length)
                game = Game(config)
                _ = game.solve_game()
                plot_all(game, config.get_output_path())
                game.write_results_to_file()

    print("completed simulation, good bye")


if __name__ == '__main__':
    main()
    exit(0)
