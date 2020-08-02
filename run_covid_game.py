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

    # important parameters
    # matthias:
    start_dates = ["31/01/2020", "07/02/2020", "28/02/2020", "11/03/2020"]
    # khaled:
    # start_dates = ["20/03/2020"]

    demands = ["./data/Demand_Oct_Wave.csv",
               "./data/Demand_Nov_Wave.csv",
               "./data/Demand_Dec_Wave.csv",
               "./data/Demand_Jan_Wave.csv",
               "./data/Demand_Feb_Wave.csv"]
    end_dates = ["20/01/2021", "20/02/2021", "20/03/2021", "20/04/2021", "20/05/2021"]

    storages = [0.0, 4.0, 9.0, 14.0, 19.0]

    for date in start_dates:
        for demand, end_date in zip(demands, end_dates):
            for storage in storages:
                config.reset_for_new_configuration(new_start_date=date,
                                                   new_end_date=end_date,
                                                   new_demand_file=demand,
                                                   new_additional_storage=storage)

                game = Game(config)
                _ = game.solve_game()
                plot_all(game, config.get_output_path())
                game.write_results_to_file()

    print("completed simulation, good bye")


if __name__ == '__main__':
    main()
    exit(0)
