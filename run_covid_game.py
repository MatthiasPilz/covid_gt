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
    game = Game(config)

    for i in range(config.get_repeat()):
        game.display()

        new_initial_storage = game.solve_game()

        if config.get_plot_flag():
            plot_all(game)
        output_files(game, config.get_output_path(), i)

        game.reset_for_repetition(new_initial_storage)


if __name__ == '__main__':
    main()
    exit(0)
