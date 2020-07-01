import time
import argparse
import yaml
import pprint


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      "-c", required=True, help="config file location")

    return parser.parse_args()


def read_parameters():
    args = _parse_args()
    configFile = args.config
    params = {}
    try:
        with open(configFile, 'r') as file:
            # all the parameters from the config file
            params = yaml.load(file, Loader=yaml.SafeLoader)
    except Exception as e:
        print('*** Error reading the config file - ' + str(e))

    return params


def main():
    start = time.time()

    params = read_parameters()
    pprint.pprint(params)

    end = time.time()
    print("Simulation took: ", end - start)


if __name__ == '__main__':
    main()
    exit(0)
