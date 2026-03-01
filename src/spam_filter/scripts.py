import argparse

from spam_filter.training import train
from spam_filter.config import parse_file
from spam_filter.config import merge_configs



def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Train the spam filter model"
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Configuration files"
    )

    args = parser.parse_args(argv)
    configs = [parse_file(file) for file in args.files]
    config = merge_configs(configs)

    print("Final config:", config)


    model, X_test, y_test = train(config)

    print("Training completed.")
    print("Test samples:", len(X_test))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())