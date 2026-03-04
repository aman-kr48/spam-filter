import argparse
import sys

from loguru import logger

from spam_filter.config import merge_configs, parse_file
from spam_filter.training import train


def main(argv: list[str] | None = None) -> int:

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO", serialize=True)

    parser = argparse.ArgumentParser(description="Train the spam filter model")

    parser.add_argument("files", nargs="+", help="Configuration files")

    args = parser.parse_args(argv)
    configs = [parse_file(file) for file in args.files]

    logger.info("Loaded {} config files", len(configs))

    config = merge_configs(configs)

    logger.debug("Final config: {}", config)

    model, X_test, y_test = train(config)

    logger.info("Training completed")
    logger.info("Test samples: {}", len(X_test))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
