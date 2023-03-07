import argparse
from cli import setup_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_parser(parser)

    args, _ = parser.parse_known_args()
    if args.train:
        import train

        train.main(args)
    if args.test:
        import test

        test.main(args)
