import argparse


def get_base_parser(description: str):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--resources",
        type=str,
        choices=["low", "mid", "high"],
        default="mid",
        nargs="?",
        help="Resource attribute variation.",
    )
    parser.add_argument(
        "--concentration",
        type=str,
        choices=["low", "mid", "high"],
        default="mid",
        nargs="?",
        help="Concentration attribute variation.",
    )
    parser.add_argument(
        "--variability",
        type=str,
        choices=["low", "mid", "high"],
        default="mid",
        nargs="?",
        help="Variability attribute variation.",
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["low", "mid", "high"],
        default="mid",
        nargs="?",
        help="Quality attribute variation.",
    )

    return parser
