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
        "--distribution",
        type=str,
        choices=["low", "mid", "high"],
        default="mid",
        nargs="?",
        help="Distribution attribute variation.",
    )

    parser.add_argument(
        "--num_clients", type=int, default=50, nargs="?", help="Number of clients."
    )

    parser.add_argument(
        "--trace_file",
        type=str,
        default="",
        nargs="?",
        help="Path to the trace file.",
    )

    parser.add_argument(
        "--client_config_file",
        type=str,
        default="",
        nargs="?",
        help="Path to the client config file.",
    )

    parser.add_argument(
        "--server_address",
        type=str,
        default="",
        nargs="?",
        help="Server address.",
    )

    parser.add_argument(
        "--batch_size",
        type=str,
        choices=["iid", "noniid"],
        default="iid",
        nargs="?",
        help="Batch size distribution.",
    )

    parser.add_argument(
        "--local_epochs",
        type=str,
        choices=["iid", "noniid"],
        default="iid",
        nargs="?",
        help="Local epochs distribution.",
    )

    parser.add_argument(
        "--data_volume",
        type=str,
        choices=["iid", "noniid"],
        default="iid",
        nargs="?",
        help="Data volume distribution.",
    )

    parser.add_argument(
        "--data_labels",
        type=str,
        choices=["iid", "noniid"],
        default="iid",
        nargs="?",
        help="Data labels distribution.",
    )

    return parser
