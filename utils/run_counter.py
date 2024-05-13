import os

FILENAME = "logs/run_counter.txt"
if not os.path.exists(FILENAME):
    with open(FILENAME, "w") as file:
        file.write("0")


def increment_run_counter() -> int:
    """Increments and returns the run counter stored in the specified file."""
    try:
        with open(FILENAME, "r+") as file:
            iter_num = int(file.read().strip()) + 1
            file.seek(0)
            file.write(str(iter_num))
            file.truncate()
    except FileNotFoundError:
        # If file doesn't exist, create it and initialize iter_num to 1
        with open(FILENAME, "w") as file:
            iter_num = 1
            file.write(str(iter_num))
    return iter_num


def read_run_counter() -> int:
    """Reads the current run counter from the specified file."""
    try:
        with open(FILENAME, "r") as file:
            return int(file.read().strip())
    except FileNotFoundError:
        # If file doesn't exist, return 0 as default
        return 0
