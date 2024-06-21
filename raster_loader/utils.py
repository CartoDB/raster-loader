from itertools import islice
import os
import re
import uuid


def ask_yes_no_question(question: str) -> bool:
    """Ask a yes or no question and return True or False."""
    yes_choices = ["yes", "y"]
    no_choices = ["no", "n"]

    while True:
        user_input = input(question)
        if user_input.lower() in yes_choices:
            return True
        elif user_input.lower() in no_choices:
            return False
        else:  # pragma: no cover
            print("Type yes or no")
            continue


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:  # pragma: no cover
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):  # noqa
        yield batch


def get_default_table_name(base_path: str, band):
    table = os.path.basename(base_path).split(".")[0]
    table = "_".join([table, "band", str(band), str(uuid.uuid4())])
    return re.sub(r"[^a-zA-Z0-9_-]", "_", table)
