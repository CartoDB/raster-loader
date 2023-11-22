from itertools import islice


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


def sql_quote(value: any) -> str:
    if isinstance(value, str):
        value = value.replace("\\", "\\\\")
        return f"'''{value}'''"
    return str(value)
