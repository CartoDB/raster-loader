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
