import builtins
from unittest.mock import patch

from raster_loader.utils import ask_yes_no_question


def test_ask_yes_no_question_answer_yes():
    with patch.object(builtins, "input", lambda _: "yes"):
        assert ask_yes_no_question("Test?") is True


def test_ask_yes_no_question_answer_no():
    with patch.object(builtins, "input", lambda _: "no"):
        assert ask_yes_no_question("Test?") is False
