# pytest configuration file
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--runintegration",
                     action="store_true",
                     help="run integration tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow_test" in item.keywords:
            item.add_marker(skip_slow)

    if config.getoption("--runintegration"):
        # --runintegration given in cli: do not skip integration tests
        return

    skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
    for item in items:
        if "integration_test" in item.keywords:
            item.add_marker(skip_integration)
