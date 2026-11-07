"""Pytest configuration and custom flags."""



def pytest_addoption(parser):
    parser.addoption(
        "--mock",
        action="store_true",
        default=False,
        help="Run tests in mock CI mode without requiring full model weights",
    )


def pytest_collection_modifyitems(config, items):
    # If --mock is passed, select tests marked with 'mock' and skip tests marked with 'slow'
    if config.getoption("--mock"):
        selected = []
        deselected = []
        for item in items:
            if "mock" in item.keywords:
                selected.append(item)
            elif "slow" in item.keywords:
                deselected.append(item)
            else:
                selected.append(item)
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
