import os
from glob import glob


def pytest_addoption(parser):
    parser.addoption(
        "--notebook-path",
        action="append",
        default=[],
        help="path to notebooks directory",
    )


def pytest_generate_tests(metafunc):
    """
    Generate tests for all notebooks in the notebooks directory.

    Notes
    -----
    - notebook-path can be passed multiple times to specify multiple directories.
    """
    if len(metafunc.config.getoption("notebook_path")) > 0:
        # If the option is passed, use it.
        dirs = metafunc.config.getoption("notebook_path")
    else:
        # Otherwise, search within all folders located in modelscripts.
        dirs = ["**"]

    # Notebooks
    filepaths = sorted(sum([glob(os.path.join("modelscripts", dir, "*.py")) for dir in dirs], []))
    filepaths += sorted(sum([glob(os.path.join("modelscripts", dir, "*.ipynb")) for dir in dirs], []))
    metafunc.parametrize("notebook_path", filepaths)
