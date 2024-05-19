import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


def test_nb_has_cleared_output(notebook_path):
    assert os.path.exists(notebook_path), f"filepath {notebook_path} does not exist"

    if notebook_path.endswith(".py"):
        pytest.skip('pass')
    elif notebook_path.endswith(".ipynb"):
        has_cleared_output(notebook_path)
    else:
        raise ValueError(f"Invalid file extension for {notebook_path}")
    

def has_cleared_output(filepath):
    """Check if the output cells have been cleared."""
    with open(filepath) as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    for cell in nb.cells:
        if cell.cell_type == "code" and cell.outputs:
            # Outputs should be an empty list
            raise AssertionError(f"Output is not cleared for {filepath}")


def test_print_path(notebook_path):
    assert os.path.exists(notebook_path), f"filepath {notebook_path} does not exist"

    if notebook_path.endswith(".py"):
        pythonscript_test_exec(notebook_path)
    elif notebook_path.endswith(".ipynb"):
        notebook_test_exec(notebook_path)
    else:
        raise ValueError(f"Invalid file extension for {notebook_path}")


def notebook_test_exec(filepath):
    filedir = os.path.dirname(filepath)

    with open(filepath) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    result = ep.preprocess(nb, {"metadata": {"path": filedir}})

    try:
        assert result is not None, f"Got empty notebook for {filepath}"
    except Exception:
        assert False, f"Failed executing {filepath}.\n\n{result}"


def pythonscript_test_exec(filepath):
    with open(filepath) as fh:
        # , globals={"__file__": filepath, "__name__": "__main__"}, locals={"__file__": filepath, "__name__": "__main__"}
        exec(fh.read())
