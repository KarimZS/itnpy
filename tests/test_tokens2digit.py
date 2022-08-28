import pandas as pd
import pytest
import sys

sys.path.append("src")

import itnpy


@pytest.mark.parametrize(
    "path",
    [
        "tests/assets/tokens2digit/passing.csv",
        "tests/assets/tokens2digit/failing.csv",
    ],
)
def test_tokens2digit(path):
    df = pd.read_csv(path, dtype={"input": object, "output": object})
    df = df.fillna("")

    for _, row in df.iterrows():
        tokens = row["input"].split()
        digit = row["output"]
        assert itnpy.tokens2digit(tokens) == digit
