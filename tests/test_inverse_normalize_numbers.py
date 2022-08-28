import pandas as pd
import pytest
import sys

sys.path.append("src")

import itnpy
import itnpy.vocab as vocab


def get_word2number_dict(path="assets/vocab.csv"):
    df = vocab.get_dataframe(path)
    return vocab.get_word2number_dict(df)


@pytest.mark.parametrize(
    "path",
    [
        "tests/assets/inverse_normalize_numbers/passing.csv",
        "tests/assets/inverse_normalize_numbers/failing.csv",
    ],
)
def test_inverse_normalize_numbers(path):
    df = pd.read_csv(path, dtype={"input": object, "output": object})
    df = df.fillna("")
    word2number = get_word2number_dict()

    for _, row in df.iterrows():
        tokens = row["input"].split()
        digit = row["output"]
        assert " ".join(itnpy.inverse_normalize_numbers(tokens, word2number)) == digit


@pytest.mark.parametrize(
    "path",
    [
        "tests/assets/inverse_normalize_numbers/100k.csv",
    ],
)
def test_inverse_normalize_numbers_100k(path):
    df = pd.read_csv(
        path, dtype={"data": object, "labels": object, "reference": object}
    )
    df = df.fillna("")
    word2number = get_word2number_dict()

    for _, row in df.iterrows():
        tokens = row["data"].split()
        digit = row["reference"]
        assert " ".join(itnpy.inverse_normalize_numbers(tokens, word2number)) == digit

        tokens = row["labels"].split()
        digit = row["reference"]
        assert " ".join(itnpy.inverse_normalize_numbers(tokens, word2number)) == digit
