import pandas as pd
import sys

sys.path.append("../src")

import itnpy
import itnpy.vocab as vocab


if __name__ == "__main__":
    path = "../assets/vocab.csv"

    df = vocab.get_dataframe(path)
    word2number = vocab.get_word2number_dict(df)

    while True:
        spoken = input("Enter spoken form text here: ")
        spoken = spoken.strip()

        digit = itnpy.inverse_normalize_numbers(spoken.split(), word2number)
        digit = " ".join(digit)

        print()
        df = [{"[spoken]".upper(): spoken, "[digit]".upper(): digit}]
        df = pd.DataFrame(df)
        df = df.set_index("[spoken]".upper())
        df = df.T
        print(df.to_string(justify="left"))
        print()
