import codecs
import pandas as pd


def read_shift_jis_data(filepath: str) -> pd.DataFrame:
    """
    Read shift-jis dataframe
    """
    # Original code
    with codecs.open(filepath, "r", "Shift-JIS", "ignore") as file:
        df: pd.DataFrame = pd.read_csv(file, encoding="utf-8")

    # directly use `shift jis` or `cp932`
    # df: pd.DataFrame = pd.read_csv(file, encoding="cp932")
    # df: pd.DataFrame = pd.read_csv(file, encoding="shift jis")
    return df
