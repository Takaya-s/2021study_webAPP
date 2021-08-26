import codecs
import pandas as pd
import streamlit as st


@st.cache
def read_shift_jis_data(filepath: str) -> pd.DataFrame:
    """
    Read shift-jis dataframe
    """
    # Original code
    with codecs.open(filepath, "r", "Shift-JIS", "ignore") as file:
        df: pd.DataFrame = pd.read_csv(file, encoding="utf-8")

    return df
