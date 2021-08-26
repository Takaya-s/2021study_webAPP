import numpy as np
import pandas as pd
import streamlit as st
import os
import itertools
import sys
from functools import reduce
from pathlib import Path
import datetime 

import codecs
import pickle



class Data():
    def __init__(self, kind):
        self.kind = None
        self.path = None

    def scr(self, kind, username):
        self.kind = "scr"
        self.username = username
        self.path = r"C:\Users\{}\Box\IIIS\metab\21年ストレス試験MM\0_data\Screening\input\scr_rawdata_20210804.csv".format(self.username)

        with codecs.open(self.path, "r", "Shift-JIS", "ignore") as file:
            data = pd.read_csv(file, encoding = "utf-8")
        return data




