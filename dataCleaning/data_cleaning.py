import random
import pandas as pd
import numpy as np
import math
import os


def load_and_clean_titanic():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(SCRIPT_DIR, "titanic.csv"))
    # strip whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    # fill numeric
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    # fill categorical
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df







