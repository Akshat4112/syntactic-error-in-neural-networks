import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_REGEX = re.compile(r"[.(),``!?:;\-='...@_]")


def clean_text(s):
    return ' '.join(CLEAN_REGEX.sub(" ", s).split())


class dataset:
    def __init__(self):
        pass

    def download_data(self):
        pass

    def preprocess_data(self):
        data_dir = PROJECT_ROOT / 'data'
        raw_dir = data_dir / 'rnn_agr_simple'

        df_train = pd.read_csv(raw_dir / 'numpred.train', sep='\t', names=['POS', "Preamble"])
        df_val = pd.read_csv(raw_dir / 'numpred.val', sep='\t', names=['POS', "Preamble"])

        df_train["Preamble"] = df_train["Preamble"].apply(clean_text)
        df_val["Preamble"] = df_val["Preamble"].apply(clean_text)

        # VBZ is singular and VBP is plural
        df_train.loc[df_train["POS"] == "VBZ", "POS"] = 0
        df_train.loc[df_train["POS"] == "VBP", "POS"] = 1

        df_val.loc[df_val["POS"] == "VBZ", "POS"] = 0
        df_val.loc[df_val["POS"] == "VBP", "POS"] = 1

        train_df = pd.DataFrame({"labels": df_train['POS'], "text": df_train['Preamble']})
        test_df = pd.DataFrame({"labels": df_val['POS'], "text": df_val['Preamble']})

        train_df.to_csv(data_dir / 'train_df.csv')
        test_df.to_csv(data_dir / 'test_df.csv')

        print("Data Preprocessed and Saved...")
