"""
============================================================
Review Sense AI
Dataset Builder
============================================================

Purpose:
    • Read Amazon Review Polarity dataset
    • Validate dataset
    • Merge title + review
    • Convert labels
    • Shuffle dataset
    • Select 200,000 reviews
    • Save processed dataset

Author: Muhammad Usman
============================================================
"""

from pathlib import Path
import pandas as pd

from sklearn.utils import shuffle

from config.config import cfg


class DatasetBuilder:

    def __init__(self):

        self.input_file = cfg.TRAIN_CSV
        self.output_file = cfg.TRAIN_PROCESSED

        self.sample_size = 200000

    # -------------------------------------------------------

    def load_dataset(self):

        print("=" * 60)
        print("Loading Dataset")
        print("=" * 60)

        if not self.input_file.exists():

            raise FileNotFoundError(
                f"\nDataset not found:\n{self.input_file}"
            )

        df = pd.read_csv(
            self.input_file,
            header=None,
            names=[
                "label",
                "title",
                "review"
            ]
        )

        print(f"Rows Loaded : {len(df):,}")

        return df

    # -------------------------------------------------------

    def validate_dataset(self, df):

        print("\nValidating Dataset...")

        required_columns = [
            "label",
            "title",
            "review"
        ]

        for column in required_columns:

            if column not in df.columns:

                raise ValueError(
                    f"Missing column : {column}"
                )

        print("Validation Successful")

    # -------------------------------------------------------

    def merge_text(self, df):

        print("\nCombining Title + Review...")

        df["text"] = (
            df["title"].fillna("")
            + " "
            + df["review"].fillna("")
        )

        return df

    # -------------------------------------------------------

    def convert_labels(self, df):

        print("\nConverting Labels...")

        label_map = {

            1: 0,
            2: 1

        }

        df["label"] = df["label"].map(label_map)

        return df

    # -------------------------------------------------------

    def clean_dataframe(self, df):

        print("\nRemoving Missing Values...")

        df = df.dropna()

        df = df.drop_duplicates()

        return df

    # -------------------------------------------------------

    def sample_dataset(self, df):

        print("\nShuffling Dataset...")

        df = shuffle(
            df,
            random_state=cfg.RANDOM_STATE
        )

        print("Selecting 200,000 Reviews...")

        df = df.iloc[: self.sample_size]

        return df

    # -------------------------------------------------------

    def print_statistics(self, df):

        print("\nDataset Statistics")

        print("-" * 40)

        print(df["label"].value_counts())

        print("-" * 40)

        print(f"Total Samples : {len(df):,}")

    # -------------------------------------------------------

    def save_dataset(self, df):

        save_df = df[
            [
                "text",
                "label"
            ]
        ]

        save_df.to_csv(
            self.output_file,
            index=False
        )

        print("\nSaved Successfully")

        print(self.output_file)

    # -------------------------------------------------------

    def build(self):

        df = self.load_dataset()

        self.validate_dataset(df)

        df = self.merge_text(df)

        df = self.convert_labels(df)

        df = self.clean_dataframe(df)

        df = self.sample_dataset(df)

        self.print_statistics(df)

        self.save_dataset(df)


def main():

    builder = DatasetBuilder()

    builder.build()


if __name__ == "__main__":

    main()