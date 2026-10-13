"""
============================================================
Review Sense AI
Vocabulary Builder
============================================================
"""

import pickle
from collections import Counter

import pandas as pd

from config.config import cfg


class Vocabulary:

    def __init__(self):

        self.word2idx = {}
        self.idx2word = {}

        self.counter = Counter()

    # -----------------------------------------------------

    def load_dataset(self):

        print("=" * 60)
        print("Loading Cleaned Dataset")
        print("=" * 60)

        df = pd.read_csv(cfg.TRAIN_CLEANED)

        print(f"Reviews Loaded : {len(df):,}")

        return df

    # -----------------------------------------------------

    def tokenize(self, text):

        return str(text).split()

    # -----------------------------------------------------

    def count_words(self, df):

        print("\nCounting Words...")

        for text in df["text"]:

            self.counter.update(self.tokenize(text))

        print(f"Unique Words Found : {len(self.counter):,}")

    # -----------------------------------------------------

    def build_vocabulary(self):

        print("\nBuilding Vocabulary...")

        self.word2idx[cfg.PAD_TOKEN] = 0
        self.word2idx[cfg.UNK_TOKEN] = 1

        index = 2

        for word, frequency in self.counter.most_common():

            if frequency < cfg.MIN_FREQUENCY:
                continue

            if index >= cfg.VOCAB_SIZE:
                break

            self.word2idx[word] = index

            index += 1

        self.idx2word = {

            index: word

            for word, index in self.word2idx.items()

        }

        print(f"Vocabulary Size : {len(self.word2idx):,}")

    # -----------------------------------------------------

    def save(self):

        cfg.TOKENIZER_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(cfg.VOCAB_FILE, "wb") as file:

            pickle.dump(

                {

                    "word2idx": self.word2idx,

                    "idx2word": self.idx2word

                },

                file

            )

        print("\nVocabulary Saved Successfully")

        print(cfg.VOCAB_FILE)

    # -----------------------------------------------------

    def preview(self):

        print("\nFirst 20 Tokens\n")

        for word, index in list(self.word2idx.items())[:20]:

            print(f"{index:>5}   {word}")

    # -----------------------------------------------------

    def build(self):

        df = self.load_dataset()

        self.count_words(df)

        self.build_vocabulary()

        self.preview()

        self.save()


def main():

    vocab = Vocabulary()

    vocab.build()


if __name__ == "__main__":

    main()