"""
=========================================================
Review Sense AI
PyTorch Dataset
=========================================================
"""

import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset

from config.config import cfg


class ReviewDataset(Dataset):

    def __init__(self):

        self.data = pd.read_csv(cfg.TRAIN_CLEANED)

        with open(cfg.VOCAB_FILE, "rb") as file:
            vocab = pickle.load(file)

        self.word2idx = vocab["word2idx"]


    # =====================================================
    # TEXT ENCODING
    # =====================================================

    def encode_text(self, text):

        tokens = str(text).split()

        sequence = []

        for token in tokens:

            sequence.append(
                self.word2idx.get(
                    token,
                    self.word2idx[cfg.UNK_TOKEN]
                )
            )


        if len(sequence) > cfg.MAX_SEQUENCE_LENGTH:

            sequence = sequence[:cfg.MAX_SEQUENCE_LENGTH]

            length = cfg.MAX_SEQUENCE_LENGTH


        else:

            length = len(sequence)

            sequence += [
                self.word2idx[cfg.PAD_TOKEN]
            ] * (
                cfg.MAX_SEQUENCE_LENGTH - len(sequence)
            )


        length = max(length, 1)


        return (
            torch.tensor(
                sequence,
                dtype=torch.long
            ),
            length
        )


    # =====================================================
    # DATASET LENGTH
    # =====================================================

    def __len__(self):

        return len(self.data)



    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        text = row["text"]

        label = row["label"]


        text_tensor, length = self.encode_text(text)


        length_tensor = torch.tensor(
            length,
            dtype=torch.long
        )


        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )


        return (
            text_tensor,
            length_tensor,
            label_tensor
        )