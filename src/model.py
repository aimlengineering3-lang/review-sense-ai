"""
=========================================================
Review Sense AI
LSTM Sentiment Classification Model
=========================================================
"""

import torch
import torch.nn as nn

from config.config import cfg


class ReviewLSTM(nn.Module):

    def __init__(self):

        super().__init__()


        # =================================================
        # Embedding Layer
        # =================================================

        self.embedding = nn.Embedding(
            num_embeddings=cfg.VOCAB_SIZE,
            embedding_dim=cfg.EMBEDDING_DIM,
            padding_idx=0
        )


        # =================================================
        # LSTM Layer
        # =================================================

        self.lstm = nn.LSTM(
            input_size=cfg.EMBEDDING_DIM,
            hidden_size=cfg.HIDDEN_DIM,
            num_layers=cfg.NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT
        )


        # =================================================
        # Dropout
        # =================================================

        self.dropout = nn.Dropout(
            cfg.DROPOUT
        )


        # =================================================
        # Classifier
        # =================================================

        self.fc = nn.Linear(
            cfg.HIDDEN_DIM,
            cfg.OUTPUT_DIM
        )


        self._init_forget_gate_bias()



    # =====================================================
    # Forget Gate Bias Initialization
    # =====================================================

    def _init_forget_gate_bias(self):

        for name, param in self.lstm.named_parameters():

            if "bias" in name:

                hidden_size = cfg.HIDDEN_DIM

                param.data[
                    hidden_size:2 * hidden_size
                ].fill_(1.0)



    # =====================================================
    # Forward Pass
    # =====================================================

    def forward(self, x, lengths):

        """
        x:
            (batch, sequence_length)

        lengths:
            (batch,)
        """


        embedded = self.embedding(x)


        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )


        packed_output, (hidden, cell) = self.lstm(
            packed_embedded
        )


        # Last LSTM layer hidden state

        hidden = hidden[-1]


        hidden = self.dropout(hidden)


        logits = self.fc(hidden)


        return logits



# =========================================================
# Test Model
# =========================================================

if __name__ == "__main__":

    model = ReviewLSTM()

    print(model)