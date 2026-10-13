"""
=========================================================
Review Sense AI
Single Review Prediction
=========================================================
"""

import torch
import pickle

from config.config import cfg
from src.model import ReviewLSTM


# -------------------------------------------------
# Load vocabulary
# -------------------------------------------------

with open(cfg.VOCAB_FILE, "rb") as file:
    vocab = pickle.load(file)

word2idx = vocab["word2idx"]


# -------------------------------------------------
# Load model
# -------------------------------------------------

device = cfg.DEVICE

model = ReviewLSTM().to(device)

model.load_state_dict(
    torch.load(
        cfg.CHECKPOINT_FILE,
        map_location=device
    )
)

model.eval()


# -------------------------------------------------
# Text Encoding
# -------------------------------------------------

def encode_text(text):

    tokens = str(text).lower().split()

    sequence = []

    for token in tokens:

        sequence.append(
            word2idx.get(
                token,
                word2idx[cfg.UNK_TOKEN]
            )
        )


    if len(sequence) > cfg.MAX_SEQUENCE_LENGTH:

        sequence = sequence[:cfg.MAX_SEQUENCE_LENGTH]
        length = cfg.MAX_SEQUENCE_LENGTH

    else:

        length = len(sequence)

        sequence += [
            word2idx[cfg.PAD_TOKEN]
        ] * (
            cfg.MAX_SEQUENCE_LENGTH - len(sequence)
        )


    length = max(length,1)


    return (
        torch.tensor(sequence,dtype=torch.long),
        torch.tensor(length,dtype=torch.long)
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

def predict(review):

    sequence, length = encode_text(review)


    sequence = sequence.unsqueeze(0).to(device)
    length = length.unsqueeze(0).to(device)


    with torch.no_grad():

        output = model(
            sequence,
            length
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    label = prediction.item()

    confidence = confidence.item()


    if label == 1:

        sentiment = "Positive 😊"

    else:

        sentiment = "Negative 😞"


    return sentiment, confidence



# -------------------------------------------------
# Test
# -------------------------------------------------

if __name__ == "__main__":


    review = input(
        "\nEnter review: "
    )


    sentiment, confidence = predict(review)


    print("\nResult:")
    print(
        sentiment
    )

    print(
        f"Confidence: {confidence*100:.2f}%"
    )