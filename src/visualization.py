
from src.dataset import ReviewDataset
import pickle
from config.config import cfg

dataset = ReviewDataset()

with open(cfg.VOCAB_FILE, "rb") as f:
    vocab = pickle.load(f)

reverse_vocab = vocab["idx2word"]

for i in range(10):

    x, y = dataset[i]

    words = [
        reverse_vocab.get(token.item(), "<UNK>")
        for token in x
        if token.item() != 0
    ]

    print("Review:")
    print(" ".join(words[:40]))
    print("Label:", y.item())
    print("-"*60)
