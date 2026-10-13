import torch

from src.model import ReviewLSTM

from config.config import cfg


model = ReviewLSTM()

print(model)

dummy = torch.randint(

    0,

    cfg.VOCAB_SIZE,

    (4, cfg.MAX_SEQUENCE_LENGTH)

)

output = model(dummy)

print()
print("Input Shape :", dummy.shape)
print("Output Shape:", output.shape)
print()

print("Input Shape :", dummy.shape)

print("Output Shape :", output.shape)