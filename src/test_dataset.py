from torch.utils.data import DataLoader

from src.dataset import ReviewDataset


dataset = ReviewDataset()

print("Dataset Size :", len(dataset))

review, label = dataset[0]

print()

print("Sequence Shape :", review.shape)

print("Label :", label)

loader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=True

)

reviews, labels = next(iter(loader))

print()

print("Batch Shape :", reviews.shape)

print("Labels Shape :", labels.shape)