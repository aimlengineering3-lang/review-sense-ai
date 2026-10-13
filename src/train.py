import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

from config.config import cfg
from src.dataset import ReviewDataset
from src.model import ReviewLSTM


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training")

    # Batches now yield (sequences, lengths, labels) since
    # ReviewDataset.__getitem__ was updated to also return the
    # true (pre-padding) length of each review.

    for sequences, lengths, labels in progress:

        sequences = sequences.to(device)
        labels = labels.to(device)

        # lengths is intentionally NOT moved to the training
        # device -- pack_padded_sequence requires it on CPU
        # regardless of where the model/tensors live.

        optimizer.zero_grad()

        outputs = model(sequences, lengths)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5
        )

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=loss.item()
        )


    accuracy = correct / total

    return (
        total_loss / len(loader),
        accuracy
    )



def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():

        for sequences, lengths, labels in loader:

            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences, lengths)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    accuracy = correct / total


    return (
        total_loss / len(loader),
        accuracy
    )



def main():

    print("=" * 60)
    print("Review Sense AI Training")
    print("=" * 60)


    device = cfg.DEVICE

    print(
        f"Using Device : {device}"
    )


    if device.type == "cuda":

        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )


    dataset = ReviewDataset()


    train_size = int(
        len(dataset) * cfg.TRAIN_SIZE
    )

    valid_size = len(dataset) - train_size


    train_dataset, valid_dataset = random_split(
        dataset,
        [
            train_size,
            valid_size
        ],
        generator=torch.Generator().manual_seed(
            cfg.RANDOM_STATE
        )
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    model = ReviewLSTM().to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=1e-5
    )


    best_accuracy = 0


    history = {

        "train_loss": [],
        "train_accuracy": [],

        "valid_loss": [],
        "valid_accuracy": []

    }



    for epoch in range(cfg.EPOCHS):

        print("\n")
        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS}"
        )


        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )


        valid_loss, valid_acc = validate(
            model,
            valid_loader,
            criterion,
            device
        )


        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy : {train_acc:.4f}"
        )

        print(
            f"Valid Loss : {valid_loss:.4f}"
        )

        print(
            f"Valid Accuracy : {valid_acc:.4f}"
        )


        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_acc)



        if valid_acc > best_accuracy:

            best_accuracy = valid_acc

            torch.save(
                model.state_dict(),
                cfg.CHECKPOINT_FILE
            )

            print(
                "Best Model Saved"
            )



    torch.save(
        model.state_dict(),
        cfg.MODEL_FILE
    )


    with open(
        cfg.LOG_DIR / "training_history.json",
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


    print("\nTraining Completed")

    print(
        f"Best Validation Accuracy : {best_accuracy:.4f}"
    )



if __name__ == "__main__":

    main()