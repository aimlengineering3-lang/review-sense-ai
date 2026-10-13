"""
=========================================================
Review Sense AI
Model Evaluation
=========================================================
"""

import torch
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config.config import cfg
from src.dataset import ReviewDataset
from src.model import ReviewLSTM


# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------

def evaluate(model, loader, device):

    model.eval()

    predictions = []
    actual = []


    with torch.no_grad():

        for sequences, lengths, labels in loader:

            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)


            outputs = model(
                sequences,
                lengths
            )


            preds = torch.argmax(
                outputs,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )

            actual.extend(
                labels.cpu().numpy()
            )


    return actual, predictions



# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("="*60)
    print("Review Sense AI Evaluation")
    print("="*60)


    device = cfg.DEVICE


    dataset = ReviewDataset()


    train_size = int(
        len(dataset) * cfg.TRAIN_SIZE
    )

    valid_size = len(dataset) - train_size


    _, valid_dataset = random_split(
        dataset,
        [
            train_size,
            valid_size
        ],
        generator=torch.Generator().manual_seed(
            cfg.RANDOM_STATE
        )
    )


    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    model = ReviewLSTM().to(device)


    model.load_state_dict(
        torch.load(
            cfg.CHECKPOINT_FILE,
            map_location=device
        )
    )


    print("\nModel Loaded Successfully")


    y_true, y_pred = evaluate(
        model,
        valid_loader,
        device
    )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred
    )


    recall = recall_score(
        y_true,
        y_pred
    )


    f1 = f1_score(
        y_true,
        y_pred
    )


    print("\nEvaluation Results")
    print("-"*40)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Negative",
                "Positive"
            ]
        )
    )


    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )



if __name__ == "__main__":

    main()