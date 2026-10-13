from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(frozen=True)
class Config:
    """
    Review Sense AI
    Central Configuration File
    """

    # ==========================================================
    # PROJECT ROOT
    # ==========================================================

    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # ==========================================================
    # DATA DIRECTORIES
    # ==========================================================

    @property
    def DATA_DIR(self):
        return self.BASE_DIR / "data"

    @property
    def RAW_DATA_DIR(self):
        return self.DATA_DIR / "raw" / "amazon"

    @property
    def PROCESSED_DATA_DIR(self):
        return self.DATA_DIR / "processed"

    @property
    def TRAINING_DATA_DIR(self):
        return self.DATA_DIR / "training"

    # ==========================================================
    # DATA FILES
    # ==========================================================

    @property
    def TRAIN_CSV(self):
        return self.RAW_DATA_DIR / "train.csv"

    @property
    def TEST_CSV(self):
        return self.RAW_DATA_DIR / "test.csv"

    @property
    def TRAIN_PROCESSED(self):
        return self.PROCESSED_DATA_DIR / "train_processed.csv"

    @property
    def TRAIN_CLEANED(self):
        return self.PROCESSED_DATA_DIR / "train_cleaned.csv"

    @property
    def TEST_CLEANED(self):
        return self.PROCESSED_DATA_DIR / "test_cleaned.csv"

    # ==========================================================
    # MODEL DIRECTORIES
    # ==========================================================

    @property
    def MODEL_DIR(self):
        return self.BASE_DIR / "models"

    @property
    def TOKENIZER_DIR(self):
        return self.MODEL_DIR / "tokenizer"

    @property
    def CHECKPOINT_DIR(self):
        return self.MODEL_DIR / "checkpoints"

    @property
    def FINAL_MODEL_DIR(self):
        return self.MODEL_DIR / "final"

    # ==========================================================
    # MODEL FILES
    # ==========================================================

    @property
    def VOCAB_FILE(self):
        return self.TOKENIZER_DIR / "vocab.pkl"

    @property
    def MODEL_FILE(self):
        return self.FINAL_MODEL_DIR / "review_sense_ai.pth"

    @property
    def CHECKPOINT_FILE(self):
        return self.CHECKPOINT_DIR / "best_model.pth"

    # ==========================================================
    # OUTPUT DIRECTORIES
    # ==========================================================

    @property
    def OUTPUT_DIR(self):
        return self.BASE_DIR / "outputs"

    @property
    def FIGURE_DIR(self):
        return self.OUTPUT_DIR / "figures"

    @property
    def REPORT_DIR(self):
        return self.OUTPUT_DIR / "reports"

    @property
    def LOG_DIR(self):
        return self.OUTPUT_DIR / "logs"

    @property
    def PREDICTION_DIR(self):
        return self.OUTPUT_DIR / "predictions"

    # ==========================================================
    # VOCABULARY
    # ==========================================================

    VOCAB_SIZE = 50000
    MIN_FREQUENCY = 2

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    MAX_SEQUENCE_LENGTH = 200

    # ==========================================================
    # MODEL
    # ==========================================================

    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    OUTPUT_DIM = 2

    # ==========================================================
    # TRAINING
    # ==========================================================

    BATCH_SIZE = 256

    EPOCHS = 10

    LEARNING_RATE = 5e-4

    RANDOM_STATE = 42

    TRAIN_SIZE = 0.80
    VALID_SIZE = 0.10
    TEST_SIZE = 0.10

    # ==========================================================
    # DEVICE
    # ==========================================================

    @property
    def DEVICE(self):
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ==========================================================
    # CREATE DIRECTORIES
    # ==========================================================

    def create_directories(self):

        folders = [

            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.TRAINING_DATA_DIR,

            self.MODEL_DIR,
            self.TOKENIZER_DIR,
            self.CHECKPOINT_DIR,
            self.FINAL_MODEL_DIR,

            self.OUTPUT_DIR,
            self.FIGURE_DIR,
            self.REPORT_DIR,
            self.LOG_DIR,
            self.PREDICTION_DIR,

        ]

        for folder in folders:
            folder.mkdir(
                parents=True,
                exist_ok=True
            )


cfg = Config()

cfg.create_directories()


if __name__ == "__main__":

    print("=" * 60)
    print("Review Sense AI Configuration")
    print("=" * 60)

    print(f"Device : {cfg.DEVICE}")
    print(f"Train CSV : {cfg.TRAIN_CSV}")
    print(f"Test CSV  : {cfg.TEST_CSV}")
    print(f"Train Data: {cfg.TRAIN_CLEANED}")
    print(f"Vocab     : {cfg.VOCAB_FILE}")

    print("=" * 60)