# Review Sense AI

Deep Learning based Customer Sentiment Analysis using PyTorch LSTM.

## Overview

Review Sense AI is an end-to-end NLP application that classifies customer reviews into Positive and Negative sentiment using a custom-built LSTM network trained on Amazon Reviews.

The project includes:

- Data preprocessing
- Vocabulary creation
- Custom PyTorch Dataset
- LSTM Model
- Model training
- Evaluation
- Interactive Streamlit dashboard
- Real-time inference

---

## Dataset

Amazon Reviews Dataset

Training Samples:
200,000

Classes:
- Positive
- Negative

---

## Architecture

Embedding Layer

↓

2-layer LSTM

↓

Dropout

↓

Fully Connected Layer

↓

Softmax

---

## Model Configuration

Embedding Dimension: 200

Hidden Dimension: 256

Layers: 2

Dropout: 0.5

Vocabulary Size: 50,000

Maximum Sequence Length: 200

Optimizer: Adam

Loss Function: CrossEntropyLoss

---

## Performance

Validation Accuracy:
92.39%

Precision:
91.92%

Recall:
93.09%

F1 Score:
92.50%

---

## Project Structure

config/

data/

models/

logs/

figures/

src/

streamlit_app.py

requirements.txt

README.md

---

## Run Training

python -m src.train

---

## Evaluate

python -m src.evaluate

---

## Inference

python -m src.inference

---

## Launch Dashboard

streamlit run streamlit_app.py

---

## Technologies

Python

PyTorch

Pandas

NumPy

Scikit-learn

Streamlit

Matplotlib

---

## Author

Muhammad Usman