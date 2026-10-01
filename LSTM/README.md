# LSTM Next-Word Prediction

A beginner-friendly notebook where I build a **next-word prediction model with an LSTM** in TensorFlow/Keras, step by step, from raw text to predictions.

📓 **Notebook:** [`LSTM.ipynb`](./LSTM.ipynb)

## What I learned

- Turning raw text into numbers with Keras `Tokenizer`
- Building **n-gram sequences** from a sentence (each prefix becomes a training example)
- Padding sequences to equal length (`pad_sequences`, `padding='pre'`)
- Splitting data into **predictors** (all words but the last) and **target** (the last word)
- One-hot encoding the target with `to_categorical`
- Stacking LSTM layers and using Dropout to reduce overfitting
- Predicting the **top-3 most likely next words** with their probabilities

## Pipeline

```
Text → Tokenizer → n-gram sequences → Pad → (X, y) split → One-hot y
     → Embedding → LSTM → Dropout → LSTM → Dense (softmax) → Next word
```

## Model architecture

| Layer | Details |
|-------|---------|
| Embedding | 100-dim embeddings, `input_length = max_sequence_len - 1` |
| LSTM | 150 units, `return_sequences=True` |
| Dropout | 0.2 |
| LSTM | 100 units |
| Dense | `total_words` units, softmax |

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- **Callback:** `EarlyStopping` on `loss` (patience = 3, `restore_best_weights=True`)

## Example

The notebook trains on a tiny corpus (`"I am living in India"`) and predicts the next word for the seed text `"I am living in"` using `predict_next_word()`, which returns the top-3 candidate words and their probabilities.

## Things I noticed

- This is a **toy dataset**, so the goal is understanding the pipeline, not accuracy. A real model needs a much larger corpus.
- `EarlyStopping` originally monitored `val_loss`, but `model.fit()` has no validation data, so it had nothing to watch. It now monitors `loss`. With a bigger dataset I'll add `validation_split` and switch back to `val_loss`.

## Requirements

```bash
pip install numpy tensorflow
```

## Next steps

- Train on a larger text corpus (e.g. a book or news dataset)
- Add a validation split so early stopping works
- Generate full sentences by predicting words in a loop
- Compare with a GRU and a Transformer-based approach
