# Transformers with Hugging Face Pipelines

Hands-on exploration of the Hugging Face `transformers` library: running pretrained models for common NLP tasks with `pipeline()`, then using one on **real, live data** pulled from arXiv.

📓 **Notebook:** [`Transformers.ipynb`](./Transformers.ipynb)

## What I learned

- Using the Hugging Face `pipeline()` API to run pretrained models in a few lines
- Picking a specific model from the Hub vs. using a task's default model
- Pulling data from an external API (arXiv) and organizing it in a pandas DataFrame
- Chaining it all together: **fetch papers → summarize abstracts**

## Part 1: NLP tasks with pipelines

| Task | Pipeline | Model |
|------|----------|-------|
| Sentiment analysis | `sentiment-analysis` | Default model |
| Text generation | `text-generation` | `distilbert/distilgpt2` |
| Question answering | `question-answering` | Default model |

## Part 2: Summarizing the latest AI/ML papers

1. Query the **arXiv API** for the 10 most recent AI/ML papers (sorted by submission date)
2. Extract `published`, `title`, `abstract`, and `categories` into a **pandas DataFrame**
3. Load the summarization model **`facebook/bart-large-cnn`**
4. Summarize an abstract into a short, readable summary

## Requirements

The notebook was run with `transformers==4.57.1`.

```bash
pip install transformers==4.57.1 torch arxiv pandas
```

> Models are downloaded from the Hugging Face Hub the first time you run each pipeline, so an internet connection is needed.

## Next steps

- Summarize all 10 papers in a loop and store the summaries in the DataFrame
- Add zero-shot classification to tag papers by topic
- Fine-tune a small model on my own dataset
- Move on to building RAG and agent applications with these models
