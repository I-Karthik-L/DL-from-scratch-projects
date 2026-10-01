# RAG from Scratch with Ollama

A minimal **Retrieval-Augmented Generation (RAG)** chatbot built from scratch in Python: no LangChain, no vector database, just Ollama, NumPy, and cosine similarity. The goal was to understand exactly what happens inside a RAG pipeline.

📓 **Notebook:** [`RAG.ipynb`](./RAG.ipynb)
📄 **Dataset:** `cats.txt` (a text file of cat facts, one fact per line; add it next to the notebook)

## How it works

```
cats.txt → embed each line → store in VECTOR_DB (list of (chunk, vector))
                                      │
User question → embed query → cosine similarity → top-3 chunks
                                      │
          chunks + question → Llama 3.2 1B → streamed answer
```

1. **Indexing:** each line of `cats.txt` is embedded and stored in an in-memory list
2. **Retrieval:** the question is embedded and compared against every stored chunk with cosine similarity; the top 3 are returned
3. **Generation:** the retrieved chunks are placed in the system prompt, telling the model to answer *only* from that context, and the response is streamed

## Models (run locally via Ollama)

| Role | Model |
|------|-------|
| Embeddings | `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` (768-dim vectors) |
| Language model | `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF` |

## Setup

```bash
pip install ollama numpy

ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

Make sure Ollama is running, put `cats.txt` in the same folder, then run the notebook.

## Example run

Retrieval for a question about cat speed returned:

| Similarity | Retrieved chunk |
|-----------|-----------------|
| 0.82 | Cats can travel at approximately 31 mph (49 km) over a short distance. |
| 0.71 | Cats can jump up to six times their length. |
| 0.70 | A cat's heart beats at a rate of 110 to 140 beats per minute... |

The retriever found the right fact first, which is the part RAG is responsible for.

## Things I noticed

- **Retrieval worked, generation didn't.** The 1B model ignored the "don't make up information" instruction: it invented a "6 minutes" figure and did a made-up calculation. Small models struggle to stay grounded in context, so a bigger model or a stricter prompt is the fix.
- The output looks spaced out (`C ats  are  known`) because each streamed token is printed with an extra space (`end=" "`). Using `end=""` fixes it.
- The in-memory list works for 10 chunks but doesn't scale; a real vector store (Qdrant, FAISS, Chroma) would be the next step.

## Next steps

- Try a larger LLM and compare answer quality
- Replace the list with a vector database
- Add chunking for longer documents and PDFs
- Move on to **Adaptive RAG** with query routing and LangGraph
