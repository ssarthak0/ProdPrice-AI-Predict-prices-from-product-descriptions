# ProdPrice-AI

Predict product prices from short text descriptions using a full ML/LLM pipeline — from raw Amazon metadata to classical models, deep learning, frontier LLMs, and OpenAI fine-tuning.

## Overview

This project builds and evaluates multiple pricing models on Amazon product data. Each notebook advances the pipeline:

| Notebook | Stage |
|----------|-------|
| `01_dataset_exploration_and_curation.ipynb` | Explore Amazon Reviews 2023, filter and visualize products, publish raw dataset to Hugging Face |
| `02_llm_batch_summarization.ipynb` | Generate concise product summaries via Groq batch API, publish processed dataset |
| `03_classical_ml_baselines.ipynb` | Benchmark random, mean, linear regression, random forest, and XGBoost |
| `04_neural_networks_and_frontier_llms.ipynb` | Train a PyTorch DNN and evaluate zero-shot frontier models |
| `05_openai_fine_tuning.ipynb` | Fine-tune GPT for price prediction and evaluate on held-out test set |
| `redemption_train.ipynb` | Standalone deep neural network training and evaluation using `DeepNeuralNetworkRunner` |
| `results.ipynb` | Compare model performance across the full pipeline |

## Project structure

```
ProdPrice-AI/
├── pricer/             # Core library
│   ├── items.py        # Item model & Hugging Face I/O
│   ├── parser.py       # Parse LLM summaries into structured items
│   ├── loaders.py      # Parallel Amazon dataset loader
│   ├── batch.py        # Groq batch processing
│   ├── evaluator.py    # Evaluation metrics & Plotly charts
│   └── deep_neural_network.py
├── jsonl/              # Fine-tuning & batch JSONL files
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys: Hugging Face, Groq, OpenAI (depending on which notebooks you run)

## Quick start

```bash
# Clone the repository
git clone https://github.com/ssarthak0/ProdPrice-AI-Predict-prices-from-product-descriptions.git
cd ProdPrice-AI-Predict-prices-from-product-descriptions

# Download all dependencies
uv sync

# Add your API keys
cp .env.example .env
```

That's it. Open any `.ipynb` file in Cursor, VS Code, or JupyterLab, select the project kernel (`.venv`), and **run the cells one by one from top to bottom** to reproduce the results.

No separate install step or manual pip commands — `uv sync` handles everything.

## Environment variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face token for dataset upload/download |
| `GROQ_API_KEY` | Groq API key for batch summarization (notebook 02) |
| `OPENAI_API_KEY` | OpenAI API key for fine-tuning and inference (notebook 05) |

Copy `.env.example` to `.env` and fill in the keys you need. Not every notebook requires all three keys.

## Running the notebooks

1. Run `uv sync` once after cloning.
2. Open a notebook — start with `01_...` if you want the full pipeline, or jump to any notebook that matches your goal (e.g. `redemption_train.ipynb` for DNN-only training).
3. Run each cell in order. Earlier cells load data and configure the environment; later cells train models and show evaluation charts.

Set `LITE_MODE = True` in notebooks 02–05 and `redemption_train.ipynb` for a smaller, faster dataset (~20k train). Set `False` for the full ~800k version.

Update `username = "ssarthakoffice"` to your Hugging Face username when publishing datasets.

## Evaluation

The `pricer.evaluator` module reports average error, MSE, R², and interactive Plotly charts. Models are compared side-by-side in `results.ipynb`.

## Acknowledgments

This project was adapted and extended from an existing price-prediction course codebase, with custom modifications to datasets, models, and evaluation workflow.

## License

MIT
