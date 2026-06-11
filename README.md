# Ezra

Ezra is a **RAG (Retrieval-Augmented Generation) QA System** designed for intelligent document-based questioning and answering. It leverages Ollama for local LLM inference and Hugging Face for embedding/generation models.

## Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Python 3.11+**
- **uv** (A fast Python package installer and resolver)
- **Ollama** (If running local models)

## Setup

Clone the repository and install the dependencies using `uv`:

```shell
git clone https://github.com/kiol1812-playground/Ezra.git
cd Ezra

# Sync dependencies and create a virtual environment
uv sync
```

## Environment Configuration (Optional)

If you are using Arch Linux (or any system running the Fish shell), you can configure your environment variables using the following commands:

```shell
# Set variables globally and export them
set -Ux HF_TOKEN "your hugging face access token"
set -Ux OLLAMA_MODEL "which model do you wanted to run"
```

Note on Fish Shell Flags:

- `Ux` (Universal & Export): Sets a universal variable that persists across shell restarts and is exported to child processes.

- `gx` (Global & Export): Sets a global variable valid for the current shell session only, but still exported to child processes.

## Usage

### Launching the GUI

To start the main graphical user interface, run:

```shell
uv run python main.py
```

### Running Evaluation Experiments

To execute the evaluation scripts and benchmarks, use:

```shell
uv run python -m experiments.run_eval
```

## Development

This project uses pre-commit hooks to maintain code quality and formatting standards.

To set up the git hook scripts and run them against your files:

```shell
# Install the git hook scripts
uv run pre-commit install

# Run checks against all files manually
uv run pre-commit run --all-files
```
