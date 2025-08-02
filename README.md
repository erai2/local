# Document Analysis & RAG System

This repository contains a Streamlit application for document analysis using RAG (Retrieval-Augmented Generation). The app allows users to upload files, query them via an LLM, summarize text, and perform simple clustering.

## Running Locally

Install the required packages and run Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Managing Secrets

The application expects an OpenAI API key. You can provide it in two ways:

1. **`.streamlit/secrets.toml`** – Create this file and define `OPENAI_API_KEY`:

   ```toml
   [general]
   OPENAI_API_KEY = "sk-..."
   ```

2. **Environment variable** – Set `OPENAI_API_KEY` in your environment.

Do **not** commit your actual `secrets.toml` file or any `.env` files. The provided `.gitignore` already excludes them to keep credentials private.

Uploaded documents are stored in the `uploaded_docs/` directory and are also ignored by Git.