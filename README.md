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

## SuriAnalyzer

The repository now includes an example script `saju_analyzer.py` demonstrating how to
store and analyze SuriAnalyzer data using SQLite. Run the file directly to see
a simple analysis routine:

```bash
python saju_analyzer.py
```

Databases created by the script are excluded from Git via `*.db` in `.gitignore`.

## SuriAnalyzer Highlights

The `SuriAnalyzer` example demonstrates how Suri Myeongri concepts can be stored and queried from a SQLite database. Key features include:

* **DB auto-creation** – Tables for core theory (`basic_theory`), terminology (`terminology`) and case studies (`case_studies`) are created on first run.
* **Data entry helpers** – Methods such as `add_basic_theory()` and `add_terminology()` simplify populating the database with new knowledge.
* **Concept search** – `search_concept()` lets you find relevant theory via a keyword search across categories, concepts and descriptions.
* **Basic analysis** – `analyze_saju()` shows a minimal approach for parsing a SaJu chart and returning summarized positions of Che and Yong.

These routines are intentionally simple so you can adapt them to your own web interfaces, chatbots or analytics tools.


## SuriAnalyzer Web Page

A simple Streamlit page `saju_page.py` provides a UI for storing and searching Suri theory. Launch it with:

```bash
streamlit run saju_page.py
```

The page lets you add theory or terminology, search existing entries and run a basic Saju analysis.
