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

## SuamSaJuAnalyzer

The repository now includes an example script `saju_analyzer.py` demonstrating how to
store and analyze Suam Myeongri data using SQLite. Run the file directly to see a simple analysis routine:

```bash
python saju_analyzer.py
```

Databases created by the script are excluded from Git via `*.db` in `.gitignore`.

## SuamSaJuAnalyzer Highlights

The `SuamSaJuAnalyzer` example demonstrates how Suam Myeongri concepts can be stored and queried from a SQLite database. Key features include:

* **DB auto-creation** – Tables for core theory (`basic_theory`), terminology (`terminology`) and case studies (`case_studies`) are created on first run.
* **Data entry helpers** – Methods such as `add_basic_theory()` and `add_terminology()` simplify populating the database with new knowledge.
* **Concept search** – `search_concept()` lets you find relevant theory via a keyword search across categories, concepts and descriptions.
* **Basic analysis** – `analyze_saju()` shows a minimal approach for parsing a SaJu chart and returning summarized positions of Che and Yong.

These routines are intentionally simple so you can adapt them to your own web interfaces, chatbots or analytics tools.

## SuamSaJuAnalyzer Web Page

A simple Streamlit page `saju_page.py` provides a UI for storing and searching Suam Myeongri theory. Launch it with:

```bash
streamlit run saju_page.py
```

The page lets you add theory or terminology, search existing entries and run a basic Saju analysis.

## Flask Case Database Example

The `saju_web_db` folder contains a minimal Flask app that stores and displays sample Saju cases using SQLite. To run it:

```bash
cd saju_web_db
python app.py
```

On first run it automatically populates a small database. Visit `http://127.0.0.1:5000/` to view the case list and click through to see details rendered with Mermaid diagrams.

## Excel & Chatbot Helpers

The repository provides a small utility script `saju_tools.py` to demonstrate how the case database can be exported to an Excel report and queried from a chatbot-like function.

```bash
python saju_tools.py
```

Running the script outputs the answer to a sample query and generates an Excel file named `사주_분석_자동리포트.xlsx` containing the current `saju_cases` table.
