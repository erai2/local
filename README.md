# Suam Myeongri Tools

This repository provides a minimal set of examples for storing and exploring Suam Myeongri theory and case studies with SQLite.

## Project Structure

```
backend/
  streamlit/
    saju_analyzer.py  # SQLite helper and demo
    saju_page.py      # one-page Streamlit interface
  flask_app/
    app.py            # Flask case browser
    templates/
      index.html
      case.html
requirements.txt       # shared dependencies
```

## Streamlit Theory Page

Launch the Streamlit interface to add theory or terminology entries and run a simple Saju analysis:

```bash
pip install -r requirements.txt
streamlit run backend/streamlit/saju_page.py
```

## Flask Case Browser

A small Flask app serves sample Saju cases and automatically initializes its SQLite database on first run:

```bash
pip install -r requirements.txt
cd backend/flask_app
python app.py
```

Then visit <http://127.0.0.1:5000/> to browse cases.

## SuamSaJuAnalyzer Script

`backend/streamlit/saju_analyzer.py` can be executed directly for a quick demonstration of database creation and Saju analysis:

```bash
python backend/streamlit/saju_analyzer.py
```

SQLite databases (`*.db`) are ignored by Git to keep the repository clean.

## Branching

This repository uses a single `main` branch for all work.
