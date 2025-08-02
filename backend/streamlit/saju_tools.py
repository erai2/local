import os
import sqlite3
import pandas as pd

# Import generate_comment from the Flask example
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "flask_app"))
from app import generate_comment

# Export the cases database to an Excel report

DEF_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "flask_app", "saju_cases.db")


def export_cases_to_excel(db_path=DEF_DB_PATH, output_file="사주_분석_자동리포트.xlsx"):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM saju_cases", conn)
        df.to_excel(output_file, index=False)
    finally:
        conn.close()

# Simple chatbot-style helper that searches cases and returns a comment

def chatbot_answer(query, db_path=DEF_DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            "SELECT * FROM saju_cases WHERE title LIKE ? OR structures LIKE ?",
            (f"%{query}%", f"%{query}%"),
        )
        row = c.fetchone()
        if row:
            row_dict = {"title": row[1], "structures": row[2], "nodes": row[3], "event": row[4]}
            comment = generate_comment(row_dict)
            return f"{row[1]}\n{comment}"
        else:
            return "관련된 사례가 없습니다."
    finally:
        conn.close()

if __name__ == "__main__":
    # basic demo
    print(chatbot_answer("단명"))
    export_cases_to_excel()
