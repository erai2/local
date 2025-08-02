from flask import Flask, render_template, g
import sqlite3
import os

app = Flask(__name__)
DATABASE = 'saju_cases.db'

structure_map = {
    "자묘파": "자(子)와 묘(卯)가 파(破) 관계로, 수(木)와 목(木)이 충돌하여 구조적 불안정이 발생합니다.",
    "공망": "공망(空亡)으로 해당 오행/십신의 현실 작용력이 약화되어 허상으로 남습니다.",
    "입묘": "입묘(入墓)된 오행/십신은 힘이 내부에 숨겨져 외부로 드러나지 않습니다.",
    "합": "합(合)은 두 글자가 결합하여 새로운 작용력이 발생하는 구조입니다.",
    "미고폐기": "미(未)고(庫)로 인해 기운이 저장, 폐기되는 현상이 나타납니다."
}

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        c = db.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS saju_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                structures TEXT,
                nodes TEXT,
                event TEXT
            )
        ''')
        db.commit()

        c.execute("SELECT COUNT(*) FROM saju_cases")
        if c.fetchone()[0] == 0:
            cases = [
                {
                    "title": "사례1: 단명",
                    "structures": "자묘파,공망",
                    "nodes": "甲癸癸壬,寅卯卯子",
                    "event": "단명"
                },
                {
                    "title": "사례2: 농민",
                    "structures": "입묘,합,미고폐기",
                    "nodes": "癸癸庚庚,亥未辰寅",
                    "event": "농민"
                }
            ]
            for case in cases:
                c.execute(
                    "INSERT INTO saju_cases (title, structures, nodes, event) VALUES (?, ?, ?, ?)",
                    (case["title"], case["structures"], case["nodes"], case["event"])
                )
            db.commit()

def generate_mermaid(row):
    mermaid = ['flowchart TD']
    case_id = "CASE"
    mermaid.append(f'{case_id}["{row["title"]}"]')
    for idx, n in enumerate(row["nodes"].split(',')):
        node_id = f"N{idx+1}"
        mermaid.append(f'{case_id} --> {node_id}["{n}"]')
    for idx, s in enumerate(row["structures"].split(',')):
        s_id = f"S{idx+1}"
        mermaid.append(f'{case_id} --> {s_id}["{s}"]')
    return "\n".join(mermaid)

def generate_comment(row):
    reasons = [structure_map.get(s, f"{s}: 해설 없음") for s in row["structures"].split(',')]
    text = (
        f"■ 주요 구조: {row['structures']}\n"
        + "\n".join(reasons)
        + f"\n■ 실제 사건: {row.get('event','-')}"
    )
    return text

@app.route('/')
def index():
    db = get_db()
    c = db.cursor()
    c.execute("SELECT id, title FROM saju_cases")
    case_list = c.fetchall()
    return render_template('index.html', cases=case_list)

@app.route('/case/<int:case_id>')
def show_case(case_id):
    db = get_db()
    c = db.cursor()
    c.execute("SELECT id, title, structures, nodes, event FROM saju_cases WHERE id=?", (case_id,))
    r = c.fetchone()
    if r:
        row = {"id": r[0], "title": r[1], "structures": r[2], "nodes": r[3], "event": r[4]}
        mermaid_code = generate_mermaid(row)
        comment = generate_comment(row)
        return render_template('case.html', title=row["title"], mermaid_code=mermaid_code, comment=comment)
    return "사례 없음"

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)
