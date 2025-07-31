from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io, json
import pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Rule(BaseModel):
    id: int
    condition: str
    action: str

# 임시 규칙 데이터
rules = [
    {"id": 1, "condition": "A일 때", "action": "B를 한다"},
    {"id": 2, "condition": "X면", "action": "Y실행"},
]

@app.get("/rules")
def list_rules():
    return rules

@app.post("/rules")
def add_rule(rule: Rule):
    rules.append(rule.dict())
    return {"ok": True}

@app.put("/rules/{rule_id}")
def edit_rule(rule_id: int, rule: Rule):
    for i, r in enumerate(rules):
        if r["id"] == rule_id:
            rules[i] = rule.dict()
            return {"ok": True}
    return {"error": "not found"}

@app.delete("/rules/{rule_id}")
def delete_rule(rule_id: int):
    global rules
    rules = [r for r in rules if r["id"] != rule_id]
    return {"ok": True}

# AI 규칙 자동추출
@app.post("/extract_rules")
async def extract_rules(file: UploadFile = File(...)):
    content = await file.read()
    # AI 추출 예시 - 실제론 파일 내용을 분석해야 함
    extracted = [
        {"id": 100, "condition": "업로드 규칙 예시", "action": "추출된 액션"}
    ]
    return {"rules": extracted}

# 내보내기 (JSON/Excel)
@app.get("/export")
def export_rules(fmt: str = "json"):
    if fmt == "json":
        buf = io.BytesIO(json.dumps(rules, ensure_ascii=False).encode("utf-8"))
        return FileResponse(buf, media_type="application/json", filename="rules.json")
    elif fmt == "excel":
        df = pd.DataFrame(rules)
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)
        return FileResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="rules.xlsx")
    return {"error": "지원하지 않는 형식"}
