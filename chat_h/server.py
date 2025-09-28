from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer, util
import traceback
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
from sqlalchemy_utils import database_exists, create_database
import os
import json

app = FastAPI()

# --- 데이터베이스 설정 ---
DATABASE_FOLDER = "database"
DATABASE_FILE = "chat_data.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 💡 수정된 부분: database 폴더를 경로에 추가
DATABASE_PATH = os.path.join(BASE_DIR, DATABASE_FOLDER, DATABASE_FILE)

engine = create_engine(f"sqlite:///{DATABASE_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# 💡 수정된 부분: declarative_base()의 오타 수정
Base = declarative_base()

# --- 데이터베이스 모델 정의 ---
class UserQuery(Base):
    __tablename__ = "user_queries"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    answer = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    similarity = Column(String)

    def __repr__(self):
        return f"<UserQuery(id={self.id}, question='{self.question}', answer='{self.answer}', timestamp='{self.timestamp}')>"

class FAQ(Base):
    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True, index=True)
    questions = Column(Text)
    answer = Column(Text)
    related = Column(Text, nullable=True)

# --- Pydantic 모델 정의 ---
class FAQItem(BaseModel):
    questions: list[str]
    answer: str
    related: list[str] | None = None

class LoginRequest(BaseModel):
    username: str
    password: str
    
class ChatRequest(BaseModel):
    message: str

# --- 데이터베이스 테이블 생성 함수 ---
def init_db():
    print(f"데이터베이스 초기화 중: {DATABASE_PATH}")
    Base.metadata.create_all(bind=engine)
    print("데이터베이스 초기화 완료.")

@app.on_event("startup")
def on_startup():
    init_db()

# --- 의존성 주입을 위한 함수 ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 미들웨어 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static 파일 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# --- 로그인 상태 확인을 위한 의존성 주입 ---
def get_current_user(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id == "logged_in":
        return True
    return False

# --- 라우터 정의 (HTML 페이지 렌더링) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def get_admin_dashboard(request: Request, is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        return RedirectResponse(url="/")
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/faq_list", response_class=HTMLResponse)
async def get_faq_list(request: Request, is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        return RedirectResponse(url="/")
    return templates.TemplateResponse("faq_list.html", {"request": request})

# --- API 엔드포인트 ---
@app.post("/api/login")
async def login(request: LoginRequest):
    if request.username == "kyj0928" and request.password == "sakwa03!!":
        response = JSONResponse(content={"success": True, "message": "로그인 성공"})
        response.set_cookie(key="session_id", value="logged_in", httponly=True, samesite="strict")
        return response
    return {"success": False, "message": "아이디 또는 비밀번호가 올바르지 않습니다."}

@app.post("/api/logout")
async def logout():
    response = JSONResponse(content={"success": True, "message": "로그아웃 성공"})
    response.delete_cookie(key="session_id")
    return response

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        input_question = request.message

        if not input_question:
            return JSONResponse(status_code=400, content={"error": "질문이 비어있습니다"})
            
        faqs_from_db = db.query(FAQ).all()
        
        faq_questions = []
        faq_data_map = {}
        
        for faq in faqs_from_db:
            questions = json.loads(faq.questions)
            for q in questions:
                faq_questions.append(q)
                faq_data_map[q] = {
                    "answer": faq.answer,
                    "related_questions": json.loads(faq.related) if faq.related else []
                }
        
        input_embedding = model.encode(input_question, convert_to_tensor=True)
        faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(input_embedding, faq_embeddings)[0]
        max_score = cos_scores.max().item()
        max_score_index = cos_scores.argmax().item()
        
        threshold = 0.6
        final_response_answer = "죄송합니다. 해당 질문에 대한 적절한 답변을 찾지 못했습니다."
        final_response_related_questions = []
        
        if max_score >= threshold:
            best_match_question = faq_questions[max_score_index]
            best_match_data = faq_data_map[best_match_question]
            final_response_answer = best_match_data["answer"]
            final_response_related_questions = best_match_data["related_questions"]
            
        try:
            new_query = UserQuery(
                question=input_question,
                answer=final_response_answer,
                similarity=f"{max_score:.4f}"
            )
            db.add(new_query)
            db.commit()
            db.refresh(new_query)
        except Exception as db_error:
            db.rollback()
            print(f"Error saving to database: {str(db_error)}")

        return JSONResponse(
            content={
                "question": input_question,
                "answer": final_response_answer,
                "related_questions": final_response_related_questions,
                "similarity_score": max_score
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"error": f"요청 처리 중 오류 발생: {str(e)}"}
        )

# --- 관리자 API 엔드포인트 ---
@app.get("/api/admin/logs")
def get_logs(db: Session = Depends(get_db), is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        raise HTTPException(status_code=401, detail="Not authenticated")
    logs = db.query(UserQuery).order_by(UserQuery.timestamp.desc()).all()
    logs_data = []
    for log in logs:
        logs_data.append({
            "id": log.id,
            "question": log.question,
            "answer": log.answer,
            "timestamp": log.timestamp.isoformat(),
            "similarity": log.similarity
        })
    return logs_data

@app.get("/api/admin/faqs")
def get_faqs(db: Session = Depends(get_db), is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        raise HTTPException(status_code=401, detail="Not authenticated")
    faqs = db.query(FAQ).all()
    faq_data = []
    for faq in faqs:
        faq_data.append({
            "id": faq.id,
            "questions": json.loads(faq.questions),
            "answer": faq.answer,
            "related": json.loads(faq.related) if faq.related else []
        })
    return faq_data

@app.get("/api/admin/faqs/{faq_id}")
def get_faq(faq_id: int, db: Session = Depends(get_db), is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        raise HTTPException(status_code=401, detail="Not authenticated")
    faq = db.query(FAQ).filter(FAQ.id == faq_id).first()
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    
    return {
        "id": faq.id,
        "questions": json.loads(faq.questions),
        "answer": faq.answer,
        "related": json.loads(faq.related) if faq.related else []
    }

@app.post("/api/admin/faqs")
def add_faq(faq_item: FAQItem, db: Session = Depends(get_db), is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        raise HTTPException(status_code=401, detail="Not authenticated")
    new_faq = FAQ(
        questions=json.dumps(faq_item.questions, ensure_ascii=False),
        answer=faq_item.answer,
        related=json.dumps(faq_item.related, ensure_ascii=False) if faq_item.related else None
    )
    db.add(new_faq)
    db.commit()
    db.refresh(new_faq)
    return new_faq

@app.put("/api/admin/faqs/{faq_id}")
def update_faq(faq_id: int, faq_item: FAQItem, db: Session = Depends(get_db), is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        raise HTTPException(status_code=401, detail="Not authenticated")
    faq_to_update = db.query(FAQ).filter(FAQ.id == faq_id).first()
    if not faq_to_update:
        raise HTTPException(status_code=404, detail="FAQ not found")

    faq_to_update.questions = json.dumps(faq_item.questions, ensure_ascii=False)
    faq_to_update.answer = faq_item.answer
    faq_to_update.related = json.dumps(faq_item.related, ensure_ascii=False) if faq_item.related else None
    db.commit()
    db.refresh(faq_to_update)
    return faq_to_update

@app.delete("/api/admin/faqs/{faq_id}")
def delete_faq(faq_id: int, db: Session = Depends(get_db), is_logged_in: bool = Depends(get_current_user)):
    if not is_logged_in:
        raise HTTPException(status_code=401, detail="Not authenticated")
    faq_to_delete = db.query(FAQ).filter(FAQ.id == faq_id).first()
    if not faq_to_delete:
        raise HTTPException(status_code=404, detail="FAQ not found")
    
    db.delete(faq_to_delete)
    db.commit()
    return {"message": "FAQ deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)