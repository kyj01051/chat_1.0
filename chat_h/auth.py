from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
async def login(request: LoginRequest):
    if request.username == "kyj0928" and request.password == "sakwa03!!":
        return {"success": True, "message": "로그인 성공"}
    return {"success": False, "message": "아이디 또는 비밀번호가 올바르지 않습니다."}

@app.post("/logout")
async def logout():
    return {"success": True, "message": "로그아웃 성공"}
