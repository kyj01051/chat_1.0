import streamlit as st
import pandas as pd
import json
import os
import uuid
from datetime import datetime

# 로그 파일 경로
LOGS_FILE = "logs.json"

# 로그 저장 함수
def save_to_logs(user_message, bot_response, similarity_score=None):
    try:
        if os.path.exists(LOGS_FILE):
            with open(LOGS_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_message": user_message,
            "bot_response": bot_response,
            "similarity_score": similarity_score
        }
        
        logs.append(log_entry)
        
        with open(LOGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"로그 저장 오류: {e}")
        return False

# 로그 불러오기 함수
def load_logs(limit=None, search_query=None, date_range=None):
    try:
        if not os.path.exists(LOGS_FILE):
            return pd.DataFrame()
        
        with open(LOGS_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        if not logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if search_query:
            mask = (df['user_message'].str.contains(search_query, na=False)) | \
                   (df['bot_response'].str.contains(search_query, na=False))
            df = df[mask]
        
        if date_range:
            start_date, end_date = date_range
            mask = (df['timestamp'] >= pd.Timestamp(start_date)) & \
                   (df['timestamp'] <= pd.Timestamp(end_date))
            df = df[mask]
        
        df = df.sort_values('timestamp', ascending=False)
        
        if limit:
            df = df.head(limit)
        
        return df
    except Exception as e:
        print(f"로그 불러오기 오류: {e}")
        return pd.DataFrame()

# 로그 삭제 함수
def delete_log(log_id):
    try:
        if not os.path.exists(LOGS_FILE):
            return False
        
        with open(LOGS_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        logs_filtered = [log for log in logs if log['id'] != log_id]
        
        if len(logs) == len(logs_filtered):
            return False
        
        with open(LOGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs_filtered, f, ensure_ascii=False, indent=4)
        
        return True
    except Exception as e:
        print(f"로그 삭제 오류: {e}")
        return False

# Streamlit 인터페이스
st.title("Chatbot with Logs")

# 사용자 입력 받기
user_message = st.text_input("Your message:")

if user_message:
    # 예시 챗봇 응답 (여기서는 간단하게 사용자가 입력한 메시지를 반전시켜서 응답)
    bot_response = f"Bot: {user_message[::-1]}"
    
    # 챗봇 응답 출력
    st.write(bot_response)
    
    # 로그 저장
    save_to_logs(user_message, bot_response)
    st.success("Your question and bot response have been logged!")

# 로그 관리 페이지
st.title("로그 관리 페이지")

# 로그를 불러와 DataFrame 형태로 출력
df_logs = load_logs(limit=100)
if not df_logs.empty:
    st.dataframe(df_logs)

    # 각 로그 항목에 대한 링크 만들기
    for index, row in df_logs.iterrows():
        log_link = f"[로그 보기 - {row['id']}](/logs/{row['id']})"
        st.markdown(log_link)
        
    # 로그 삭제 버튼
    log_id_to_delete = st.text_input("삭제할 로그 ID 입력")
    if st.button("삭제"):
        if delete_log(log_id_to_delete):
            st.success(f"로그 {log_id_to_delete} 삭제 성공")
        else:
            st.error("삭제 실패: 로그 ID가 없거나 다른 오류가 발생했습니다.")
else:
    st.write("로그가 없습니다.")
