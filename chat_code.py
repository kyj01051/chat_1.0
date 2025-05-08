import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import time
import os
from datetime import datetime
import mysql.connector
import numpy as np
import uuid

# 1. SBERT 모델 로딩
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 2. FAQ 데이터 로딩 (절대 경로 사용)
with open(r'C:\Users\user\Desktop\please\chat_1.0\faq.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# 3. 질문/답변/연관질문 리스트 구성
all_questions = []
question_to_answer = {}
question_to_related = {}

for entry in faq_data:
    answer = entry['answer']
    related_questions = entry.get('related_questions', [])
    
    for q in entry['questions']:
        all_questions.append(q)
        question_to_answer[q] = answer
        question_to_related[q] = related_questions

# 4. 질문 임베딩 생성
question_embeddings = model.encode(all_questions, convert_to_tensor=True)

# 5. 유사도 기반 답변 및 연관 질문 검색 함수
def find_best_answer_and_related(user_question, threshold=0.6, num_related=3):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]

    best_score, best_idx = similarities.max(0)
    best_score = best_score.item()
    matched_question = all_questions[best_idx]

    related_questions = []
    if matched_question in question_to_related:
        related_questions.extend(question_to_related[matched_question])

    if len(related_questions) < num_related:
        top_indices = np.argsort(similarities.cpu().numpy())[::-1]
        needed_count = num_related - len(related_questions)
        count = 0
        for idx in top_indices:
            if count >= needed_count:
                break
            current_question = all_questions[idx]
            if idx != best_idx and current_question not in related_questions and similarities[idx].item() > threshold * 0.7:
                related_questions.append(current_question)
                count += 1

    related_questions = list(dict.fromkeys(related_questions))[:num_related]

    if best_score >= threshold:
        return {
            "matched_question": matched_question,
            "answer": question_to_answer[matched_question],
            "related_questions": related_questions
        }
    else:
        return {
            "matched_question": None,
            "answer": "죄송합니다. 질문을 잘 이해하지 못했습니다. 다른 표현으로도 물어봐 주시길 바랍니다.",
            "related_questions": related_questions if related_questions else []
        }

# 6. MySQL 데이터베이스 연결 및 로그 저장 함수 (UUID 사용)
def save_to_db(user_message, bot_response):
    cursor = None
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='0928',
            database='chatbot_db'
        )

        cursor = connection.cursor()

        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS chat_logs (
                id VARCHAR(36) PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT,
                bot_response TEXT
            )
        ''')

        unique_id = str(uuid.uuid4())
        cursor.execute(''' 
            INSERT INTO chat_logs (id, user_message, bot_response) 
            VALUES (%s, %s, %s)
        ''', (unique_id, user_message, bot_response))

        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# 연관 질문 버튼 콜백 함수
def ask_related_question(question):
    st.session_state.new_question = question
    st.session_state.process_new_question = True

# 7. Streamlit UI
st.title("대구대 문헌정보학과 챗봇")

st.markdown(""" 
대구대학교 문헌정보학과에서 운영하는 챗봇입니다.<br>
입학 및 학과에 관해 궁금한 점이 있다면 챗봇에게 질문해주시길 바랍니다.<br>
챗봇이 답변하지 못하는 질문들은 학과사무실 또는 학과홈페이지 참고 부탁드립니다.<br><br>
📞 **대구대학교 문헌정보학과 학과 사무실 전화번호**: 053-850-6350<br>
🌐 **학과 홈페이지**: [https://lis.daegu.ac.kr/main]
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "process_new_question" not in st.session_state:
    st.session_state.process_new_question = False
if "new_question" not in st.session_state:
    st.session_state.new_question = ""

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and "related_questions" in msg:
            for q_idx, related_q in enumerate(msg["related_questions"]):
                unique_key = f"related_q_{idx}_{q_idx}_{uuid.uuid5(uuid.NAMESPACE_DNS, related_q)}"
                st.button(f"💬 {related_q}", key=unique_key, on_click=ask_related_question, args=(related_q,))

if st.session_state.process_new_question:
    user_input = st.session_state.new_question
    st.session_state.process_new_question = False
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    result = find_best_answer_and_related(user_input)
    response = result["answer"]
    related_questions = result["related_questions"]
    response_with_br = response.replace("\n", "<br>")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for word in response_with_br.split():
            full_response += word + " "
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            time.sleep(0.05)
        for q_idx, related_q in enumerate(related_questions):
            unique_key = f"rel_new_{len(st.session_state.messages)}_{q_idx}_{uuid.uuid5(uuid.NAMESPACE_DNS, related_q)}"
            st.button(f"💬 {related_q}", key=unique_key, on_click=ask_related_question, args=(related_q,))

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "related_questions": related_questions
    })
    save_to_db(user_input, full_response)
    st.rerun()

if user_input := st.chat_input("궁금한 점을 질문해보세요!"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    result = find_best_answer_and_related(user_input)
    response = result["answer"]
    related_questions = result["related_questions"]
    response_with_br = response.replace("\n", "<br>")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for word in response_with_br.split():
            full_response += word + " "
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            time.sleep(0.05)
        for q_idx, related_q in enumerate(related_questions):
            unique_key = f"rel_resp_{len(st.session_state.messages)}_{q_idx}_{uuid.uuid5(uuid.NAMESPACE_DNS, related_q)}"
            st.button(f"💬 {related_q}", key=unique_key, on_click=ask_related_question, args=(related_q,))

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response, 
        "related_questions": related_questions
    })
    save_to_db(user_input, full_response)

if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()

