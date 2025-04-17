import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import time
import os
from datetime import datetime

#1. SBERT 모델 로딩
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

#2. FAQ 데이터 로딩
with open('faq.json', 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

#3. 질문/답변 리스트 구성
all_questions = []
question_to_answer = {}
for entry in faq_data:
    answer = entry['answer']
    for q in entry['questions']:
        all_questions.append(q)
        question_to_answer[q] = answer

#4. 질문 임베딩 생성
question_embeddings = model.encode(all_questions, convert_to_tensor=True)

#5. 유사도 기반 답변 검색 함수
def find_best_answer(user_question, threshold=0.6):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_score, best_idx = similarities.max(0)
    best_score = best_score.item()

    if best_score >= threshold:
        matched_question = all_questions[best_idx]
        return question_to_answer[matched_question]
    else:
        return "죄송합니다. 질문을 잘 이해하지 못했습니다. 다른 표현으로도 물어봐 주시길 바랍니다."

#6. 로그 저장 함수
def save_chat_log(user_message, bot_response):
    log_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "user": user_message,
        "bot": bot_response
    }

    log_file = "chat_log.json"

    # 기존 로그 불러오기 또는 새로 생성
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    # 로그 저장
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

#7. Streamlit UI
st.title("대구대 문헌정보학과 챗봇")

st.markdown(""" 
대구대학교 문헌정보학과에서 운영하는 챗봇입니다.<br>
입학 및 학과에 관해 궁금한 점이 있다면 챗봇에게 질문해주시길 바랍니다.<br>
챗봇이 답변하지 못하는 질문들은 학과사무실 또는 학과홈페이지 참고 부탁드립니다.<br><br>
📞 **대구대학교 문헌정보학과 학과 사무실 전화번호**: 053-850-6350<br>
🌐 **학과 홈페이지**: [https://lis.daegu.ac.kr/hakgwa_home/lis/index.php](https://lis.daegu.ac.kr/hakgwa_home/lis/index.php)
""", unsafe_allow_html=True)

#8. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

#9. 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

#10. 사용자 입력 처리
if user_input := st.chat_input("궁금한 점을 질문해보세요!"):
    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 답변 생성
    response = find_best_answer(user_input)
    response_with_br = response.replace("\n", "<br>")

    # 챗봇 응답 출력 (타이핑 효과)
    def response_generator(text):
        for word in text.split():
            yield word + " "
            time.sleep(0.1)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response_generator(response_with_br):
            full_response += chunk
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

    # 대화 히스토리 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 로그 저장
    save_chat_log(user_input, response)

