import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import time
import os
from datetime import datetime
import mysql.connector
import numpy as np
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# 1. SBERT 모델 로딩
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# GitHub의 raw 파일 URL
FAQ_URL = "https://raw.githubusercontent.com/kyj01051/chat_1.0/main/faq.json"

def load_faq_data():
    with open(FAQ_URL, 'r', encoding='utf-8') as f:
        return json.load(f)

faq_data = load_faq_data()

# 3. 질문/답변/연관질문 리스트 구성
def build_question_lists():
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
    
    return all_questions, question_to_answer, question_to_related

all_questions, question_to_answer, question_to_related = build_question_lists()

# 4. 질문 임베딩 생성
question_embeddings = model.encode(all_questions, convert_to_tensor=True)

# FAQ 데이터 저장 함수
def save_faq_data(data):
    with open(FAQ_URL, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    # 모델과 데이터 다시 로드
    global faq_data, all_questions, question_to_answer, question_to_related, question_embeddings
    faq_data = data
    all_questions, question_to_answer, question_to_related = build_question_lists()
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
            "related_questions": related_questions,
            "score": best_score  # 유사도 점수 추가
        }
    else:
        return {
            "matched_question": None,
            "answer": "죄송합니다. 질문을 잘 이해하지 못했습니다. 다른 표현으로도 물어봐 주시길 바랍니다.",
            "related_questions": related_questions if related_questions else [],
            "score": best_score  # 유사도 점수 추가
        }

# 6. MySQL 데이터베이스 연결 및 로그 저장 함수 (UUID 사용)
def save_to_db(user_message, bot_response, similarity_score=None):
    cursor = None
    connection = None
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )

        cursor = connection.cursor()

        # 유사도 점수 필드 추가
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS chat_logs (
                id VARCHAR(36) PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT,
                bot_response TEXT,
                similarity_score FLOAT NULL
            )
        ''')

        unique_id = str(uuid.uuid4())
        cursor.execute(''' 
            INSERT INTO chat_logs (id, user_message, bot_response, similarity_score) 
            VALUES (%s, %s, %s, %s)
        ''', (unique_id, user_message, bot_response, similarity_score))

        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# 7. 데이터베이스에서 로그 불러오는 함수 추가
def load_logs(limit=None, search_query=None, date_range=None):
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        
        cursor = connection.cursor(dictionary=True)
        
        # 검색 조건 추가
        where_clauses = []
        params = []
        
        if search_query:
            where_clauses.append("(user_message LIKE %s OR bot_response LIKE %s)")
            search_pattern = f"%{search_query}%"
            params.extend([search_pattern, search_pattern])
        
        if date_range:
            start_date, end_date = date_range
            where_clauses.append("timestamp BETWEEN %s AND %s")
            params.extend([start_date, end_date])
        
        where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        query = f'''
            SELECT id, timestamp, user_message, bot_response, similarity_score
            FROM chat_logs
            {where_clause}
            ORDER BY timestamp DESC
            {limit_clause}
        '''
        
        cursor.execute(query, params)
        
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if result:
            df = pd.DataFrame(result)
            # timestamp를 datetime 객체로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"로그 불러오기 오류: {e}")
        return pd.DataFrame()

# 로그 삭제 함수
def delete_log(log_id):
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        
        cursor = connection.cursor()
        query = "DELETE FROM chat_logs WHERE id = %s"
        cursor.execute(query, (log_id,))
        connection.commit()
        
        result = cursor.rowcount > 0
        
        cursor.close()
        connection.close()
        
        return result
    except Exception as e:
        print(f"로그 삭제 오류: {e}")
        return False

# CSV 다운로드 링크 생성 함수
def get_csv_download_link(df, filename="chat_logs.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">다운로드</a>'
    return href

# 엑셀 다운로드 링크 생성 함수
def get_excel_download_link(df, filename="chat_logs.xlsx"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='로그')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">다운로드</a>'
    return href

# 워드클라우드 생성 함수
def generate_wordcloud(text_data):
    if not text_data:
        return None
    
    # 한글 폰트 설정 (시스템에 맞게 수정 필요)
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우 기준 맑은 고딕 폰트 경로
    
    wordcloud = WordCloud(
        font_path=font_path,
        width=800, 
        height=400, 
        background_color='white',
        max_words=100
    ).generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    return plt

# 연관 질문 버튼 콜백 함수
def ask_related_question(question):
    st.session_state.new_question = question
    st.session_state.process_new_question = True

# 8. Streamlit UI
st.title("대구대 문헌정보학과 챗봇")

# 9. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "process_new_question" not in st.session_state:
    st.session_state.process_new_question = False
if "new_question" not in st.session_state:
    st.session_state.new_question = ""
if "last_db_entry" not in st.session_state:
    st.session_state.last_db_entry = {}
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False
if 'show_login_form' not in st.session_state:
    st.session_state.show_login_form = False
if 'admin_tab' not in st.session_state:
    st.session_state.admin_tab = "대시보드"  # 기본값 설정
if 'edit_faq_index' not in st.session_state:
    st.session_state.edit_faq_index = -1
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# 10. 관리자 로그인 사이드바 구현
# 사이드바에서 로그인 버튼만 표시
with st.sidebar:

    # 대화 초기화 버튼 - 로그인 여부와 상관없이 항상 표시
    st.markdown("## 💬 대화 관리")
    if st.button("대화 초기화", key="clear_chat"):
       st.session_state.messages = []
       st.rerun()

    # 로그인 버튼을 클릭하면 관리자 로그인 폼을 표시
    if st.button("로그인"):
        st.session_state.show_login_form = True


# 로그인 폼 표시 여부 체크
if st.session_state.show_login_form:
    with st.sidebar:
        st.markdown("### 로그인 폼")
        username = st.text_input("아이디", key="admin_username")
        password = st.text_input("비밀번호", type="password", key="admin_password")

        if st.button("로그인", key="admin_login"):
            # secrets.toml 파일에서 관리자 정보 불러오기
            if username == st.secrets["admin"]["username"] and password == st.secrets["admin"]["password"]:
                st.session_state.admin_logged_in = True
                st.success("로그인 성공!")
                st.session_state.show_login_form = False  # 로그인 후 폼 숨기기
                st.rerun()  # 페이지 새로 고침
            else:
                st.error("아이디 또는 비밀번호가 틀렸습니다.")


# 로그인된 상태에서 관리자 기능 추가
if st.session_state.admin_logged_in:
    st.success("관리자 로그인됨 ✅")
    
    # 사이드바에 관리자 탭 메뉴와 로그아웃 버튼 추가
    with st.sidebar:
        admin_tabs = ["대시보드", "FAQ 관리", "로그 관리", "챗봇 분석"]
        st.session_state.admin_tab = st.radio("메뉴 선택", admin_tabs)

        # 로그아웃 버튼
        if st.button("로그아웃", key="admin_logout", type="primary"):
            st.session_state.admin_logged_in = False
            st.rerun()  # 페이지 새로 고침


# 메인 영역 - 관리자 로그인 여부에 따른 UI 분기
if st.session_state.admin_logged_in:
    # 관리자 페이지 표시
    if st.session_state.admin_tab == "대시보드":
        st.markdown("## 👨‍💼 관리자 대시보드")
        
        # 최근 로그 데이터 로드
        logs_df = load_logs(limit=100)
        
        if not logs_df.empty:
            # 기본 통계
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 대화 수", len(logs_df))
            
            with col2:
                fail_count = logs_df['bot_response'].str.contains("죄송합니다").sum()
                fail_ratio = (fail_count / len(logs_df)) * 100
                st.metric("답변 실패율", f"{fail_ratio:.1f}%")
            
            with col3:
                avg_score = logs_df['similarity_score'].mean() if 'similarity_score' in logs_df.columns else 0
                st.metric("평균 유사도 점수", f"{avg_score:.2f}")
            
            # 시간별 대화량 차트
            st.subheader("시간별 대화량")
            logs_df['hour'] = logs_df['timestamp'].dt.hour
            hourly_counts = logs_df['hour'].value_counts().sort_index().reset_index()
            hourly_counts.columns = ['시간', '대화 수']
            
            fig = px.bar(hourly_counts, x='시간', y='대화 수', title='시간별 대화량')
            st.plotly_chart(fig)
            
            # 최근 질문 목록
            st.subheader("최근 질문")
            recent_logs = logs_df[['timestamp', 'user_message', 'bot_response', 'similarity_score']].head(10)
            recent_logs['timestamp'] = recent_logs['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            recent_logs.columns = ['시간', '질문', '답변', '유사도']
            st.dataframe(recent_logs)
            
            # 자주 묻는 질문 (빈도순)
            st.subheader("자주 묻는 질문 (빈도순)")
            top_questions = logs_df['user_message'].value_counts().head(10)
            top_q_df = pd.DataFrame({
                '질문': top_questions.index,
                '빈도': top_questions.values
            })
            st.dataframe(top_q_df)
            
            # 워드클라우드 생성
            st.subheader("질문 워드클라우드")
            all_questions_text = ' '.join(logs_df['user_message'].tolist())
            wordcloud_fig = generate_wordcloud(all_questions_text)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
        else:
            st.warning("아직 저장된 로그가 없습니다.")
    
    elif st.session_state.admin_tab == "FAQ 관리":
        st.markdown("## 📝 FAQ 관리")
        
        # FAQ 목록 표시
        st.subheader("FAQ 목록")
        
        for i, faq in enumerate(faq_data):
            with st.expander(f"FAQ #{i+1}: {faq['questions'][0]}"):
                st.markdown("### 질문들")
                for q in faq['questions']:
                    st.markdown(f"- {q}")
                
                st.markdown("### 답변")
                st.text_area("", value=faq['answer'], height=150, key=f"view_answer_{i}", disabled=True)
                
                st.markdown("### 연관 질문")
                for rq in faq.get('related_questions', []):
                    st.markdown(f"- {rq}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("수정", key=f"edit_{i}"):
                        st.session_state.edit_faq_index = i
                
                with col2:
                    if st.button("삭제", key=f"delete_{i}"):
                        faq_data.pop(i)
                        save_faq_data(faq_data)
                        st.success("FAQ가 삭제되었습니다.")
                        st.rerun()
        
        # FAQ 추가 또는 수정
        st.subheader("FAQ 추가/수정")
        
        is_editing = st.session_state.edit_faq_index >= 0
        
        if is_editing:
            current_faq = faq_data[st.session_state.edit_faq_index]
            questions_text = "\n".join(current_faq['questions'])
            answer_text = current_faq['answer']
            related_text = "\n".join(current_faq.get('related_questions', []))
        else:
            questions_text = ""
            answer_text = ""
            related_text = ""
        
        questions = st.text_area(
            "질문들 (한 줄에 하나씩 입력)", 
            value=questions_text,
            height=100, 
            help="여러 형태의 질문을 한 줄에 하나씩 입력하세요."
        )
        
        answer = st.text_area(
            "답변", 
            value=answer_text,
            height=200
        )
        
        related = st.text_area(
            "연관 질문 (한 줄에 하나씩 입력, 선택사항)", 
            value=related_text,
            height=100
        )
        
        if is_editing:
            if st.button("FAQ 수정"):
                # 질문과 연관 질문을 줄바꿈으로 분리
                questions_list = [q.strip() for q in questions.split('\n') if q.strip()]
                related_list = [r.strip() for r in related.split('\n') if r.strip()]
                
                if not questions_list or not answer:
                    st.error("질문과 답변은 필수입니다.")
                else:
                    # FAQ 수정
                    faq_data[st.session_state.edit_faq_index] = {
                        'questions': questions_list,
                        'answer': answer,
                        'related_questions': related_list
                    }
                    save_faq_data(faq_data)
                    st.session_state.edit_faq_index = -1
                    st.success("FAQ가 수정되었습니다.")
                    st.rerun()
            
            if st.button("취소"):
                st.session_state.edit_faq_index = -1
                st.rerun()
        else:
            if st.button("FAQ 추가"):
                # 질문과 연관 질문을 줄바꿈으로 분리
                questions_list = [q.strip() for q in questions.split('\n') if q.strip()]
                related_list = [r.strip() for r in related.split('\n') if r.strip()]
                
                if not questions_list or not answer:
                    st.error("질문과 답변은 필수입니다.")
                else:
                    # 새 FAQ 추가
                    new_faq = {
                        'questions': questions_list,
                        'answer': answer,
                        'related_questions': related_list
                    }
                    faq_data.append(new_faq)
                    save_faq_data(faq_data)
                    st.success("새 FAQ가 추가되었습니다.")
                    st.rerun()
    
    elif st.session_state.admin_tab == "로그 관리":
        st.markdown("## 📊 로그 관리")
        
        # 로그 검색 필터
        st.subheader("로그 검색")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_query = st.text_input("검색어", st.session_state.search_query)
        
        with col2:
            date_range = st.date_input(
                "기간 선택",
                value=None,
                help="로그 검색 기간을 선택하세요. 비워두면 전체 기간을 검색합니다."
            )
        
        if st.button("검색"):
            st.session_state.search_query = search_query
        
        # 로그 목록 표시
        processed_date_range = None
        if date_range:
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                processed_date_range = (
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                )
        
        logs_df = load_logs(search_query=search_query, date_range=processed_date_range)
        
        if not logs_df.empty:
            st.markdown(f"### 검색 결과: {len(logs_df)}개의 로그")
            
            # 다운로드 옵션
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_csv_download_link(logs_df), unsafe_allow_html=True)
            with col2:
                st.markdown(get_excel_download_link(logs_df), unsafe_allow_html=True)
            
            # 로그 테이블 표시
            log_table = logs_df.copy()
            log_table['timestamp'] = log_table['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            log_table.columns = ['ID', '시간', '질문', '답변', '유사도']
            
            st.dataframe(log_table)
            
            # 로그 삭제 기능
            st.subheader("로그 삭제")
            log_id_to_delete = st.text_input("삭제할 로그 ID")
            
            if st.button("선택한 로그 삭제"):
                if log_id_to_delete:
                    if delete_log(log_id_to_delete):
                        st.success(f"로그 ID: {log_id_to_delete}가 삭제되었습니다.")
                        st.rerun()
                    else:
                        st.error("로그 삭제에 실패했습니다.")
                else:
                    st.warning("삭제할 로그 ID를 입력하세요.")
        else:
            st.warning("검색 결과가 없습니다.")
    
    elif st.session_state.admin_tab == "챗봇 분석":
        st.markdown("## 📈 챗봇 성능 분석")
        
        # 전체 로그 데이터 로드
        logs_df = load_logs()
        if not logs_df.empty:
            # 일별 대화량 추이
            st.subheader("일별 대화량 추이")
            logs_df['date'] = logs_df['timestamp'].dt.date
            daily_counts = logs_df['date'].value_counts().sort_index().reset_index()
            daily_counts.columns = ['날짜', '대화 수']
            fig = px.line(daily_counts, x='날짜', y='대화 수', title='일별 대화량 추이')
            st.plotly_chart(fig)
            
            # 유사도 점수 분포
            if 'similarity_score' in logs_df.columns:
                st.subheader("유사도 점수 분포")
                fig = px.histogram(
                    logs_df, 
                    x='similarity_score', 
                    nbins=20, 
                    title="유사도 점수 분포"
                )
                st.plotly_chart(fig)
            
            # 답변 성공/실패 비율
            st.subheader("답변 성공/실패 비율")
            logs_df['is_failure'] = logs_df['bot_response'].str.contains("죄송합니다")
            success_fail_counts = logs_df['is_failure'].value_counts()
            labels = ['성공', '실패']
            values = [
                success_fail_counts.get(False, 0),
                success_fail_counts.get(True, 0)
            ]
            
            fig = px.pie(
                names=labels,
                values=values,
                title="답변 성공/실패 비율"
            )
            st.plotly_chart(fig)
            
            # 시간대별 사용량
            st.subheader("시간대별 사용량")
            logs_df['hour'] = logs_df['timestamp'].dt.hour
            hour_counts = logs_df['hour'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['시간대', '대화 수']
            
            fig = px.bar(
                hour_counts,
                x='시간대',
                y='대화 수',
                title="시간대별 사용량"
            )
            st.plotly_chart(fig)
            
            # 주요 실패 질문 분석
            st.subheader("주요 실패 질문 분석")
            if 'is_failure' in logs_df.columns:
                failed_queries = logs_df[logs_df['is_failure'] == True]['user_message'].value_counts().head(10)
                
                if not failed_queries.empty:
                    failed_df = pd.DataFrame({
                        '실패 질문': failed_queries.index,
                        '횟수': failed_queries.values
                    })
                    st.dataframe(failed_df)
                else:
                    st.write("실패한 질문이 없습니다.")
                    
            # 데이터 다운로드 기능
            st.subheader("데이터 다운로드")
            csv = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "로그 데이터 CSV 다운로드",
                csv,
                "chatbot_logs.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.error("로그 데이터가 없습니다.")
else:
    # 일반 사용자 챗봇 UI
    if not st.session_state.admin_logged_in or (st.session_state.admin_logged_in and st.session_state.admin_tab == None):
        # 챗봇 소개 및 정보 표시
        st.markdown(""" 
        대구대학교 문헌정보학과에서 운영하는 챗봇입니다.<br>
        입학 및 학과에 관해 궁금한 점이 있다면 챗봇에게 질문해주시길 바랍니다.<br>
        챗봇이 답변하지 못하는 질문들은 학과사무실 또는 학과홈페이지 참고 부탁드립니다.<br><br>
        📞 **대구대학교 문헌정보학과 학과 사무실 전화번호**: 053-850-6350<br>
        🌐 **학과 홈페이지**: [https://lis.daegu.ac.kr/main]
        """, unsafe_allow_html=True)

        # 메시지 히스토리 표시
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if msg["role"] == "assistant" and "related_questions" in msg:
                    for q_idx, related_q in enumerate(msg["related_questions"]):
                        unique_key = f"related_q_{idx}_{q_idx}_{uuid.uuid5(uuid.NAMESPACE_DNS, related_q)}"
                        st.button(f"💬 {related_q}", key=unique_key, on_click=ask_related_question, args=(related_q,))

        # 연관 질문 버튼이 클릭된 경우
        if st.session_state.process_new_question:
            user_input = st.session_state.new_question
            st.session_state.new_question = ""
            st.session_state.process_new_question = False
            
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            result = find_best_answer_and_related(user_input)
            response = result["answer"]
            related_questions = result["related_questions"]
            response_with_br = response.replace("\n", "<br>")
            
            # 유사도 점수 추출 (있을 경우)
            similarity_score = result.get("score", None)

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
            
            # DB에 로그 저장 (유사도 점수 포함)
            save_to_db(user_input, full_response, similarity_score)
            st.rerun()

        # 일반 사용자 입력 처리
        if user_input := st.chat_input("궁금한 점을 질문해보세요!"):
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            result = find_best_answer_and_related(user_input)
            response = result["answer"]
            related_questions = result["related_questions"]
            response_with_br = response.replace("\n", "<br>")
            
            # 유사도 점수 추출 (있을 경우)
            similarity_score = result.get("score", None)

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
            
            # DB에 로그 저장 (유사도 점수 포함)
            save_to_db(user_input, full_response, similarity_score)
            
            # 페이지가 재로드되지 않을 경우를 대비해 여기서도 저장
            save_to_db(user_input, response, result.get("score"))