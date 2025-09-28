document.addEventListener('DOMContentLoaded', function () {
    const messagesContainer = document.getElementById('messagesContainer');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const loginBtn = document.getElementById('loginBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const adminBadge = document.getElementById('adminBadge');
    const adminMenu = document.getElementById('adminMenu');
    const loginForm = document.getElementById('loginForm');
    const submitLoginBtn = document.getElementById('submitLoginBtn');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const relatedQuestionsContainer = document.getElementById('relatedQuestionsContainer');

    let chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];

    // 💡 수정된 displayMessage 함수
    function displayMessage(message, sender) {
        const messageBlockElement = document.createElement('div');
        messageBlockElement.classList.add('message-block', sender);

        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        messageElement.innerHTML = `<p>${message.replace(/\n/g, '<br>')}</p>`;

        messageBlockElement.appendChild(messageElement);
        messagesContainer.appendChild(messageBlockElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function displayRelatedQuestions(relatedQuestions) {
        if (!relatedQuestions || relatedQuestions.length === 0) return;

        if (relatedQuestionsContainer) {
            relatedQuestionsContainer.innerHTML = '';
            relatedQuestions.forEach(q => {
                const button = document.createElement('button');
                button.classList.add('related-question-btn');
                button.textContent = q;
                button.onclick = () => {
                    messageInput.value = q;
                    sendBtn.click();
                };
                relatedQuestionsContainer.appendChild(button);
            });
        }
    }

    function loadChat() {
        messagesContainer.innerHTML = '';
        const initialMessage = '안녕하세요! 저는 대구대학교 문헌정보학과 챗봇입니다.<p> 무엇이 궁금하신가요?';
        displayMessage(initialMessage, 'bot');

        chatHistory.forEach(chat => {
            displayMessage(chat.message, chat.sender);
            if (chat.sender === 'bot' && chat.related_questions) {
                displayRelatedQuestions(chat.related_questions);
            }
        });
    }

    function checkLoginStatus() {
        const isLoggedIn = document.cookie.split(';').some((item) => item.trim().startsWith('session_id=logged_in'));
        if (isLoggedIn) {
            loginBtn.style.display = 'none';
            logoutBtn.style.display = 'inline-block';
            adminBadge.style.display = 'inline-block';
            adminMenu.style.display = 'flex';
            loginForm.style.display = 'none';
        } else {
            loginBtn.style.display = 'inline-block';
            logoutBtn.style.display = 'none';
            adminBadge.style.display = 'none';
            adminMenu.style.display = 'none';
        }
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        displayMessage(message, 'user');
        chatHistory.push({ message: message, sender: 'user' });
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        messageInput.value = '';

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) throw new Error('네트워크 응답이 올바르지 않습니다.');

            const data = await response.json();
            const botMessage = data.answer;
            const relatedQuestions = data.related_questions;

            displayMessage(botMessage, 'bot');
            chatHistory.push({ message: botMessage, sender: 'bot', related_questions: relatedQuestions });
            displayRelatedQuestions(relatedQuestions);

            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        } catch (error) {
            console.error('Fetch error:', error);
            const errorMessage = '죄송합니다. 오류가 발생했습니다. 다시 시도해 주세요.';
            displayMessage(errorMessage, 'bot');
            chatHistory.push({ message: errorMessage, sender: 'bot' });
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        }
    }

    if (window.location.pathname === '/' || window.location.pathname === '/index.html') {
        // 💡 이 위치에 loadChat() 함수 호출을 추가했습니다.
        loadChat();
        checkLoginStatus();

        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') sendMessage();
        });

        clearChatBtn.addEventListener('click', function () {
            chatHistory = [];
            localStorage.removeItem('chatHistory');
            messagesContainer.innerHTML = '';
            if (relatedQuestionsContainer) {
                relatedQuestionsContainer.innerHTML = '';  // 관련 질문도 비우기
            }
            const initialMessage = '안녕하세요! 저는 대구대학교 문헌정보학과 챗봇입니다.<br>무엇이 궁금하신가요?';
            displayMessage(initialMessage, 'bot');
        });

        loginBtn.addEventListener('click', function () {
            loginForm.style.display = 'flex';
            loginBtn.style.display = 'none';
        });

        submitLoginBtn.addEventListener('click', async function () {
            const username = usernameInput.value;
            const password = passwordInput.value;

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });

                const data = await response.json();
                if (data.success) {
                    window.location.href = '/admin';
                } else {
                    alert(data.message);
                }
            } catch (error) {
                alert('로그인 중 오류가 발생했습니다.');
            }
        });

        logoutBtn.addEventListener('click', async function () {
            try {
                const response = await fetch('/api/logout', { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    alert(data.message);
                    checkLoginStatus();
                }
            } catch (error) {
                alert('로그아웃 중 오류가 발생했습니다.');
            }
        });
    }

    // ======== FAQ 관리 페이지 관련 변수 및 함수 ========
    if (window.location.pathname === '/faq_list') {
        const faqList = document.getElementById('faqList');
        const faqForm = document.getElementById('faqForm');
        const faqFormTitle = document.getElementById('faqFormTitle');
        const faqId = document.getElementById('faqId');
        const questionInput = document.getElementById('question');
        const answerTextarea = document.getElementById('answer');
        const relatedInput = document.getElementById('related');
        const addFaqBtn = document.getElementById('addFaqBtn');
        const cancelFaqBtn = document.getElementById('cancelFaqBtn');

        let faqs = [];

        async function fetchFaqs() {
            const response = await fetch('/api/admin/faqs');
            if (response.ok) {
                faqs = await response.json();
                renderFaqList();
            }
        }

        // 💡 수정된 부분: renderFaqList 함수
        function renderFaqList() {
            const faqTableBody = document.getElementById('faqList');
            if (!faqTableBody) return;

            faqTableBody.innerHTML = '';

            if (faqs.length === 0) {
                const noDataRow = document.createElement('tr');
                noDataRow.innerHTML = `<td colspan="5" style="text-align: center;">등록된 FAQ가 없습니다.</td>`;
                faqTableBody.appendChild(noDataRow);
                return;
            }

            faqs.forEach(faq => {
                const tr = document.createElement('tr');
                const questionsToDisplay = Array.isArray(faq.questions) ? faq.questions.join(', ') : faq.questions;
                const relatedToDisplay = Array.isArray(faq.related) ? faq.related.join(', ') : faq.related;

                tr.innerHTML = `
                    <td>${faq.id}</td>
                    <td>${questionsToDisplay}</td>
                    <td>${faq.answer}</td>
                    <td>${relatedToDisplay || ''}</td>
                    <td class="action-buttons">
                        <button class="edit-btn" data-id="${faq.id}"><i class="fas fa-edit"></i></button>
                        <button class="delete-btn" data-id="${faq.id}"><i class="fas fa-trash-alt"></i></button>
                    </td>
                `;
                faqTableBody.appendChild(tr);
            });
        }

        function showFaqForm() {
            faqForm.style.display = 'block';
            document.querySelector('.faq-list-container').style.display = 'none';
        }

        function hideFaqForm() {
            faqForm.style.display = 'none';
            document.querySelector('.faq-list-container').style.display = 'block';
            resetFaqForm();
        }

        function resetFaqForm() {
            faqId.value = '';
            faqFormTitle.textContent = '새 FAQ 추가';
            questionInput.value = '';
            answerTextarea.value = '';
            relatedInput.value = '';
        }

        async function saveFaq(event) {
            event.preventDefault();

            const id = faqId.value;
            const method = id ? 'PUT' : 'POST';
            const url = id ? `/api/admin/faqs/${id}` : '/api/admin/faqs';

            const questionsValue = questionInput.value.trim();
            let questionsArray = [];
            if (questionsValue) {
                questionsArray = questionsValue.split(',').map(q => q.trim()).filter(q => q);
            }
            if (questionsArray.length === 0) {
                alert('질문 내용을 입력해주세요.');
                return;
            }

            const relatedQuestionsValue = relatedInput.value.trim();
            let relatedArray = [];
            if (relatedQuestionsValue) {
                relatedArray = relatedQuestionsValue.split(',').map(q => q.trim()).filter(q => q);
            }

            const payload = {
                questions: questionsArray,
                answer: answerTextarea.value.trim(),
                related: relatedArray
            };

            try {
                const response = await fetch(url, {
                    method: method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (response.ok) {
                    alert('FAQ가 성공적으로 저장되었습니다.');
                    hideFaqForm();
                    fetchFaqs();
                } else {
                    const error = await response.json();
                    console.error('서버 오류:', error);
                    alert('FAQ 저장에 실패했습니다.\n서버 응답: ' + JSON.stringify(error));
                }
            } catch (err) {
                console.error('요청 실패:', err);
                alert('FAQ 저장 중 오류가 발생했습니다.');
            }
        }

        async function deleteFaq(id) {
            if (confirm('정말로 이 FAQ를 삭제하시겠습니까?')) {
                const response = await fetch(`/api/admin/faqs/${id}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    alert('FAQ가 삭제되었습니다.');
                    fetchFaqs();
                } else {
                    alert('FAQ 삭제에 실패했습니다.');
                }
            }
        }

        faqList.addEventListener('click', (event) => {
            if (event.target.closest('.edit-btn')) {
                const id = event.target.closest('.edit-btn').dataset.id;
                const faq = faqs.find(f => f.id == id);
                if (faq) {
                    faqId.value = faq.id;
                    faqFormTitle.textContent = `FAQ 수정 (ID: ${faq.id})`;

                    questionInput.value = Array.isArray(faq.questions) ? faq.questions.join(', ') : faq.questions;
                    answerTextarea.value = faq.answer;
                    relatedInput.value = Array.isArray(faq.related) ? faq.related.join(', ') : faq.related;

                    showFaqForm();
                }
            }

            if (event.target.closest('.delete-btn')) {
                const id = event.target.closest('.delete-btn').dataset.id;
                deleteFaq(id);
            }
        });

        addFaqBtn.addEventListener('click', () => {
            resetFaqForm();
            showFaqForm();
        });

        faqForm.addEventListener('submit', saveFaq);
        cancelFaqBtn.addEventListener('click', hideFaqForm);

        fetchFaqs();
    }
});