document.addEventListener('DOMContentLoaded', () => {
    // FAQ 관련 코드를 모두 제거했습니다.
    // FAQ 목록을 불러오고, 폼을 다루는 모든 함수와 변수를 삭제했습니다.
    
    const idHeader = document.getElementById('id-header');
    let sortDirection = 'desc'; // 기본값은 최신순 (내림차순)

    // 로그 기록을 불러와서 페이지에 표시하는 함수
    async function loadLogs() {
        try {
            const response = await fetch('/api/admin/logs');
            if (response.status === 401) {
                window.location.href = '/';
                return;
            }
            let logs = await response.json();

            // ID 기준으로 로그 데이터 정렬
            logs = sortLogsById(logs, sortDirection);

            const logListTableBody = document.getElementById('logListTableBody');
            
            logListTableBody.innerHTML = '';
            
            if (logs.length === 0) {
                logListTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center;">로그 기록이 없습니다.</td></tr>';
            } else {
                logs.forEach(log => {
                    const row = document.createElement('tr');
                    const timestamp = new Date(log.timestamp).toLocaleString('ko-KR', {
                        year: 'numeric', month: '2-digit', day: '2-digit',
                        hour: '2-digit', minute: '2-digit', second: '2-digit',
                        hour12: false
                    });

                    row.innerHTML = `
                        <td>${log.id}</td>
                        <td>${log.question}</td>
                        <td>${log.answer}</td>
                        <td>${log.similarity}</td>
                        <td>${timestamp}</td>
                    `;
                    logListTableBody.appendChild(row);
                });
            }
        } catch (error) {
            console.error('Error loading logs:', error);
        }
    }

    // ID 기준 로그 정렬 함수
    function sortLogsById(logs, direction) {
        return logs.sort((a, b) => {
            if (direction === 'asc') {
                return a.id - b.id;
            } else {
                return b.id - a.id;
            }
        });
    }

    // ID 헤더 클릭 시 정렬 방향 변경 및 로그 재로드
    if (idHeader) {
        idHeader.addEventListener('click', () => {
            sortDirection = sortDirection === 'desc' ? 'asc' : 'desc';

            const icon = idHeader.querySelector('i');
            if (icon) {
                if (sortDirection === 'asc') {
                    icon.classList.remove('fa-sort-down');
                    icon.classList.add('fa-sort-up');
                } else {
                    icon.classList.remove('fa-sort-up');
                    icon.classList.add('fa-sort-down');
                }
            }
            loadLogs();
        });
    }

    // 초기 페이지 로드 시 로그 기록을 바로 불러옵니다.
    loadLogs();
});