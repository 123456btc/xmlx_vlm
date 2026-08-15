import { state, elements } from '../core/state.js';
import { scrollToBottom, escapeHtml, formatBytes, showTypingIndicator, hideTypingIndicator } from '../core/utils.js';
import { updatePortfolio } from './market.js';

// Load Session List
export async function loadSessions(autoCreate = true) {
    try {
        const resp = await fetch('/api/sessions');
        const sessions = await resp.json();
        
        elements.sessionsList.innerHTML = '';
        if (sessions.length === 0) {
            elements.sessionsList.innerHTML = '<div class="no-positions">No sessions available.</div>';
            if (autoCreate) {
                try {
                    const createResp = await fetch('/api/sessions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ title: "New Session" })
                    });
                    const session = await createResp.json();
                    await loadSessions(false);
                    selectSession(session.session_id);
                } catch (err) {
                    console.error('Failed to auto-create first session:', err);
                }
            }
            return;
        }
        
        sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = `session-item ${session.session_id === state.activeSessionId ? 'active' : ''}`;
            item.dataset.id = session.session_id;
            
            const timeStr = new Date(session.last_active_at * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            item.innerHTML = `
                <div class="session-item-info">
                    <div class="session-item-title">${escapeHtml(session.title)}</div>
                    <div class="session-item-meta">
                        <span>${session.mode.toUpperCase()}</span>
                        <span>${timeStr}</span>
                    </div>
                </div>
                <button class="btn-delete-session" title="Delete Session">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            `;
            
            // Delete button listener
            item.querySelector('.btn-delete-session').addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm('Delete this session and all its messages?')) {
                    deleteSession(session.session_id);
                }
            });
            
            // Row click listener
            item.addEventListener('click', () => {
                selectSession(session.session_id);
            });
            
            elements.sessionsList.appendChild(item);
        });

        // Auto-select first session if none is active
        if (sessions.length > 0 && !state.activeSessionId) {
            selectSession(sessions[0].session_id);
        }
    } catch (err) {
        console.error('Load sessions failed:', err);
    }
}

export async function deleteSession(id) {
    try {
        await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
        if (state.activeSessionId === id) {
            state.activeSessionId = null;
            closeChat();
        }
        await loadSessions();
    } catch (err) {
        console.error('Delete session failed:', err);
    }
}

export function closeChat() {
    elements.chatInput.disabled = true;
    elements.btnSendMessage.disabled = true;
    if (elements.btnAttachFile) elements.btnAttachFile.disabled = true;
    elements.activeSessionTitle.textContent = 'Select a Session';
    elements.chatTimeline.innerHTML = `
        <div class="empty-state">
            <i class="fa-solid fa-arrow-left pulse-icon"></i>
            <h3>Select or create a chat session on the left to begin trading.</h3>
        </div>
    `;
    if (state.chatWs) {
        state.chatWs.onclose = null;
        state.chatWs.onerror = null;
        state.chatWs.close();
        state.chatWs = null;
    }
}

// Select Session
export async function selectSession(sessionId) {
    state.activeSessionId = sessionId;
    
    // Highlight session item in sidebar
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === sessionId);
    });

    // Update headers
    const sessionItem = document.querySelector(`.session-item[data-id="${sessionId}"]`);
    const title = sessionItem ? sessionItem.querySelector('.session-item-title').textContent : 'Quant Session';
    elements.activeSessionTitle.textContent = title;
    
    // Enable input elements
    elements.chatInput.disabled = false;
    elements.btnSendMessage.disabled = false;
    if (elements.btnAttachFile) elements.btnAttachFile.disabled = false;
    elements.chatInput.focus();

    // Clear staged attachments
    state.stagedAttachmentsList = [];
    renderStagedAttachments();

    // Load Chat History
    elements.chatTimeline.innerHTML = '<div class="loading-placeholder"><i class="fa-solid fa-circle-notch fa-spin"></i> Loading chat...</div>';
    
    try {
        const resp = await fetch(`/api/sessions/${sessionId}/messages`);
        const messages = await resp.json();
        
        elements.chatTimeline.innerHTML = '';
        if (messages.length === 0 || (messages.length === 1 && messages[0].role === 'system')) {
            renderEmptyTimeline();
        } else {
            messages.forEach(msg => {
                if (msg.role !== 'system') {
                    let cleanContent = msg.content;
                    let msgAttachments = [];
                    
                    if (Array.isArray(msg.content)) {
                        const textParts = msg.content.filter(p => p.type === 'text').map(p => p.text);
                        cleanContent = textParts.join('\n');
                        
                        msg.content.forEach(p => {
                            if (p.type === 'image_url') {
                                msgAttachments.push({
                                    type: 'image',
                                    url: p.image_url.url,
                                    name: p.image_url.url.split('/').pop()
                                });
                            } else if (p.type === 'video_url') {
                                msgAttachments.push({
                                    type: 'video',
                                    url: p.video_url.url,
                                    name: p.video_url.url.split('/').pop()
                                });
                            }
                        });
                    }
                    renderMessageBubble(msg.role, cleanContent, msg.timestamp, msgAttachments);
                }
            });
            scrollToBottom();
        }
    } catch (err) {
        console.error('Failed to load messages:', err);
    }

    // Connect WebSocket
    connectChatWs(sessionId);
}

export function renderEmptyTimeline() {
    elements.chatTimeline.innerHTML = `
        <div class="empty-state">
            <i class="fa-solid fa-chart-line pulse-icon"></i>
            <h3>Quant Agent connected. Ask me any market data or trading questions!</h3>
            <p>Select a quick command to begin:</p>
            <ul>
                <li onclick="sendPresetPrompt('BTC 现在什么行情？')">"BTC 现在什么行情？"</li>
                <li onclick="sendPresetPrompt('画一张 BTC 1小时 K线图')">"画一张 BTC 1小时 K线图"</li>
                <li onclick="sendPresetPrompt('查看当前账户持仓')">"查看当前账户持仓"</li>
                <li onclick="sendPresetPrompt('模拟买入 0.01 BTC')">"模拟买入 0.01 BTC"</li>
            </ul>
        </div>
    `;
}

// WS Connection
export function connectChatWs(sessionId) {
    if (state.chatWs) {
        state.chatWs.onclose = null;
        state.chatWs.onerror = null;
        state.chatWs.close();
    }
    
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/api/chat/${sessionId}/ws`;
    
    state.chatWs = new WebSocket(wsUrl);
    
    state.chatWs.onopen = () => {
        console.log('Chat WebSocket connected');
        // Flush pending messages
        if (state.pendingChatMessages && state.pendingChatMessages.length > 0) {
            console.log(`Flushing ${state.pendingChatMessages.length} pending chat messages.`);
            state.pendingChatMessages.forEach(payload => {
                state.chatWs.send(JSON.stringify(payload));
            });
            state.pendingChatMessages = [];
        }
    };
    
    state.chatWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWsEvent(data);
    };
    
    state.chatWs.onclose = (e) => {
        console.log('Chat WebSocket closed:', e);
        hideTypingIndicator();
        // Auto reconnect if the session is still active
        if (state.activeSessionId === sessionId) {
            console.log('Attempting chat WS reconnect in 3s...');
            setTimeout(() => {
                if (state.activeSessionId === sessionId) {
                    connectChatWs(sessionId);
                }
            }, 3000);
        }
    };
    
    state.chatWs.onerror = (err) => {
        console.error('WS Error:', err);
        hideTypingIndicator();
    };
}

export function stripToolCalls(text) {
    if (!text) return "";
    return text
        .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, '')
        .replace(/<tool_call>[\s\S]*$/gi, '')
        .replace(/<function=[^>]+>[\s\S]*?<\/function>/gi, '')
        .replace(/<function=[^>]+>[\s\S]*$/gi, '')
        .replace(/<parameter=[^>]+>[\s\S]*?<\/parameter>/gi, '')
        .replace(/<parameter=[^>]+>[\s\S]*$/gi, '')
        .replace(/call:\s*\w+\s*\{[^}]*\}/gi, '')
        .trim();
}

// Handle WS stream events
let currentAssistantBubble = null;
let currentAssistantContent = "";
let currentAssistantThinking = "";
let currentToolBlock = null;

export function handleWsEvent(data) {
    switch(data.type) {
        case 'thinking':
            hideTypingIndicator();
            if (!currentAssistantBubble) {
                currentAssistantBubble = renderMessageBubble('assistant', '', Date.now() / 1000);
                currentAssistantContent = "";
                currentAssistantThinking = "";
            }
            if (typeof currentAssistantThinking === 'undefined') {
                currentAssistantThinking = "";
            }
            currentAssistantThinking += data.content;
            
            const cleanThinking = stripToolCalls(currentAssistantThinking);
            if (cleanThinking) {
                // Render the thinking content inside a details container in the bubble
                let thinkingContainer = currentAssistantBubble.querySelector('.message-thinking-container');
                if (!thinkingContainer) {
                    thinkingContainer = document.createElement('div');
                    thinkingContainer.className = 'message-thinking-container';
                    currentAssistantBubble.insertBefore(thinkingContainer, currentAssistantBubble.querySelector('.message-content'));
                }
                
                thinkingContainer.innerHTML = `
                    <details class="thinking-details" open>
                        <summary class="thinking-summary">
                            <i class="fa-solid fa-brain thinking-brain-icon"></i>
                            <span>AI is thinking...</span>
                        </summary>
                        <div class="thinking-content">${escapeHtml(cleanThinking)}</div>
                    </details>
                `;
            }
            scrollToBottom();
            break;

        case 'text':
            hideTypingIndicator();
            if (!currentAssistantBubble) {
                // First text delta, create bubble
                currentAssistantBubble = renderMessageBubble('assistant', '', Date.now() / 1000);
                currentAssistantContent = "";
            }
            
            // Collapse the thinking block once content generation starts
            const detailsEl = currentAssistantBubble.querySelector('.thinking-details');
            if (detailsEl && detailsEl.hasAttribute('open')) {
                detailsEl.removeAttribute('open');
                const summarySpan = detailsEl.querySelector('.thinking-summary span');
                if (summarySpan) {
                    summarySpan.textContent = 'Thinking Process';
                }
            }
            
            currentAssistantContent += data.content;
            const cleanContent = stripToolCalls(currentAssistantContent);
            if (cleanContent) {
                currentAssistantBubble.querySelector('.message-content').innerHTML = marked.parse(cleanContent);
            }
            scrollToBottom();
            break;
            
        case 'tool_start':
            showTypingIndicator(`Calling tool: ${data.name}...`);
            
            // Finalize current assistant bubble if present before starting tool card
            if (currentAssistantBubble) {
                const finalContent = stripToolCalls(currentAssistantContent);
                const finalThinking = stripToolCalls(currentAssistantThinking);
                if (finalContent) {
                    currentAssistantBubble.querySelector('.message-content').innerHTML = marked.parse(finalContent);
                } else if (finalThinking) {
                    currentAssistantBubble.querySelector('.message-content').innerHTML = marked.parse(finalThinking);
                    const tContainer = currentAssistantBubble.querySelector('.message-thinking-container');
                    if (tContainer) tContainer.remove();
                } else {
                    currentAssistantBubble.remove();
                }
                currentAssistantBubble = null;
                currentAssistantContent = "";
                currentAssistantThinking = "";
            }

            // Create tool call box in timeline
            currentToolBlock = document.createElement('div');
            currentToolBlock.className = 'tool-block';
            currentToolBlock.innerHTML = `
                <div class="tool-header-row" onclick="this.parentElement.classList.toggle('open')">
                    <i class="fa-solid fa-gears tool-icon"></i>
                    <span>Running <span class="tool-name-val">${escapeHtml(data.name)}</span></span>
                    <i class="fa-solid fa-chevron-down tool-arrow"></i>
                </div>
                <div class="tool-details">Arguments: ${escapeHtml(JSON.stringify(data.arguments, null, 2))}</div>
            `;
            elements.chatTimeline.appendChild(currentToolBlock);
            scrollToBottom();
            break;
            
        case 'tool_end':
            hideTypingIndicator();
            if (currentToolBlock) {
                // Update header title to show completed
                currentToolBlock.querySelector('.tool-header-row span').innerHTML = `Ran <span class="tool-name-val">${escapeHtml(data.name)}</span> (Finished)`;
                currentToolBlock.querySelector('.tool-icon').className = 'fa-solid fa-circle-check tool-icon';
                
                // Append output details
                const detailDiv = currentToolBlock.querySelector('.tool-details');
                detailDiv.textContent += `\n\nOutput:\n${data.output}`;
                currentToolBlock = null;
            }
            // Reset assistant bubble reference so the next response begins in a new bubble below the tool card
            currentAssistantBubble = null;
            currentAssistantContent = "";
            currentAssistantThinking = "";
            showTypingIndicator("AI is synthesizing analysis...");
            scrollToBottom();
            
            // Reload portfolio stats after trading tool executions
            if (data.name === 'trading') {
                setTimeout(updatePortfolio, 1000);
            }
            break;
            
        case 'image_render':
            hideTypingIndicator();
            // Append inline chart image card to timeline
            const chartCard = document.createElement('div');
            chartCard.className = 'chart-card';
            chartCard.innerHTML = `
                <img src="${data.url}" class="chart-img" alt="K-Line Chart">
                <div class="chart-caption"><i class="fa-solid fa-magnifying-glass-plus"></i> View Full Chart</div>
            `;
            
            // Zoom lightbox click listener
            chartCard.addEventListener('click', () => {
                elements.lightboxModal.style.display = 'block';
                elements.lightboxImg.src = data.url;
                elements.lightboxCaption.textContent = 'Rendered K-Line Chart';
            });
            
            elements.chatTimeline.appendChild(chartCard);
            scrollToBottom();
            break;
            
        case 'approval_required':
            hideTypingIndicator();
            const approvalCard = document.createElement('div');
            approvalCard.className = 'message approval-card';
            
            const timeStr = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            const formattedArgs = JSON.stringify(data.arguments, null, 2);
            
            approvalCard.innerHTML = `
                <div class="message-meta">
                    <span>SYSTEM SECURITY GATE</span>
                    <span>${timeStr}</span>
                </div>
                <div class="approval-card-title">
                    <i class="fa-solid fa-shield-halved"></i>
                    <span>Trading Action Approval Required</span>
                </div>
                <div class="approval-args">${escapeHtml(formattedArgs)}</div>
                <div class="approval-actions">
                    <button class="btn btn-sm btn-reject"><i class="fa-solid fa-xmark"></i> Reject</button>
                    <button class="btn btn-sm btn-approve"><i class="fa-solid fa-check"></i> Approve</button>
                </div>
            `;
            
            const rejectBtn = approvalCard.querySelector('.btn-reject');
            const approveBtn = approvalCard.querySelector('.btn-approve');
            
            const handleDecision = (approved) => {
                rejectBtn.disabled = true;
                approveBtn.disabled = true;
                
                approvalCard.querySelector('.approval-actions').innerHTML = `
                    <span class="status-val font-mono" style="color: ${approved ? 'var(--color-success)' : 'var(--color-danger)'}">
                        <i class="fa-solid ${approved ? 'fa-check' : 'fa-xmark'}"></i>
                        Action ${approved ? 'APPROVED' : 'REJECTED'}
                    </span>
                `;
                
                sendOrQueueMessage({
                    type: 'approval_response',
                    tool_call_id: data.tool_call_id,
                    approved: approved
                });
                
                showTypingIndicator(approved ? 'Executing approved transaction...' : 'Transaction cancelled...');
            };
            
            rejectBtn.addEventListener('click', () => handleDecision(false));
            approveBtn.addEventListener('click', () => handleDecision(true));
            
            elements.chatTimeline.appendChild(approvalCard);
            scrollToBottom();
            break;
            
        case 'done':
            hideTypingIndicator();
            if (currentAssistantBubble) {
                const detailsEl = currentAssistantBubble.querySelector('.thinking-details');
                if (detailsEl) {
                    if (!currentAssistantContent && currentAssistantThinking) {
                        // If model only generated thinking tokens, display it as main content
                        currentAssistantBubble.querySelector('.message-content').innerHTML = marked.parse(currentAssistantThinking);
                        detailsEl.remove();
                    } else if (detailsEl.hasAttribute('open')) {
                        detailsEl.removeAttribute('open');
                        const summarySpan = detailsEl.querySelector('.thinking-summary span');
                        if (summarySpan) {
                            summarySpan.textContent = 'Thinking Process';
                        }
                    }
                }
            }
            currentAssistantBubble = null;
            currentAssistantContent = "";
            currentAssistantThinking = "";
            break;
            
        case 'error':
            hideTypingIndicator();
            renderMessageBubble('assistant', `⚠️ **Error**: ${data.message}`, Date.now() / 1000);
            currentAssistantBubble = null;
            currentAssistantContent = "";
            currentAssistantThinking = "";
            scrollToBottom();
            break;
            
        case 'title_update':
            // Update the session's title in the active view if it is currently selected
            if (data.session_id === state.activeSessionId) {
                elements.activeSessionTitle.textContent = data.title;
            }
            // Update the title in the sidebar list
            const sidebarItem = document.querySelector(`.session-item[data-id="${data.session_id}"]`);
            if (sidebarItem) {
                const titleEl = sidebarItem.querySelector('.session-item-title');
                if (titleEl) {
                    titleEl.textContent = data.title;
                }
            }
            break;
    }
}

// Send Message
export function sendMessage() {
    const text = elements.chatInput.value.trim();
    if (!text && state.stagedAttachmentsList.length === 0) return;
    if (!state.activeSessionId) return;
    
    // Clear input field
    elements.chatInput.value = '';
    
    // Reset state for new assistant stream
    currentAssistantBubble = null;
    currentAssistantContent = "";
    currentAssistantThinking = "";
    
    // Render user bubble
    renderMessageBubble('user', text, Date.now() / 1000, state.stagedAttachmentsList);
    scrollToBottom();
    
    // Show typing loader
    showTypingIndicator('AI is generating response...');
    
    const messagePayload = {
        prompt: text,
        attachments: state.stagedAttachmentsList
    };
    
    // Clear staged attachments
    state.stagedAttachmentsList = [];
    renderStagedAttachments();
    
    sendOrQueueMessage(messagePayload);
}

export function sendOrQueueMessage(payload) {
    if (state.chatWs && state.chatWs.readyState === WebSocket.OPEN) {
        state.chatWs.send(JSON.stringify(payload));
    } else {
        if (!state.pendingChatMessages) {
            state.pendingChatMessages = [];
        }
        state.pendingChatMessages.push(payload);
        
        // Connect if closed or closing
        if (!state.chatWs || state.chatWs.readyState === WebSocket.CLOSED || state.chatWs.readyState === WebSocket.CLOSING) {
            console.log('Chat WebSocket closed/not-ready, buffering message and connecting...');
            connectChatWs(state.activeSessionId);
        } else if (state.chatWs.readyState === WebSocket.CONNECTING) {
            console.log('Chat WebSocket is connecting, buffering message...');
        }
    }
}

// Preset prompt click trigger
export function sendPresetPrompt(promptText) {
    elements.chatInput.value = promptText;
    sendMessage();
}

// Bind to window so HTML inline onclick can access it
window.sendPresetPrompt = sendPresetPrompt;

// Render Message Bubble Utility
export function renderMessageBubble(role, content, timestamp, attachments = []) {
    // Check if empty state exists and delete it
    const emptyState = elements.chatTimeline.querySelector('.empty-state');
    if (emptyState) {
        elements.chatTimeline.innerHTML = '';
    }

    const bubble = document.createElement('div');
    bubble.className = `message ${role}`;
    
    const timeStr = new Date(timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    const roleLabel = role === 'user' ? 'User' : 'Trader Agent';
    
    bubble.innerHTML = `
        <div class="message-meta">
            <span>${roleLabel}</span>
            <span>${timeStr}</span>
        </div>
        <div class="message-content"></div>
    `;
    
    let parsedContent = content;
    let localAttachments = [...attachments];
    
    // If content is list format (complex user uploads), extract text
    if (Array.isArray(content)) {
        const textParts = content.filter(p => p.type === 'text').map(p => p.text);
        parsedContent = textParts.join('\n');
        
        content.forEach(part => {
            if (part.type === 'image_url') {
                if (!localAttachments.some(a => a.url === part.image_url.url)) {
                    localAttachments.push({
                        type: 'image',
                        url: part.image_url.url,
                        name: part.image_url.url.split('/').pop()
                    });
                }
            } else if (part.type === 'video_url') {
                if (!localAttachments.some(a => a.url === part.video_url.url)) {
                    localAttachments.push({
                        type: 'video',
                        url: part.video_url.url,
                        name: part.video_url.url.split('/').pop()
                    });
                }
            }
        });
    }
    
    bubble.querySelector('.message-content').innerHTML = marked.parse(parsedContent || '');
    
    elements.chatTimeline.appendChild(bubble);
    
    // Render attachments
    if (localAttachments && localAttachments.length > 0) {
        const container = document.createElement('div');
        container.className = 'message-attachments';
        
        localAttachments.forEach(att => {
            if (att.type === 'image') {
                const imgCard = document.createElement('div');
                imgCard.className = 'chart-card';
                imgCard.innerHTML = `
                    <img src="${att.url}" class="chart-img">
                    <div class="chart-caption"><i class="fa-solid fa-image"></i> ${escapeHtml(att.name)}</div>
                `;
                imgCard.addEventListener('click', () => {
                    elements.lightboxModal.style.display = 'block';
                    elements.lightboxImg.src = att.url;
                    elements.lightboxCaption.textContent = att.name;
                });
                container.appendChild(imgCard);
            } else if (att.type === 'video') {
                const videoEl = document.createElement('video');
                videoEl.className = 'message-video-player';
                videoEl.controls = true;
                videoEl.src = att.url;
                container.appendChild(videoEl);
            } else {
                const fileCard = document.createElement('a');
                fileCard.className = 'message-attachment-card';
                fileCard.href = att.url;
                fileCard.target = '_blank';
                
                let iconClass = 'fa-file-lines';
                if (att.mime_type && att.mime_type.includes('csv')) {
                    iconClass = 'fa-file-csv';
                } else if (att.type === 'text') {
                    iconClass = 'fa-file-code';
                }
                
                fileCard.innerHTML = `
                    <i class="fa-solid ${iconClass} file-icon"></i>
                    <div class="file-info">
                        <div class="file-name">${escapeHtml(att.name)}</div>
                        <div class="file-meta">${formatBytes(att.size || 0)}</div>
                    </div>
                `;
                container.appendChild(fileCard);
            }
        });
        
        bubble.appendChild(container);
    }

    return bubble;
}

export async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    showTypingIndicator(`Uploading ${file.name}...`);
    try {
        const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const result = await resp.json();
        hideTypingIndicator();
        if (result.status === 'success') {
            state.stagedAttachmentsList.push(result);
            renderStagedAttachments();
        } else {
            alert('Upload failed: ' + (result.message || 'unknown error'));
        }
    } catch (err) {
        console.error('File upload failed:', err);
        hideTypingIndicator();
        alert('Upload failed');
    }
}

export function renderStagedAttachments() {
    if (!elements.stagedAttachments) return;
    if (state.stagedAttachmentsList.length === 0) {
        elements.stagedAttachments.classList.add('hidden');
        elements.stagedAttachments.innerHTML = '';
        return;
    }
    elements.stagedAttachments.classList.remove('hidden');
    elements.stagedAttachments.innerHTML = '';
    state.stagedAttachmentsList.forEach((att, index) => {
        const item = document.createElement('div');
        item.className = 'staged-attachment-item';
        let previewHtml = '';
        if (att.type === 'image') {
            previewHtml = `<img src="${att.url}" class="item-preview">`;
        } else if (att.type === 'video') {
            previewHtml = `<div class="item-preview"><i class="fa-solid fa-file-video"></i></div>`;
        } else if (att.type === 'text') {
            previewHtml = `<div class="item-preview"><i class="fa-solid fa-file-lines"></i></div>`;
        } else {
            previewHtml = `<div class="item-preview"><i class="fa-solid fa-file"></i></div>`;
        }
        item.innerHTML = `
            ${previewHtml}
            <div class="item-details">
                <div class="item-name" title="${escapeHtml(att.name)}">${escapeHtml(att.name)}</div>
                <div class="item-size">${formatBytes(att.size)}</div>
            </div>
            <button class="btn-remove-attachment" data-index="${index}" title="Remove">
                <i class="fa-solid fa-xmark"></i>
            </button>
        `;
        item.querySelector('.btn-remove-attachment').addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(e.currentTarget.dataset.index);
            state.stagedAttachmentsList.splice(idx, 1);
            renderStagedAttachments();
        });
        elements.stagedAttachments.appendChild(item);
    });
}
