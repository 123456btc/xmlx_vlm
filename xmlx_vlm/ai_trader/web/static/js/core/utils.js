import { elements } from './state.js';

export function scrollToBottom() {
    if (elements.chatTimeline) {
        elements.chatTimeline.scrollTop = elements.chatTimeline.scrollHeight;
    }
}

export function escapeHtml(str) {
    if (typeof str !== 'string') return String(str);
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#039;");
}

export function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function showTypingIndicator(text = 'Thinking...') {
    if (elements.typingStatusText && elements.typingIndicator) {
        elements.typingStatusText.textContent = text;
        elements.typingIndicator.classList.remove('hidden');
    }
}

export function hideTypingIndicator() {
    if (elements.typingIndicator) {
        elements.typingIndicator.classList.add('hidden');
    }
}
