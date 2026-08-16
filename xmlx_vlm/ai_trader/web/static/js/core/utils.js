import { elements } from './state.js?v=2.0.2';

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

/* ==========================================================================
   Modern Toast Notifications & Quant Modal System
   ========================================================================== */

let toastContainer = null;
function getToastContainer() {
    if (!toastContainer) {
        toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
    }
    return toastContainer;
}

const TOAST_ICONS = {
    success: '<i class="fa-solid fa-circle-check"></i>',
    error: '<i class="fa-solid fa-circle-exclamation"></i>',
    warning: '<i class="fa-solid fa-triangle-exclamation"></i>',
    info: '<i class="fa-solid fa-circle-info"></i>',
};

const TOAST_TITLES = {
    success: '操作成功',
    error: '操作失败',
    warning: '系统提示',
    info: '系统通知',
};

/**
 * Show a sleek dark quant toast notification.
 * @param {string} message - Message text or HTML
 * @param {string} type - 'success' | 'error' | 'warning' | 'info'
 * @param {number} duration - Milliseconds before auto-dismiss (0 for persistent)
 * @param {object} options - { title, icon, onClose }
 */
export function showToast(message, type = 'info', duration = 4000, options = {}) {
    const container = getToastContainer();
    const toast = document.createElement('div');
    const toastType = ['success', 'error', 'warning', 'info'].includes(type) ? type : 'info';
    toast.className = `quant-toast toast-${toastType}`;

    const title = options.title || TOAST_TITLES[toastType];
    const iconHtml = options.icon || TOAST_ICONS[toastType];

    toast.innerHTML = `
        <div class="toast-icon">${iconHtml}</div>
        <div class="toast-body">
            <div class="toast-title">${escapeHtml(title)}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close-btn" title="关闭">&times;</button>
        ${duration > 0 ? '<div class="toast-progress-bar"></div>' : ''}
    `;

    // Limit maximum stacked toasts
    while (container.children.length >= 5) {
        removeToast(container.children[0]);
    }

    container.appendChild(toast);

    const closeBtn = toast.querySelector('.toast-close-btn');
    let timer = null;
    let startTime = Date.now();
    let remainingTime = duration;
    const progressBar = toast.querySelector('.toast-progress-bar');

    function removeToast(el) {
        if (!el || el._isClosing) return;
        el._isClosing = true;
        el.classList.add('toast-hiding');
        if (progressBar) progressBar.style.transition = 'none';
        setTimeout(() => {
            if (el.parentNode) el.parentNode.removeChild(el);
            if (options.onClose) options.onClose();
        }, 250);
    }

    closeBtn.addEventListener('click', () => {
        if (timer) clearTimeout(timer);
        removeToast(toast);
    });

    if (duration > 0 && progressBar) {
        // Animate progress bar
        progressBar.style.transform = 'scaleX(1)';
        requestAnimationFrame(() => {
            progressBar.style.transition = `transform ${duration}ms linear`;
            progressBar.style.transform = 'scaleX(0)';
        });

        const startTimer = () => {
            startTime = Date.now();
            timer = setTimeout(() => removeToast(toast), remainingTime);
        };

        const pauseTimer = () => {
            clearTimeout(timer);
            const elapsed = Date.now() - startTime;
            remainingTime = Math.max(0, remainingTime - elapsed);
            if (progressBar) {
                const computed = window.getComputedStyle(progressBar);
                progressBar.style.transform = computed.transform;
                progressBar.style.transition = 'none';
            }
        };

        toast.addEventListener('mouseenter', pauseTimer);
        toast.addEventListener('mouseleave', () => {
            if (remainingTime > 0) {
                if (progressBar) {
                    progressBar.style.transition = `transform ${remainingTime}ms linear`;
                    progressBar.style.transform = 'scaleX(0)';
                }
                startTimer();
            }
        });

        startTimer();
    }

    return toast;
}

export const notify = {
    success: (msg, duration = 4000, opts = {}) => showToast(msg, 'success', duration, opts),
    error: (msg, duration = 5000, opts = {}) => showToast(msg, 'error', duration, opts),
    warning: (msg, duration = 4500, opts = {}) => showToast(msg, 'warning', duration, opts),
    info: (msg, duration = 3500, opts = {}) => showToast(msg, 'info', duration, opts),
};

/**
 * Show a sleek modal confirmation dialog.
 * @param {string} title - Dialog title
 * @param {string} message - Message explanation
 * @param {object} options - { confirmText, cancelText, danger: bool, icon: html }
 * @returns {Promise<boolean>}
 */
export function showConfirm(title, message, options = {}) {
    return new Promise((resolve) => {
        let backdrop = document.getElementById('quant-modal-container');
        if (!backdrop) {
            backdrop = document.createElement('div');
            backdrop.id = 'quant-modal-container';
            backdrop.className = 'quant-modal-backdrop';
            document.body.appendChild(backdrop);
        }

        const isDanger = options.danger !== false;
        const confirmText = options.confirmText || (isDanger ? '确认执行' : '确定');
        const cancelText = options.cancelText || '取消';
        const iconHtml = options.icon || (isDanger ? '<i class="fa-solid fa-triangle-exclamation"></i>' : '<i class="fa-solid fa-circle-question"></i>');

        backdrop.innerHTML = `
            <div class="quant-modal-card ${isDanger ? 'modal-danger' : ''}">
                <div class="quant-modal-header">
                    <div class="quant-modal-icon">${iconHtml}</div>
                    <div class="quant-modal-title">${escapeHtml(title)}</div>
                </div>
                <div class="quant-modal-body">
                    ${message}
                </div>
                <div class="quant-modal-footer">
                    <button class="quant-modal-btn quant-modal-btn-cancel">${escapeHtml(cancelText)}</button>
                    <button class="quant-modal-btn ${isDanger ? 'quant-modal-btn-danger' : 'quant-modal-btn-confirm'}">${escapeHtml(confirmText)}</button>
                </div>
            </div>
        `;

        backdrop.classList.add('active');

        const card = backdrop.querySelector('.quant-modal-card');
        const btnCancel = backdrop.querySelector('.quant-modal-btn-cancel');
        const btnConfirm = backdrop.querySelector(isDanger ? '.quant-modal-btn-danger' : '.quant-modal-btn-confirm');

        function cleanup(confirmed) {
            document.removeEventListener('keydown', handleKeyDown);
            backdrop.classList.remove('active');
            setTimeout(() => {
                backdrop.innerHTML = '';
                resolve(confirmed);
            }, 200);
        }

        function handleKeyDown(e) {
            if (e.key === 'Escape') {
                e.preventDefault();
                cleanup(false);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                cleanup(true);
            }
        }

        btnCancel.addEventListener('click', () => cleanup(false));
        btnConfirm.addEventListener('click', () => cleanup(true));
        backdrop.addEventListener('click', (e) => {
            if (e.target === backdrop) cleanup(false);
        });

        document.addEventListener('keydown', handleKeyDown);
        btnConfirm.focus();
    });
}

// Global browser alert monkey-patch to ensure zero ugly native alert popups
if (typeof window !== 'undefined') {
    window.showToast = showToast;
    window.notify = notify;
    window.showConfirm = showConfirm;
    window.alert = (msg) => showToast(String(msg), 'info', 4000);
}

