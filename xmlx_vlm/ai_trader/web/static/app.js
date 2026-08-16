import { state, elements } from './js/core/state.js?v=2.0.3';
import { initRouter, updateHashForTab } from './js/core/router.js?v=2.0.3';
import { showToast, notify, showConfirm } from './js/core/utils.js?v=2.0.3';
import { loadSessions, sendMessage, uploadFile, selectSession } from './js/modules/chat.js?v=2.0.3';
import { checkKmsStatus, initKmsVault, unlockKmsVault, lockKmsVault, addKmsKey, deactivateKmsKey, loadKmsAuditLogs } from './js/modules/kms.js?v=2.0.3';
import { refreshExchangeData } from './js/modules/exchange.js?v=2.0.3';
import { loadStrategyDecisions } from './js/modules/strategy.js?v=2.0.3';
import { startMarketLoop, startPortfolioLoop, updateWatchlist, updatePortfolio } from './js/modules/market.js?v=2.0.3';

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventListeners();
    initRouter();
});

export async function fetchConfig() {
    try {
        const resp = await fetch('/api/config');
        const config = await resp.json();
        
        elements.headerModelName.textContent = config.model.split('/').pop();
        elements.headerModelName.title = config.model;
        
        const isLive = config.mode === 'live';
        elements.headerTradingMode.textContent = isLive ? `LIVE (${config.label || 'Hyperliquid'})` : 'PAPER (Sim)';
        elements.headerTradingMode.className = `status-val mode-badge ${isLive ? 'live' : 'paper'}`;
        
        elements.configRisk.textContent = config.risk_profile.toUpperCase();
        
        elements.connectionDot.className = 'connection-dot online';
        elements.connectionStatus.textContent = 'Connected';
    } catch (err) {
        console.error('Config fetch failed:', err);
        elements.connectionDot.className = 'connection-dot offline';
        elements.connectionStatus.textContent = 'Offline';
    }
}

async function initApp() {
    // 1. Fetch system configs
    await fetchConfig();
    
    // 2. Load sessions list
    await loadSessions();
    
    // 3. Check KMS Vault status
    await checkKmsStatus();
    
    // 4. Start real-time loops
    startMarketLoop();
    startPortfolioLoop();
}

function setupEventListeners() {
    // New Session Button
    elements.btnNewSession.addEventListener('click', async () => {
        try {
            const resp = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: "New Session" })
            });
            const session = await resp.json();
            await loadSessions();
            selectSession(session.session_id);
            notify.success('新建会话成功');
        } catch (err) {
            console.error('Failed to create session:', err);
            notify.error('创建会话失败: ' + (err.message || '未知错误'));
        }
    });

    // Clear Chat Button
    elements.btnClearChat.addEventListener('click', async () => {
        if (!state.activeSessionId) return;
        const confirmed = await showConfirm(
            '清空会话记录',
            '确定要清空当前会话的全部对话与分析历史吗？<br><span style="color:var(--color-danger)">此操作无法撤回。</span>',
            { confirmText: '清空会话', cancelText: '取消', danger: true, icon: '<i class="fa-solid fa-trash"></i>' }
        );
        if (!confirmed) return;
        
        try {
            await fetch(`/api/sessions/${state.activeSessionId}`, { method: 'DELETE' });
            // Re-create session with same ID to keep context clean
            await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.activeSessionId, title: elements.activeSessionTitle.textContent })
            });
            selectSession(state.activeSessionId);
            notify.success('已清空当前会话历史');
        } catch (err) {
            console.error('Failed to clear session:', err);
            notify.error('清空会话失败: ' + (err.message || '网络异常'));
        }
    });

    // Send Message Button
    elements.btnSendMessage.addEventListener('click', sendMessage);

    // Keyboard trigger (Enter to send, Shift+Enter for newline)
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Emergency Stop
    elements.btnEmergencyStop.addEventListener('click', async () => {
        const confirmed = await showConfirm(
            '🚨 紧急熔断平仓 (Emergency Stop)',
            '<b>警告：</b>您正在触发最高优先级的<b>急停熔断平仓</b>机制！<br><br>' +
            '系统将立即以市价全部清仓当前所有活跃的多空持仓，撤销全部挂单，并锁定交易系统阻止新开仓。<br><br>' +
            '<span style="color:var(--color-danger);font-weight:700;">确定要立即市价平仓并锁定停机吗？</span>',
            { confirmText: '立即市价全平并停机', cancelText: '取消', danger: true, icon: '<i class="fa-solid fa-radiation"></i>' }
        );
        if (!confirmed) return;
        
        elements.btnEmergencyStop.disabled = true;
        elements.btnEmergencyStop.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> 正在紧急平仓...`;
        
        try {
            const resp = await fetch('/api/oms/emergency_stop', { method: 'POST' });
            const res = await resp.json();
            notify.success(res.message || '急停平仓指令已提交并锁定', 6000, { title: '🚨 急停已执行' });
            // Refresh portfolio instantly
            await updatePortfolio();
        } catch (err) {
            console.error('Emergency stop trigger failed:', err);
            notify.error('紧急平仓执行失败: ' + (err.message || '网络或接口异常'), 6000, { title: '急停失败' });
        } finally {
            elements.btnEmergencyStop.disabled = false;
            elements.btnEmergencyStop.innerHTML = `<i class="fa-solid fa-radiation"></i> EMERGENCY FLAT & STOP`;
        }
    });

    // Lightbox close
    elements.closeLightbox.addEventListener('click', () => {
        elements.lightboxModal.style.display = 'none';
    });
    
    elements.lightboxModal.addEventListener('click', (e) => {
        if (e.target === elements.lightboxModal) {
            elements.lightboxModal.style.display = 'none';
        }
    });

    // File attachment click trigger
    if (elements.btnAttachFile) {
        elements.btnAttachFile.addEventListener('click', () => {
            elements.attachmentFileInput.click();
        });
    }

    if (elements.attachmentFileInput) {
        elements.attachmentFileInput.addEventListener('change', async () => {
            const files = elements.attachmentFileInput.files;
            if (files.length > 0) {
                for (let i = 0; i < files.length; i++) {
                    await uploadFile(files[i]);
                }
                elements.attachmentFileInput.value = '';
            }
        });
    }

    // Drag and drop events on the chat container
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            chatContainer.classList.add('drag-active');
        });
        chatContainer.addEventListener('dragleave', () => {
            chatContainer.classList.remove('drag-active');
        });
        chatContainer.addEventListener('drop', async (e) => {
            e.preventDefault();
            chatContainer.classList.remove('drag-active');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                for (let i = 0; i < files.length; i++) {
                    await uploadFile(files[i]);
                }
            }
        });
    }

    // Clipboard paste events on chat input
    elements.chatInput.addEventListener('paste', async (e) => {
        const clipboardData = e.clipboardData || window.clipboardData;
        const text = clipboardData.getData('text');
        
        if (text && text.length > 2000) {
            e.preventDefault();
            const file = new File([text], `pasted_text_${Date.now()}.txt`, { type: 'text/plain' });
            await uploadFile(file);
            showTypingIndicator('Auto-converted long text to attachment.');
            setTimeout(hideTypingIndicator, 2000);
            return;
        }
        
        // Handle images
        const items = clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                const file = new File([blob], `screenshot_${Date.now()}.png`, { type: blob.type });
                e.preventDefault();
                await uploadFile(file);
                return;
            }
        }
    });

    // Quick Trading Mode Switch click on header badge
    if (elements.headerTradingMode) {
        elements.headerTradingMode.style.cursor = 'pointer';
        elements.headerTradingMode.addEventListener('click', async () => {
            const isLive = elements.headerTradingMode.classList.contains('live');
            if (isLive) {
                const confirmed = await showConfirm(
                    '切换交易模式',
                    '确定切换回 <b>模拟回测 (PAPER)</b> 模式吗？<br>实盘挂单和持仓管理将转为模拟状态。',
                    { confirmText: '切换为模拟模式', cancelText: '取消', danger: false, icon: '<i class="fa-solid fa-flask"></i>' }
                );
                if (confirmed) {
                    await deactivateKmsKey();
                }
            } else {
                const kmsTab = document.querySelector('.tab-btn[data-tab="kms-view"]');
                if (kmsTab) {
                    kmsTab.click();
                    notify.warning(
                        "若要开启 <b>LIVE 实盘交易</b>，请在下方【API 密钥与钱包】列表中激活对应的交易凭据。",
                        5000,
                        { title: '需要激活实盘密钥' }
                    );
                }
            }
        });
    }

    // --- Tab Switching Navigation ---
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            
            // Toggle active class on buttons
            tabButtons.forEach(b => b.classList.toggle('active', b === btn));
            
            // Toggle active class on views
            const viewContainers = document.querySelectorAll('.view-container');
            viewContainers.forEach(container => {
                const isTarget = container.id === targetTab;
                container.classList.toggle('active-view', isTarget);
                container.classList.toggle('hidden-view', !isTarget);
            });
            
            // If switched to KMS tab, load logs and status
            if (targetTab === 'kms-view') {
                checkKmsStatus();
                loadKmsAuditLogs();
            }
            
            // If switched to Exchange tab, load data
            if (targetTab === 'exchange-view') {
                refreshExchangeData();
            }
            
            // If switched to Strategy tab, load data
            if (targetTab === 'strategy-view') {
                loadStrategyDecisions();
            }

            // If switched to Markets tab, load data immediately
            if (targetTab === 'market-view') {
                updateWatchlist();
            }

            // Update URL Hash for SPA Routing
            updateHashForTab(targetTab);
        });
    });

    // --- Strategy Audits Event Listeners ---
    if (elements.btnRefreshStrategy) {
        elements.btnRefreshStrategy.addEventListener('click', loadStrategyDecisions);
    }
    if (elements.strategyIdSelect) {
        elements.strategyIdSelect.addEventListener('change', loadStrategyDecisions);
    }

    // --- Exchange Sub-Tabs switching ---
    const exTabButtons = document.querySelectorAll('.ex-tab-btn');
    exTabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetSub = btn.dataset.exTab;
            
            exTabButtons.forEach(b => b.classList.toggle('active', b === btn));
            
            const subViews = document.querySelectorAll('.ex-sub-view');
            subViews.forEach(view => {
                const isTarget = view.id === targetSub;
                view.classList.toggle('active-ex-view', isTarget);
                view.classList.toggle('hidden-ex-view', !isTarget);
            });
        });
    });

    // --- KMS Event Listeners ---
    if (elements.btnKmsInit) {
        elements.btnKmsInit.addEventListener('click', initKmsVault);
    }
    if (elements.btnKmsUnlock) {
        elements.btnKmsUnlock.addEventListener('click', unlockKmsVault);
    }
    if (elements.btnKmsLock) {
        elements.btnKmsLock.addEventListener('click', lockKmsVault);
    }
    if (elements.btnKmsAddKey) {
        elements.btnKmsAddKey.addEventListener('click', addKmsKey);
    }
    if (elements.btnRefreshExchange) {
        elements.btnRefreshExchange.addEventListener('click', refreshExchangeData);
    }
    if (elements.exchangeWalletSelect) {
        elements.exchangeWalletSelect.addEventListener('change', refreshExchangeData);
    }
    
    // --- Watchlist Event Listeners ---
    if (elements.watchlistSortSelect) {
        elements.watchlistSortSelect.addEventListener('change', (e) => {
            state.watchlistSortOption = e.target.value;
            elements.watchlistContainer.scrollTop = 0;
            updateWatchlist();
        });
    }
}
