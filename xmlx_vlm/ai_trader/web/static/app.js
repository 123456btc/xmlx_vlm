import { state, elements } from './js/core/state.js';
import { initRouter, updateHashForTab } from './js/core/router.js';
import { loadSessions, sendMessage, uploadFile, selectSession } from './js/modules/chat.js';
import { checkKmsStatus, initKmsVault, unlockKmsVault, lockKmsVault, addKmsKey, deactivateKmsKey, loadKmsAuditLogs } from './js/modules/kms.js';
import { refreshExchangeData } from './js/modules/exchange.js';
import { loadStrategyDecisions } from './js/modules/strategy.js';
import { startMarketLoop, startPortfolioLoop, updateWatchlist, updatePortfolio } from './js/modules/market.js';

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
        
        elements.headerTradingMode.textContent = config.mode;
        elements.headerTradingMode.className = `status-val mode-badge ${config.mode.toLowerCase()}`;
        
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
        } catch (err) {
            console.error('Failed to create session:', err);
            alert('Failed to create session');
        }
    });

    // Clear Chat Button
    elements.btnClearChat.addEventListener('click', async () => {
        if (!state.activeSessionId) return;
        if (!confirm('Are you sure you want to clear this session\'s messages?')) return;
        
        try {
            await fetch(`/api/sessions/${state.activeSessionId}`, { method: 'DELETE' });
            // Re-create session with same ID to keep context clean
            await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.activeSessionId, title: elements.activeSessionTitle.textContent })
            });
            selectSession(state.activeSessionId);
        } catch (err) {
            console.error('Failed to clear session:', err);
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
        if (!confirm('🚨 WARNING: You are triggering EMERGENCY LIQUIDATION. This will immediately flat all open positions and block new orders. Proceed?')) {
            return;
        }
        
        elements.btnEmergencyStop.disabled = true;
        elements.btnEmergencyStop.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> LIQUIDATING...`;
        
        try {
            const resp = await fetch('/api/oms/emergency_stop', { method: 'POST' });
            const res = await resp.json();
            alert(res.message);
            // Refresh portfolio instantly
            await updatePortfolio();
        } catch (err) {
            console.error('Emergency stop trigger failed:', err);
            alert('Emergency stop action failed!');
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
                if (confirm("Switch to PAPER trading mode? (All live trading orders will be stopped)")) {
                    await deactivateKmsKey();
                }
            } else {
                const kmsTab = document.querySelector('.tab-btn[data-tab="kms-view"]');
                if (kmsTab) {
                    kmsTab.click();
                    alert("To enable LIVE trading, please activate a credential in the 'API Keys & Wallet' list below.");
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
    }
}
