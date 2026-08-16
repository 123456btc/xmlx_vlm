import { state, elements } from '../core/state.js?v=2.0.3';

let strategyDecisionsCache = [];

export async function populateStrategySelector() {
    if (!elements.strategyIdSelect) return;
    
    try {
        const resp = await fetch('/api/strategy/list');
        if (!resp.ok) throw new Error("Strategy list request failed");
        const list = await resp.json();
        
        if (list && list.length > 0) {
            const currentSelected = elements.strategyIdSelect.value;
            elements.strategyIdSelect.innerHTML = '';
            
            // Normalize list items (handling both object format and legacy string array)
            const normalized = list.map(item => {
                if (typeof item === 'string') {
                    const isLive = item.includes('hyperliquid') || (!item.includes('paper') && !item.includes('test'));
                    const typeLabel = item.includes('hyperliquid') ? '【🟢 实盘 Hyperliquid】' : (item.includes('paper') ? '【🟡 模拟盘 Paper】' : '【策略】');
                    return { id: item, label: `${typeLabel} ${item}`, is_live: isLive };
                }
                return item;
            });
            
            // Create option groups
            const liveGroup = document.createElement('optgroup');
            liveGroup.label = "🟢 实盘交易策略 (Live Strategies)";
            
            const paperGroup = document.createElement('optgroup');
            paperGroup.label = "🟡 模拟盘策略 (Paper / Sim Strategies)";
            
            const otherGroup = document.createElement('optgroup');
            otherGroup.label = "📁 历史/其他策略 (Historical / Other)";
            
            let defaultId = null;
            
            normalized.forEach(item => {
                const option = document.createElement('option');
                option.value = item.id;
                option.textContent = item.label || item.id;
                
                if (item.is_live || item.id.includes('hyperliquid')) {
                    liveGroup.appendChild(option);
                    if (!defaultId) defaultId = item.id;
                } else if (item.id.includes('paper')) {
                    paperGroup.appendChild(option);
                    if (!defaultId) defaultId = item.id;
                } else {
                    otherGroup.appendChild(option);
                    if (!defaultId) defaultId = item.id;
                }
            });
            
            if (liveGroup.children.length > 0) elements.strategyIdSelect.appendChild(liveGroup);
            if (paperGroup.children.length > 0) elements.strategyIdSelect.appendChild(paperGroup);
            if (otherGroup.children.length > 0) elements.strategyIdSelect.appendChild(otherGroup);
            
            const allIds = normalized.map(i => i.id);
            // Restore previous selection or default to trend_follow_all_hyperliquid / latest
            if (currentSelected && allIds.includes(currentSelected)) {
                elements.strategyIdSelect.value = currentSelected;
            } else if (allIds.includes('trend_follow_all_hyperliquid')) {
                elements.strategyIdSelect.value = 'trend_follow_all_hyperliquid';
            } else if (defaultId) {
                elements.strategyIdSelect.value = defaultId;
            } else {
                elements.strategyIdSelect.value = allIds[0];
            }
        }
    } catch (e) {
        console.error("Failed to populate strategy selector from strategy list:", e);
        
        // Fallback to watchlist
        try {
            const resp = await fetch('/api/market/watchlist');
            if (!resp.ok) throw new Error("Watchlist request failed");
            const watchlist = await resp.json();
            if (watchlist && watchlist.length > 0) {
                const currentSelected = elements.strategyIdSelect.value;
                elements.strategyIdSelect.innerHTML = '';
                
                const liveOpt = document.createElement('option');
                liveOpt.value = "trend_follow_all_hyperliquid";
                liveOpt.textContent = "【🟢 实盘】trend_follow_all_hyperliquid (Hyperliquid 全币种)";
                elements.strategyIdSelect.appendChild(liveOpt);

                const paperOpt = document.createElement('option');
                paperOpt.value = "trend_follow_all_paper";
                paperOpt.textContent = "【🟡 模拟盘】trend_follow_all_paper (Paper 全币种)";
                elements.strategyIdSelect.appendChild(paperOpt);

                elements.strategyIdSelect.value = "trend_follow_all_hyperliquid";
            }
        } catch (e2) {
            console.error("Watchlist fallback failed:", e2);
        }
    }
}

export function initStrategy() {
    if (elements.strategyIdSelect && !elements.strategyIdSelect._hasPopulated) {
        elements.strategyIdSelect._hasPopulated = true;
        // Populate options asynchronously
        populateStrategySelector().then(() => {
            // Trigger initial load after populating selector
            if (elements.strategyDecisionsList && elements.strategyDecisionsList.innerHTML.includes('Loading decisions...')) {
                loadStrategyDecisions();
            }
        });
    }
    if (elements.btnPrevStrategyPage && !elements.btnPrevStrategyPage._hasListener) {
        elements.btnPrevStrategyPage.addEventListener('click', () => {
            if (state.strategyPage > 1) {
                state.strategyPage--;
                loadStrategyDecisions();
            }
        });
        elements.btnPrevStrategyPage._hasListener = true;
    }
    if (elements.btnNextStrategyPage && !elements.btnNextStrategyPage._hasListener) {
        elements.btnNextStrategyPage.addEventListener('click', () => {
            state.strategyPage++;
            loadStrategyDecisions();
        });
        elements.btnNextStrategyPage._hasListener = true;
    }
    
    // Reset page index to 1 when trader ID changes
    if (elements.strategyIdSelect && !elements.strategyIdSelect._hasPageReset) {
        elements.strategyIdSelect.addEventListener('change', () => {
            state.strategyPage = 1;
        });
        elements.strategyIdSelect._hasPageReset = true;
    }
    
    // Reset page index to 1 on manual refresh
    if (elements.btnRefreshStrategy && !elements.btnRefreshStrategy._hasPageReset) {
        elements.btnRefreshStrategy.addEventListener('click', () => {
            state.strategyPage = 1;
        });
        elements.btnRefreshStrategy._hasPageReset = true;
    }
}

export async function loadStrategyDecisions() {
    initStrategy();
    
    if (!elements.strategyDecisionsList) return;
    
    // Clear list and show loading
    elements.strategyDecisionsList.innerHTML = '<div class="loading-placeholder"><i class="fa-solid fa-circle-notch fa-spin"></i> Loading decisions...</div>';
    
    // Reset detail view to empty state
    if (elements.auditEmptyState) elements.auditEmptyState.style.display = 'block';
    if (elements.auditReplayContent) elements.auditReplayContent.style.display = 'none';

    const itemsPerPage = 5;
    try {
        const traderId = elements.strategyIdSelect ? elements.strategyIdSelect.value : "trend_follow_all_hyperliquid";
        
        // Fetch itemsPerPage + 1 to check if there is a next page
        const resp = await fetch(`/api/strategy/decisions?trader_id=${traderId}&limit=${itemsPerPage + 1}&offset=${(state.strategyPage - 1) * itemsPerPage}`);
        if (!resp.ok) throw new Error("API request failed");
        
        const data = await resp.json();
        const hasNextPage = data.length > itemsPerPage;
        const displayData = hasNextPage ? data.slice(0, itemsPerPage) : data;
        
        strategyDecisionsCache = displayData;
        
        // Update pagination UI elements
        if (elements.strategyPageNum) {
            elements.strategyPageNum.textContent = `Page ${state.strategyPage}`;
        }
        if (elements.btnPrevStrategyPage) {
            elements.btnPrevStrategyPage.disabled = state.strategyPage <= 1;
        }
        if (elements.btnNextStrategyPage) {
            elements.btnNextStrategyPage.disabled = !hasNextPage;
        }
        
        if (!displayData || displayData.length === 0) {
            elements.strategyDecisionsList.innerHTML = '<div class="table-empty"><i class="fa-solid fa-folder-open"></i> No strategy logs found for this strategy</div>';
            return;
        }

        elements.strategyDecisionsList.innerHTML = '';
        displayData.forEach((item, index) => {
            const date = new Date(item.timestamp);
            const timeStr = date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
            const dateStr = date.toLocaleDateString([], {month: 'short', day: 'numeric'});
            
            // Get major decision action
            let actionText = "HOLD";
            let badgeClass = "status-hold";
            let actionIcon = "fa-circle";
            
            if (item.decisions && item.decisions.length > 0) {
                const primaryAction = item.decisions[0].action.toLowerCase();
                if (primaryAction.includes("long") || primaryAction.includes("buy")) {
                    actionText = "BUY";
                    badgeClass = "status-buy";
                    actionIcon = "fa-circle-chevron-up";
                } else if (primaryAction.includes("short") || primaryAction.includes("sell")) {
                    actionText = "SELL";
                    badgeClass = "status-sell";
                    actionIcon = "fa-circle-chevron-down";
                } else if (primaryAction === "wait") {
                    actionText = "WAIT";
                    badgeClass = "status-wait";
                    actionIcon = "fa-ban";
                }
            }
            
            const card = document.createElement('div');
            card.className = 'audit-timeline-item';
            card.dataset.index = index;
            card.innerHTML = `
                <div class="timeline-dot-wrapper">
                    <span class="timeline-dot ${badgeClass}"></span>
                </div>
                <div class="timeline-card-content">
                    <div class="card-meta">
                        <span class="card-time font-mono">${dateStr} ${timeStr}</span>
                        <span class="status-badge ${badgeClass}"><i class="fa-solid ${actionIcon}"></i> ${actionText}</span>
                    </div>
                    <div class="card-title">Cycle #${item.cycle_number}</div>
                    <div class="card-desc">${item.decisions[0]?.reasoning || 'No action directive'}</div>
                </div>
            `;
            
            card.addEventListener('click', () => {
                // Highlight active card
                document.querySelectorAll('.audit-timeline-item').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                renderDecisionDetail(item);
            });
            
            elements.strategyDecisionsList.appendChild(card);
        });

        // Automatically select the first decision item if available
        if (displayData.length > 0) {
            const firstCard = elements.strategyDecisionsList.querySelector('.audit-timeline-item');
            if (firstCard) {
                firstCard.classList.add('active');
                renderDecisionDetail(displayData[0]);
            }
        }

    } catch (err) {
        console.error("Failed to load strategy decisions:", err);
        elements.strategyDecisionsList.innerHTML = `<div class="table-empty text-danger"><i class="fa-solid fa-circle-exclamation"></i> Error loading strategy logs</div>`;
    }
}

export function renderDecisionDetail(item) {
    if (!elements.auditReplayContent) return;
    
    if (elements.auditEmptyState) elements.auditEmptyState.style.display = 'none';
    elements.auditReplayContent.style.display = 'block';
    
    // Set Mode and Strategy Name badges
    const traderId = item.trader_id || (elements.strategyIdSelect ? elements.strategyIdSelect.value : '');
    const isLive = traderId.includes('hyperliquid') || (!traderId.includes('paper') && !traderId.includes('test'));
    
    if (elements.auditModeBadge) {
        if (isLive) {
            elements.auditModeBadge.className = 'mode-badge live';
            elements.auditModeBadge.textContent = '🟢 LIVE 实盘';
        } else {
            elements.auditModeBadge.className = 'mode-badge paper';
            elements.auditModeBadge.textContent = '🟡 PAPER 模拟';
        }
    }
    
    if (elements.auditStrategyName) {
        elements.auditStrategyName.textContent = traderId || '-';
    }

    // Set headers
    if (elements.auditCycle) elements.auditCycle.textContent = `#${item.cycle_number}`;
    if (elements.auditTimestamp) {
        const date = new Date(item.timestamp);
        elements.auditTimestamp.textContent = date.toLocaleString();
    }
    if (elements.auditLatency) elements.auditLatency.textContent = `${item.latency_ms} ms`;
    
    // Set badge
    if (elements.auditActionBadge) {
        elements.auditActionBadge.className = 'status-badge';
        let actionText = "HOLD";
        let badgeClass = "status-hold";
        
        if (item.decisions && item.decisions.length > 0) {
            const primaryAction = item.decisions[0].action.toLowerCase();
            if (primaryAction.includes("long") || primaryAction.includes("buy")) {
                actionText = "BUY / LONG";
                badgeClass = "status-buy";
            } else if (primaryAction.includes("short") || primaryAction.includes("sell")) {
                actionText = "SELL / SHORT";
                badgeClass = "status-sell";
            } else if (primaryAction === "wait") {
                actionText = "WAIT / INTERCEPT";
                badgeClass = "status-wait";
            }
        }
        elements.auditActionBadge.textContent = actionText;
        elements.auditActionBadge.classList.add(badgeClass);
    }
    
    // Render COT Trace (using marked.js if available)
    if (elements.auditCotTrace) {
        if (typeof marked !== 'undefined') {
            elements.auditCotTrace.innerHTML = marked.parse(item.cot_trace || '*No trace reasoning logged.*');
        } else {
            // Fallback to pre-formatted text
            elements.auditCotTrace.textContent = item.cot_trace || 'No trace reasoning logged.';
        }
    }
    
    // Set Prompts
    if (elements.auditUserPrompt) elements.auditUserPrompt.textContent = item.user_prompt || 'N/A';
    if (elements.auditSystemPrompt) elements.auditSystemPrompt.textContent = item.system_prompt || 'N/A';
    
    // Set Directive JSON
    if (elements.auditDirectiveJson) {
        try {
            // Format JSON prettily
            const directive = item.decisions || [];
            elements.auditDirectiveJson.textContent = JSON.stringify(directive, null, 2);
        } catch (e) {
            elements.auditDirectiveJson.textContent = '[]';
        }
    }
}

