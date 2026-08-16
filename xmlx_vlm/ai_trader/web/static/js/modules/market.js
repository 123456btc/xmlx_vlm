import { state, elements } from '../core/state.js?v=2.0.1';

function formatPrice(val) {
    const p = parseFloat(val);
    if (isNaN(p)) return '-';
    const absP = Math.abs(p);
    if (absP === 0) return '0.00';
    if (absP < 0.0001) {
        return p.toLocaleString('en-US', { minimumFractionDigits: 8, maximumFractionDigits: 8 });
    }
    if (absP < 0.01) {
        return p.toLocaleString('en-US', { minimumFractionDigits: 6, maximumFractionDigits: 6 });
    }
    if (absP < 0.1) {
        return p.toLocaleString('en-US', { minimumFractionDigits: 5, maximumFractionDigits: 5 });
    }
    if (absP < 1) {
        return p.toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 });
    }
    return p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// Tickers Sidebar (Watchlist) Loop
export function startMarketLoop() {
    connectWatchlistWs();
}

export function connectWatchlistWs() {
    if (state.watchlistWs) {
        if (state.watchlistWs.readyState === WebSocket.CONNECTING || state.watchlistWs.readyState === WebSocket.OPEN) {
            return;
        }
        try {
            state.watchlistWs.close();
        } catch (e) {}
    }
    
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/api/market/watchlist/ws`;
    
    state.watchlistWs = new WebSocket(wsUrl);
    
    state.watchlistWs.onopen = () => {
        console.log('Market Watchlist WebSocket connected');
    };
    
    state.watchlistWs.onmessage = (event) => {
        try {
            const list = JSON.parse(event.data);
            state.currentWatchlist = list;
            renderWatchlist(list);
        } catch (err) {
            console.error('Failed to parse watchlist WS message:', err);
        }
    };
    
    state.watchlistWs.onclose = () => {
        console.log('Market Watchlist WebSocket closed. Reconnecting in 3s...');
        state.watchlistWs = null;
        setTimeout(connectWatchlistWs, 3000);
    };
    
    state.watchlistWs.onerror = (err) => {
        console.error('Market Watchlist WS Error:', err);
    };
}

export function renderWatchlist(list) {
    if (!list || list.length === 0) {
        elements.watchlistContainer.innerHTML = '<div class="no-positions">No watch symbols.</div>';
        return;
    }
    
    // Sort list
    list.sort((a, b) => {
        let valA = a[state.watchlistSortOption] !== undefined ? a[state.watchlistSortOption] : 0;
        let valB = b[state.watchlistSortOption] !== undefined ? b[state.watchlistSortOption] : 0;
        
        if (state.watchlistSortOption === 'funding_rate') {
            return Math.abs(valB) - Math.abs(valA);
        }
        return valB - valA;
    });
    
    // Save current scroll position before re-rendering
    const savedScrollTop = elements.watchlistContainer.scrollTop;
    elements.watchlistContainer.innerHTML = '';
    
    let hasTicks = false;
    list.forEach(item => {
        const sym = item.symbol;
        const price = item.price;
        const change = item.change_24h_pct;
        const volume = item.volume_24h || 0;
        
        // Check price change to flash tick colors
        let tickClass = '';
        if (state.priceCache[sym] !== undefined) {
            if (price > state.priceCache[sym]) {
                tickClass = 'tick-up';
                hasTicks = true;
            } else if (price < state.priceCache[sym]) {
                tickClass = 'tick-down';
                hasTicks = true;
            }
        }
        state.priceCache[sym] = price;
        
        const div = document.createElement('div');
        div.className = 'watchlist-item';
        
        const formattedPrice = price ? formatPrice(price) : '-';
        const formattedChange = change ? `${change > 0 ? '+' : ''}${change.toFixed(2)}%` : '0.00%';
        const changeClass = change > 0 ? 'positive' : (change < 0 ? 'negative' : '');
        const formattedVolume = volume ? `$${(volume / 1e6).toFixed(2)}M` : '$0.00M';
        const fundingPct = item.funding_rate ? `${(item.funding_rate * 100).toFixed(4)}%` : '0.0000%';
        
        div.innerHTML = `
            <div class="watchlist-sym-group">
                <span class="watchlist-sym">${sym}</span>
                <span class="watchlist-meta">Vol: ${formattedVolume} | Fdg: ${fundingPct}</span>
            </div>
            <div class="watchlist-price-group">
                <span class="watchlist-price ${tickClass}">$${formattedPrice}</span>
                <span class="watchlist-change ${changeClass}">${formattedChange}</span>
            </div>
        `;
        
        elements.watchlistContainer.appendChild(div);
    });
    
    // Restore scroll position
    if (savedScrollTop > 0) {
        elements.watchlistContainer.scrollTop = savedScrollTop;
    }
    
    // Remove flash classes from both original and clone elements after 800ms
    if (hasTicks) {
        setTimeout(() => {
            const flashingElements = elements.watchlistContainer.querySelectorAll('.tick-up, .tick-down');
            flashingElements.forEach(el => el.classList.remove('tick-up', 'tick-down'));
        }, 800);
    }
}

export async function updateWatchlist() {
    if (state.watchlistWs && state.watchlistWs.readyState === WebSocket.OPEN && state.currentWatchlist && state.currentWatchlist.length > 0) {
        renderWatchlist(state.currentWatchlist);
        return;
    }
    
    try {
        const resp = await fetch('/api/market/watchlist');
        let list = await resp.json();
        state.currentWatchlist = list;
        renderWatchlist(list);
    } catch (err) {
        console.error('Watchlist fetch failed:', err);
    }
}


// Portfolio & Positions Loop
export function startPortfolioLoop() {
    updatePortfolio();
    // 15s is sufficient for portfolio display — positions change slowly.
    // The previous 4s interval caused full DOM rebuilds 15x/min, adding
    // layout thrash cost on top of the watchlist WS updates.
    state.portfolioInterval = setInterval(updatePortfolio, 15000);
}

export async function updatePortfolio() {
    try {
        const resp = await fetch('/api/oms/portfolio');
        const summary = await resp.json();
        
        // 1. Account Metadata & Network Tag
        const isLive = summary.is_live === true || summary.is_live === 'true';
        if (elements.portfolioModeBadge) {
            elements.portfolioModeBadge.className = `mode-badge ${isLive ? 'live' : 'paper'}`;
            elements.portfolioModeBadge.innerHTML = isLive ? '<i class="fa-solid fa-bolt"></i> LIVE 实盘' : '<i class="fa-solid fa-flask"></i> PAPER 模拟';
        }
        
        if (elements.headerTradingMode) {
            elements.headerTradingMode.className = `status-val mode-badge ${isLive ? 'live' : 'paper'}`;
            elements.headerTradingMode.textContent = isLive ? `LIVE (${summary.label || 'Hyperliquid'})` : 'PAPER (Sim)';
        }

        if (elements.portfolioNetworkTag) {
            if (isLive) {
                elements.portfolioNetworkTag.innerHTML = `<i class="fa-solid fa-shield-halved"></i> ${summary.label || 'Hyperliquid'} ${summary.network || 'Mainnet'}`;
                elements.portfolioNetworkTag.className = 'account-network-tag live';
            } else {
                elements.portfolioNetworkTag.innerHTML = '<i class="fa-solid fa-box-archive"></i> Local Paper Simulator';
                elements.portfolioNetworkTag.className = 'account-network-tag paper';
            }
        }

        if (elements.portfolioWalletAddr && elements.portfolioWalletBox) {
            if (isLive && summary.wallet_address) {
                const addr = summary.wallet_address;
                elements.portfolioWalletAddr.textContent = `${addr.slice(0, 6)}...${addr.slice(-4)}`;
                elements.portfolioWalletBox.style.display = 'inline-flex';
                elements.portfolioWalletBox.title = `点击复制完整钱包地址: ${addr}`;
                elements.portfolioWalletBox.onclick = async () => {
                    try {
                        await navigator.clipboard.writeText(addr);
                        if (notify && notify.success) {
                            notify.success(`已复制真实钱包地址: ${addr}`);
                        } else {
                            showToast(`已复制真实钱包地址: ${addr}`, 'success');
                        }
                    } catch (e) {
                        showToast(`钱包地址: ${addr}`, 'info', 6000);
                    }
                };
            } else {
                elements.portfolioWalletAddr.textContent = 'Virtual / Paper';
                elements.portfolioWalletBox.title = '当前为虚拟模拟账户';
                elements.portfolioWalletBox.onclick = null;
            }
        }

        // 2. Render Account Header & Financial Metrics
        const acc = summary.account || {};
        const available = parseFloat(acc.available_margin || 0);
        const equity = parseFloat(acc.equity || available || 0);
        
        if (elements.portfolioEquity) {
            elements.portfolioEquity.textContent = `$${equity.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        }
        
        if (elements.portfolioBalance) {
            elements.portfolioBalance.textContent = `$${available.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        }
        
        const unrealized = parseFloat(summary.unrealized_pnl || 0);
        if (elements.portfolioUnrealized) {
            elements.portfolioUnrealized.textContent = `${unrealized >= 0 ? '+' : ''}$${unrealized.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            elements.portfolioUnrealized.className = `summary-val font-mono ${unrealized > 0 ? 'profit' : (unrealized < 0 ? 'loss' : 'neutral')}`;
        }
        
        if (elements.portfolioMarginUtil) {
            elements.portfolioMarginUtil.textContent = `${parseInt(summary.margin_utilization_pct || 0)}%`;
        }
        
        // 3. Render Position List
        if (elements.positionsList) {
            elements.positionsList.innerHTML = '';
            const positions = summary.positions || [];
            
            if (positions.length === 0) {
                elements.positionsList.innerHTML = '<div class="no-positions">No open positions</div>';
                return;
            }
            
            positions.forEach(pos => {
                const card = document.createElement('div');
                card.className = 'position-card';
                
                const sideClass = (pos.side || '').toLowerCase() === 'long' ? 'long' : 'short';
                const unrealizedPnl = parseFloat(pos.unrealized_pnl || 0);
                const pnlClass = unrealizedPnl > 0 ? 'profit' : (unrealizedPnl < 0 ? 'loss' : 'neutral');
                const formattedPnl = `${unrealizedPnl >= 0 ? '+' : ''}$${unrealizedPnl.toFixed(2)}`;
                
                card.innerHTML = `
                    <div class="position-card-header">
                        <span class="pos-sym">${pos.symbol}</span>
                        <span class="pos-side ${sideClass}">${pos.side}</span>
                    </div>
                    <div class="position-card-body">
                        <div>Size: ${parseFloat(pos.qty || 0).toFixed(4)}</div>
                        <div>Entry: $${formatPrice(pos.avg_entry_price)}</div>
                        <div>Mark: $${formatPrice(pos.mark_price)}</div>
                    </div>
                    <span class="pos-pnl ${pnlClass}">${formattedPnl}</span>
                `;
                
                elements.positionsList.appendChild(card);
            });
        }
    } catch (err) {
        console.error('Portfolio sync failed:', err);
    }
}
