import { state, elements } from '../core/state.js';

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
    
    if (state.watchlistRotationActive) {
        startWatchlistRotation();
    }
    
    // Pause auto-rotation on hover
    if (elements.watchlistContainer) {
        if (!elements.watchlistContainer._hasHoverListeners) {
            elements.watchlistContainer.addEventListener('mouseenter', () => {
                stopWatchlistRotation();
            });
            elements.watchlistContainer.addEventListener('mouseleave', () => {
                if (state.watchlistRotationActive) {
                    startWatchlistRotation();
                }
            });
            elements.watchlistContainer._hasHoverListeners = true;
        }
    }
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
    
    // Clone items for seamless marquee loop
    const items = Array.from(elements.watchlistContainer.children);
    if (items.length > 0) {
        items.forEach(item => {
            const clone = item.cloneNode(true);
            elements.watchlistContainer.appendChild(clone);
        });
    }
    
    // Restore scroll position or set to middle for rotation room
    if (savedScrollTop > 0) {
        elements.watchlistContainer.scrollTop = savedScrollTop;
    } else if (state.watchlistRotationActive) {
        const halfHeight = elements.watchlistContainer.scrollHeight / 2;
        elements.watchlistContainer.scrollTop = halfHeight;
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

export function startWatchlistRotation() {
    if (state.watchlistRotationInterval) {
        clearInterval(state.watchlistRotationInterval);
    }
    
    state.watchlistRotationInterval = setInterval(() => {
        const container = elements.watchlistContainer;
        if (!container || !state.watchlistRotationActive) return;
        
        const halfHeight = container.scrollHeight / 2;
        const maxScroll = container.scrollHeight - container.clientHeight;
        if (maxScroll <= 0) return;
        
        // Visually scroll down (content moves down, scrollTop decreases)
        container.scrollTop -= 0.3; // 60fps smooth scrolling increment
        
        // Wrap around seamlessly
        if (container.scrollTop <= 0) {
            container.scrollTop = halfHeight;
        }
    }, 16); // ~60fps
}

export function stopWatchlistRotation() {
    if (state.watchlistRotationInterval) {
        clearInterval(state.watchlistRotationInterval);
        state.watchlistRotationInterval = null;
    }
}

// Portfolio & Positions Loop
export function startPortfolioLoop() {
    updatePortfolio();
    state.portfolioInterval = setInterval(updatePortfolio, 4000);
}

export async function updatePortfolio() {
    try {
        const resp = await fetch('/api/oms/portfolio');
        const summary = await resp.json();
        
        // Render Account Header
        const acc = summary.account || {};
        const available = parseFloat(acc.available_margin || 0);
        elements.portfolioBalance.textContent = `$${available.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const unrealized = parseFloat(summary.unrealized_pnl || 0);
        elements.portfolioUnrealized.textContent = `${unrealized >= 0 ? '+' : ''}$${unrealized.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        elements.portfolioUnrealized.className = `summary-val font-mono ${unrealized > 0 ? 'profit' : (unrealized < 0 ? 'loss' : 'neutral')}`;
        
        elements.portfolioMarginUtil.textContent = `${parseInt(summary.margin_utilization_pct || 0)}%`;
        
        // Render Position List
        elements.positionsList.innerHTML = '';
        const positions = summary.positions || [];
        
        if (positions.length === 0) {
            elements.positionsList.innerHTML = '<div class="no-positions">No open positions</div>';
            return;
        }
        
        positions.forEach(pos => {
            const card = document.createElement('div');
            card.className = 'position-card';
            
            const sideClass = pos.side.toLowerCase() === 'long' ? 'long' : 'short';
            const unrealizedPnl = parseFloat(pos.unrealized_pnl || 0);
            const pnlClass = unrealizedPnl > 0 ? 'profit' : (unrealizedPnl < 0 ? 'loss' : 'neutral');
            const formattedPnl = `${unrealizedPnl >= 0 ? '+' : ''}$${unrealizedPnl.toFixed(2)}`;
            
            card.innerHTML = `
                <div class="position-card-header">
                    <span class="pos-sym">${pos.symbol}</span>
                    <span class="pos-side ${sideClass}">${pos.side}</span>
                </div>
                <div class="position-card-body">
                    <div>Size: ${parseFloat(pos.qty).toFixed(4)}</div>
                    <div>Entry: $${formatPrice(pos.avg_entry_price)}</div>
                    <div>Mark: $${formatPrice(pos.mark_price)}</div>
                </div>
                <span class="pos-pnl ${pnlClass}">${formattedPnl}</span>
            `;
            
            elements.positionsList.appendChild(card);
        });
    } catch (err) {
        console.error('Portfolio sync failed:', err);
    }
}
