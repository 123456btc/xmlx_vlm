/**
 * Centralized SPA URL Hash Router
 */

// Route hash -> DOM Container ID mapping
const routeMap = {
    '#/chat': 'chat-view',
    '#/market': 'market-view',
    '#/kms': 'kms-view',
    '#/exchange': 'exchange-view',
    '#/strategy': 'strategy-view'
};

const reverseRouteMap = {
    'chat-view': '#/chat',
    'market-view': '#/market',
    'kms-view': '#/kms',
    'exchange-view': '#/exchange',
    'strategy-view': '#/strategy'
};

// Route lifecycle callbacks registration registry
const routeCallbacks = {};

export function registerRouteCallback(viewId, callback) {
    if (!routeCallbacks[viewId]) {
        routeCallbacks[viewId] = [];
    }
    routeCallbacks[viewId].push(callback);
}

export function handleRouting() {
    const hash = window.location.hash || '#/chat';
    const targetTab = routeMap[hash] || 'chat-view';
    
    // Prevent redundant rendering if the view is already active
    const targetContainer = document.getElementById(targetTab);
    if (targetContainer && targetContainer.classList.contains('active-view')) {
        return;
    }
    
    const tabBtn = document.querySelector(`.tab-btn[data-tab="${targetTab}"]`);
    if (tabBtn) {
        tabBtn.click();
    }
}

// Update Location Hash when user clicks a Tab programmatically
export function updateHashForTab(targetTab) {
    if (reverseRouteMap[targetTab] && window.location.hash !== reverseRouteMap[targetTab]) {
        window.location.hash = reverseRouteMap[targetTab];
    }
    
    // Trigger callbacks registered for this view
    if (routeCallbacks[targetTab]) {
        routeCallbacks[targetTab].forEach(cb => {
            try { cb(); } catch (err) { console.error(`Error in route callback for ${targetTab}:`, err); }
        });
    }
}

export function initRouter() {
    handleRouting();
    window.addEventListener('hashchange', handleRouting);
}
