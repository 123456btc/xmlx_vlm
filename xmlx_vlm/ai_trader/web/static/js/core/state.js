/**
 * Global App State & DOM Elements Selector
 */

// Global State object to share reactive state between modules
export const state = {
    activeSessionId: null,
    chatWs: null,
    watchlistWs: null,
    currentWatchlist: [],
    pendingChatMessages: [],
    watchlistInterval: null,
    portfolioInterval: null,
    priceCache: {}, // Cache symbol -> price to compute tick directions
    watchlistSortOption: 'volume_24h',
    watchlistRotationActive: true,
    watchlistRotationInterval: null,
    stagedAttachmentsList: [],
    strategyPage: 1
};

// Centralized DOM selectors dict
export const elements = {
    headerModelName: document.getElementById('header-model-name'),
    headerTradingMode: document.getElementById('header-trading-mode'),
    connectionDot: document.getElementById('server-connection-dot'),
    connectionStatus: document.getElementById('server-connection-status'),
    
    configRisk: document.getElementById('config-risk'),
    
    btnNewSession: document.getElementById('btn-new-session'),
    btnClearChat: document.getElementById('btn-clear-chat'),
    btnSendMessage: document.getElementById('btn-send-message'),
    btnEmergencyStop: document.getElementById('btn-emergency-stop'),
    
    sessionsList: document.getElementById('sessions-list'),
    chatTimeline: document.getElementById('chat-timeline'),
    chatInput: document.getElementById('chat-input'),
    activeSessionTitle: document.getElementById('active-session-title'),
    activeSessionMeta: document.getElementById('active-session-meta') || document.getElementById('active-meta'),
    
    typingIndicator: document.getElementById('typing-indicator'),
    typingStatusText: document.getElementById('typing-status-text'),
    
    portfolioBalance: document.getElementById('portfolio-balance'),
    portfolioUnrealized: document.getElementById('portfolio-unrealized'),
    portfolioMarginUtil: document.getElementById('portfolio-margin-util'),
    positionsList: document.getElementById('positions-list'),
    watchlistContainer: document.getElementById('watchlist-container'),
    watchlistSortSelect: document.getElementById('watchlist-sort-select'),
    btnWatchlistRotation: document.getElementById('btn-watchlist-rotation'),
    
    lightboxModal: document.getElementById('lightbox-modal'),
    lightboxImg: document.getElementById('lightbox-img'),
    lightboxCaption: document.getElementById('lightbox-caption'),
    closeLightbox: document.querySelector('.close-lightbox'),
    btnAttachFile: document.getElementById('btn-attach-file'),
    attachmentFileInput: document.getElementById('attachment-file-input'),
    stagedAttachments: document.getElementById('staged-attachments'),

    // KMS DOM Elements
    kmsHsmBadge: document.getElementById('kms-hsm-badge'),
    kmsInitSection: document.getElementById('kms-init-section'),
    kmsLockSection: document.getElementById('kms-lock-section'),
    kmsUnlockedSection: document.getElementById('kms-unlocked-section'),
    
    kmsInitPwd: document.getElementById('kms-init-pwd'),
    kmsInitPwdConfirm: document.getElementById('kms-init-pwd-confirm'),
    btnKmsInit: document.getElementById('btn-kms-init'),
    
    kmsMasterPwd: document.getElementById('kms-master-pwd'),
    btnKmsUnlock: document.getElementById('btn-kms-unlock'),
    btnKmsLock: document.getElementById('btn-kms-lock'),
    
    kmsUnlockInputs: document.getElementById('kms-unlock-inputs'),
    kmsLockInputs: document.getElementById('kms-lock-inputs'),
    kmsLockTitle: document.getElementById('kms-lock-title'),
    kmsLockDesc: document.getElementById('kms-lock-desc'),
    kmsLockIcon: document.getElementById('kms-lock-icon'),
    
    kmsActiveSignerLabel: document.getElementById('kms-active-signer-label'),
    kmsActiveSignerSub: document.getElementById('kms-active-signer-sub'),
    kmsKeysTbody: document.getElementById('kms-keys-tbody'),
    
    kmsNewLabel: document.getElementById('kms-new-label'),
    kmsNewAddress: document.getElementById('kms-new-address'),
    kmsNewKey: document.getElementById('kms-new-key'),
    kmsNewTestnet: document.getElementById('kms-new-testnet'),
    kmsReverifyPwd: document.getElementById('kms-reverify-pwd'),
    btnKmsAddKey: document.getElementById('btn-kms-add-key'),
    kmsAuditLogs: document.getElementById('kms-audit-logs'),
    
    // Exchange Dashboard Elements
    btnRefreshExchange: document.getElementById('btn-refresh-exchange'),
    exchangeActiveAccountLabel: document.getElementById('exchange-active-account-label'),
    exTotalEquity: document.getElementById('ex-total-equity'),
    exAvailMargin: document.getElementById('ex-avail-margin'),
    exUsedMargin: document.getElementById('ex-used-margin'),
    exMarginUtil: document.getElementById('ex-margin-util'),
    
    exPositionsTbody: document.getElementById('ex-positions-tbody'),
    exSpotTbody: document.getElementById('ex-spot-tbody'),
    exOrdersTbody: document.getElementById('ex-orders-tbody'),
    exHistoryTbody: document.getElementById('ex-history-tbody'),
    exchangeWalletSelect: document.getElementById('exchange-wallet-select'),

    // Strategy Audit Elements
    btnRefreshStrategy: document.getElementById('btn-refresh-strategy'),
    strategyIdSelect: document.getElementById('strategy-id-select'),
    strategyDecisionsList: document.getElementById('strategy-decisions-list'),
    auditEmptyState: document.getElementById('audit-empty-state'),
    auditReplayContent: document.getElementById('audit-replay-content'),
    auditCycle: document.getElementById('audit-cycle'),
    auditTimestamp: document.getElementById('audit-timestamp'),
    auditLatency: document.getElementById('audit-latency'),
    auditActionBadge: document.getElementById('audit-action-badge'),
    auditCotTrace: document.getElementById('audit-cot-trace'),
    auditUserPrompt: document.getElementById('audit-user-prompt'),
    auditSystemPrompt: document.getElementById('audit-system-prompt'),
    auditDirectiveJson: document.getElementById('audit-directive-json'),
    btnPrevStrategyPage: document.getElementById('btn-prev-strategy-page'),
    btnNextStrategyPage: document.getElementById('btn-next-strategy-page'),
    strategyPageNum: document.getElementById('strategy-page-num'),
};
