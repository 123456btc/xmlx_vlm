import { elements } from '../core/state.js';
import { updatePortfolio } from './market.js';

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

export async function refreshExchangeData() {
    if (!elements.btnRefreshExchange) return;
    
    elements.btnRefreshExchange.disabled = true;
    elements.btnRefreshExchange.innerHTML = `<i class="fa-solid fa-rotate fa-spin"></i> Syncing...`;
    
    const selectedWallet = elements.exchangeWalletSelect ? elements.exchangeWalletSelect.value : "";
    const queryParam = selectedWallet ? `?wallet=${encodeURIComponent(selectedWallet)}` : "";
    
    try {
        // 1. Fetch Assets summary
        const assetResp = await fetch(`/api/kms/exchange/assets${queryParam}`);
        if (!assetResp.ok) {
            const err = await assetResp.json();
            if (elements.exchangeActiveAccountLabel) elements.exchangeActiveAccountLabel.textContent = "No Active KMS Wallet Connected";
            if (elements.exTotalEquity) elements.exTotalEquity.textContent = "$0.00";
            if (elements.exAvailMargin) elements.exAvailMargin.textContent = "$0.00";
            if (elements.exUsedMargin) elements.exUsedMargin.textContent = "$0.00";
            if (elements.exMarginUtil) elements.exMarginUtil.textContent = "0%";
            
            if (elements.exPositionsTbody) elements.exPositionsTbody.innerHTML = '<tr><td colspan="8" class="table-empty">Please unlock vault and activate a key to sync exchange data.</td></tr>';
            if (elements.exSpotTbody) elements.exSpotTbody.innerHTML = '<tr><td colspan="4" class="table-empty">Vault is locked.</td></tr>';
            if (elements.exOrdersTbody) elements.exOrdersTbody.innerHTML = '<tr><td colspan="7" class="table-empty">Vault is locked.</td></tr>';
            if (elements.exHistoryTbody) elements.exHistoryTbody.innerHTML = '<tr><td colspan="7" class="table-empty">Vault is locked.</td></tr>';
            return;
        }
        
        const assets = await assetResp.json();
        if (elements.exchangeActiveAccountLabel) {
            elements.exchangeActiveAccountLabel.innerHTML = `<i class="fa-solid fa-circle text-success" style="font-size:8px;"></i> ${assets.label} (${assets.account_address.slice(0, 12)}...${assets.account_address.slice(-10)}) - ${assets.network}`;
        }
        
        if (elements.exTotalEquity) elements.exTotalEquity.textContent = `$${assets.perp_equity.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        if (elements.exAvailMargin) elements.exAvailMargin.textContent = `$${assets.available_margin.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        if (elements.exUsedMargin) elements.exUsedMargin.textContent = `$${assets.used_margin.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        
        const util = assets.perp_equity > 0 ? (assets.used_margin / assets.perp_equity * 100) : 0;
        if (elements.exMarginUtil) elements.exMarginUtil.textContent = `${util.toFixed(1)}%`;
        
        // 2. Fetch Spot Balances
        if (elements.exSpotTbody) {
            elements.exSpotTbody.innerHTML = '';
            if (assets.spot_balances.length === 0) {
                elements.exSpotTbody.innerHTML = '<tr><td colspan="4" class="table-empty">No spot assets.</td></tr>';
            } else {
                assets.spot_balances.forEach(sb => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td class="font-bold">${sb.coin}</td>
                        <td class="font-mono">${sb.total.toLocaleString()}</td>
                        <td class="font-mono text-success">${sb.available.toLocaleString()}</td>
                        <td class="font-mono text-tertiary">${sb.hold.toLocaleString()}</td>
                    `;
                    elements.exSpotTbody.appendChild(tr);
                });
            }
        }
        
        // 3. Fetch Perp Positions
        const posResp = await fetch(`/api/kms/exchange/positions${queryParam}`);
        const positions = await posResp.json();
        if (elements.exPositionsTbody) {
            elements.exPositionsTbody.innerHTML = '';
            if (positions.length === 0) {
                elements.exPositionsTbody.innerHTML = '<tr><td colspan="8" class="table-empty">No open positions.</td></tr>';
            } else {
                positions.forEach(p => {
                    const tr = document.createElement('tr');
                    const sideClass = p.side === 'LONG' ? 'long' : 'short';
                    const pnlClass = p.unrealized_pnl > 0 ? 'profit' : (p.unrealized_pnl < 0 ? 'loss' : 'neutral');
                    
                    tr.innerHTML = `
                        <td class="font-bold">${p.symbol}</td>
                        <td><span class="pos-side ${sideClass}">${p.side}</span></td>
                        <td class="font-mono">${p.qty} (${p.leverage}x ${p.margin_type})</td>
                        <td class="font-mono">$${formatPrice(p.avg_entry_price)}</td>
                        <td class="font-mono">$${formatPrice(p.mark_price)}</td>
                        <td class="font-mono text-danger">$${p.liq_price > 0 ? formatPrice(p.liq_price) : '-'}</td>
                        <td><span class="pos-pnl ${pnlClass}">$${p.unrealized_pnl.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span></td>
                        <td>
                            <button class="btn btn-sm btn-danger btn-close-pos" data-sym="${p.symbol}">
                                <i class="fa-solid fa-circle-xmark"></i> Market Close
                            </button>
                        </td>
                    `;
                    
                    tr.querySelector('.btn-close-pos').addEventListener('click', async (e) => {
                        if (confirm(`Market close position for ${p.symbol}?`)) {
                            const sym = e.currentTarget.dataset.sym;
                            await closeExchangePosition(sym);
                        }
                    });
                    
                    elements.exPositionsTbody.appendChild(tr);
                });
            }
        }
        
        // 4. Fetch Open Orders
        const ordResp = await fetch(`/api/kms/exchange/orders${queryParam}`);
        const orders = await ordResp.json();
        if (elements.exOrdersTbody) {
            elements.exOrdersTbody.innerHTML = '';
            if (orders.length === 0) {
                elements.exOrdersTbody.innerHTML = '<tr><td colspan="7" class="table-empty">No open orders.</td></tr>';
            } else {
                orders.forEach(o => {
                    const tr = document.createElement('tr');
                    const sideClass = o.side === 'BUY' ? 'long' : 'short';
                    const dateStr = new Date(o.timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    
                    tr.innerHTML = `
                        <td class="font-bold">${o.symbol}</td>
                        <td><span class="pos-side ${sideClass}">${o.side}</span></td>
                        <td class="font-mono">${o.qty}</td>
                        <td class="font-mono">$${formatPrice(o.price)}</td>
                        <td>${o.type}</td>
                        <td>${dateStr}</td>
                        <td>
                            <button class="btn btn-sm btn-danger btn-cancel-order" data-id="${o.order_id}">
                                Cancel
                            </button>
                        </td>
                    `;
                    
                    tr.querySelector('.btn-cancel-order').addEventListener('click', async (e) => {
                        const oid = e.currentTarget.dataset.id;
                        await cancelExchangeOrder(oid);
                    });
                    
                    elements.exOrdersTbody.appendChild(tr);
                });
            }
        }
        
        // 5. Fetch History
        const histResp = await fetch(`/api/kms/exchange/history${queryParam}`);
        const history = await histResp.json();
        if (elements.exHistoryTbody) {
            elements.exHistoryTbody.innerHTML = '';
            if (history.length === 0) {
                elements.exHistoryTbody.innerHTML = '<tr><td colspan="7" class="table-empty">No execution history.</td></tr>';
            } else {
                history.forEach(h => {
                    const tr = document.createElement('tr');
                    const sideClass = h.side === 'BUY' ? 'long' : 'short';
                    const pnlClass = h.pnl > 0 ? 'profit' : (h.pnl < 0 ? 'loss' : 'neutral');
                    const dateStr = new Date(h.timestamp * 1000).toLocaleString([], {month: '2-digit', day: '2-digit', hour: '2-digit', minute:'2-digit'});
                    
                    tr.innerHTML = `
                        <td class="text-secondary">${dateStr}</td>
                        <td class="font-bold">${h.symbol}</td>
                        <td><span class="pos-side ${sideClass}">${h.side}</span></td>
                        <td class="font-mono">${h.qty}</td>
                        <td class="font-mono">$${formatPrice(h.price)}</td>
                        <td class="font-mono text-tertiary">$${h.fee}</td>
                        <td><span class="pos-pnl ${pnlClass}">${h.pnl > 0 ? '+' : ''}$${h.pnl.toFixed(2)}</span></td>
                    `;
                    elements.exHistoryTbody.appendChild(tr);
                });
            }
        }
        
        // Also refresh regular OMS side panel
        await updatePortfolio();
    } catch (err) {
        console.error("Exchange data sync failed:", err);
    } finally {
        if (elements.btnRefreshExchange) {
            elements.btnRefreshExchange.disabled = false;
            elements.btnRefreshExchange.innerHTML = `<i class="fa-solid fa-rotate"></i> Sync Data`;
        }
    }
}

export async function cancelExchangeOrder(orderId) {
    try {
        const selectedWallet = elements.exchangeWalletSelect ? elements.exchangeWalletSelect.value : "";
        const resp = await fetch('/api/kms/exchange/cancel_order', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ order_id: orderId, wallet: selectedWallet })
        });
        if (resp.ok) {
            await refreshExchangeData();
        } else {
            const res = await resp.json();
            alert("Cancel order failed: " + res.detail);
        }
    } catch (err) {
        console.error("Cancel exchange order failed:", err);
    }
}

export async function closeExchangePosition(symbol) {
    try {
        const selectedWallet = elements.exchangeWalletSelect ? elements.exchangeWalletSelect.value : "";
        const resp = await fetch('/api/kms/exchange/close_position', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, wallet: selectedWallet })
        });
        if (resp.ok) {
            await refreshExchangeData();
        } else {
            const res = await resp.json();
            alert("Close position failed: " + res.detail);
        }
    } catch (err) {
        console.error("Close position failed:", err);
    }
}
