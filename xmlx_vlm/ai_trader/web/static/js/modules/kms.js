import { elements } from '../core/state.js';
import { escapeHtml, formatBytes } from '../core/utils.js';
import { refreshExchangeData } from './exchange.js';
import { fetchConfig } from '../../app.js';

export async function checkKmsStatus() {
    try {
        const resp = await fetch('/api/kms/status', { method: 'POST' });
        const status = await resp.json();
        
        // Always show the unlocked credentials section
        if (elements.kmsUnlockedSection) {
            elements.kmsUnlockedSection.classList.remove('hidden');
        }
        if (elements.kmsInitSection) {
            elements.kmsInitSection.classList.add('hidden');
        }
        if (elements.kmsLockSection) {
            elements.kmsLockSection.classList.add('hidden');
        }
        
        // Load credentials list
        await listKmsKeys();
        
        // Active Signer Label
        if (elements.kmsActiveSignerLabel && elements.kmsActiveSignerSub) {
            if (status.active_key) {
                elements.kmsActiveSignerLabel.textContent = status.active_key.label;
                elements.kmsActiveSignerSub.textContent = `Wallet: ${status.active_key.wallet_address.slice(0, 8)}...${status.active_key.wallet_address.slice(-6)} (${status.active_key.testnet ? 'Testnet' : 'Mainnet'})`;
            } else {
                elements.kmsActiveSignerLabel.textContent = "No active key";
                elements.kmsActiveSignerSub.textContent = "Please set an active signer below.";
            }
        }
    } catch (err) {
        console.error("Failed to fetch KMS status:", err);
    }
}

export async function initKmsVault() {
    const password = elements.kmsInitPwd.value;
    const confirm = elements.kmsInitPwdConfirm.value;
    
    if (!password || password.length < 6) {
        alert("Password must be at least 6 characters long.");
        return;
    }
    if (password !== confirm) {
        alert("Passwords do not match.");
        return;
    }
    
    try {
        const resp = await fetch('/api/kms/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password })
        });
        const res = await resp.json();
        if (resp.ok) {
            alert("Secure Vault successfully initialized!");
            elements.kmsInitPwd.value = '';
            elements.kmsInitPwdConfirm.value = '';
            await checkKmsStatus();
            await loadKmsAuditLogs();
        } else {
            alert("Initialization failed: " + res.detail);
        }
    } catch (err) {
        console.error("Init vault failed:", err);
        alert("Init vault failed");
    }
}

export async function unlockKmsVault() {
    const password = elements.kmsMasterPwd.value;
    if (!password) {
        alert("Please enter your master password.");
        return;
    }
    
    try {
        const resp = await fetch('/api/kms/unlock', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password })
        });
        const res = await resp.json();
        elements.kmsMasterPwd.value = '';
        
        if (resp.ok) {
            await checkKmsStatus();
            await loadKmsAuditLogs();
            
            // Re-fetch system config to update headers trading mode
            await fetchConfig();
        } else {
            alert("Unlock failed: " + res.detail);
        }
    } catch (err) {
        console.error("Unlock vault failed:", err);
        alert("Unlock vault failed");
    }
}

export async function lockKmsVault() {
    try {
        await fetch('/api/kms/lock', { method: 'POST' });
        await checkKmsStatus();
        await loadKmsAuditLogs();
        
        // Re-fetch system config to update headers trading mode
        await fetchConfig();
    } catch (err) {
        console.error("Lock vault failed:", err);
    }
}

export async function listKmsKeys() {
    try {
        const resp = await fetch('/api/kms/keys');
        if (!resp.ok) return;
        const keys = await resp.json();
        
        // Populate the exchange-wallet-select dropdown
        if (elements.exchangeWalletSelect) {
            const currentSelected = elements.exchangeWalletSelect.value;
            elements.exchangeWalletSelect.innerHTML = '<option value="">-- Active Wallet --</option>';
            keys.forEach(k => {
                const opt = document.createElement('option');
                opt.value = k.wallet_address;
                const netLabel = k.testnet ? 'Testnet' : 'Mainnet';
                opt.textContent = `${k.label} (${k.wallet_address.slice(0, 6)}...${k.wallet_address.slice(-4)}) [${netLabel}]`;
                elements.exchangeWalletSelect.appendChild(opt);
            });
            if (currentSelected && Array.from(elements.exchangeWalletSelect.options).some(o => o.value === currentSelected)) {
                elements.exchangeWalletSelect.value = currentSelected;
            }
        }
        
        elements.kmsKeysTbody.innerHTML = '';
        
        // Check if any key is currently active
        const hasActiveKey = keys.some(k => k.status === 'active');
        const localSimulatorActive = !hasActiveKey;
        
        // 1. Render permanent Local Simulator row
        const localTr = document.createElement('tr');
        localTr.className = 'local-simulator-row';
        if (localSimulatorActive) {
            localTr.classList.add('table-active-row');
        }
        
        const localStatusBadge = `<span class="status-badge ${localSimulatorActive ? 'active' : 'inactive'}">${localSimulatorActive ? 'ACTIVE' : 'INACTIVE'}</span>`;
        const localActionButtons = localSimulatorActive ? `
            <span class="text-success font-bold"><i class="fa-solid fa-circle-check"></i> Selected</span>
        ` : `
            <button class="btn btn-sm btn-secondary btn-activate-local">
                <i class="fa-solid fa-play"></i> Activate
            </button>
        `;
        
        localTr.innerHTML = `
            <td class="font-bold text-accent"><i class="fa-solid fa-computer"></i> Local Simulator (本机模拟盘)</td>
            <td class="font-mono text-tertiary">N/A (Local / Simulated)</td>
            <td>Local Sandbox</td>
            <td>${localStatusBadge}</td>
            <td>${localActionButtons}</td>
        `;
        
        if (!localSimulatorActive) {
            const actBtn = localTr.querySelector('.btn-activate-local');
            if (actBtn) {
                actBtn.addEventListener('click', async () => {
                    await deactivateKmsKey();
                });
            }
        }
        elements.kmsKeysTbody.appendChild(localTr);
        
        // 2. Render secured keys
        keys.forEach(k => {
            const tr = document.createElement('tr');
            const isActive = k.status === 'active';
            if (isActive) {
                tr.classList.add('table-active-row');
            }
            const statusBadge = `<span class="status-badge ${isActive ? 'active' : 'inactive'}">${k.status.toUpperCase()}</span>`;
            
            const actionButtons = isActive ? `
                <button class="btn btn-sm btn-warning btn-deactivate-key" data-id="${k.key_id}">
                    <i class="fa-solid fa-stop"></i> Deactivate
                </button>
                <button class="btn btn-sm btn-danger btn-delete-key" data-id="${k.key_id}">
                    <i class="fa-solid fa-trash"></i> Delete
                </button>
            ` : `
                <button class="btn btn-sm btn-secondary btn-activate-key" data-id="${k.key_id}">
                    <i class="fa-solid fa-play"></i> Activate
                </button>
                <button class="btn btn-sm btn-danger btn-delete-key" data-id="${k.key_id}">
                    <i class="fa-solid fa-trash"></i> Delete
                </button>
            `;
            
            tr.innerHTML = `
                <td class="font-bold">${escapeHtml(k.label)}</td>
                <td class="font-mono text-secondary">${escapeHtml(k.wallet_address.slice(0, 10))}...${escapeHtml(k.wallet_address.slice(-8))}</td>
                <td>${k.testnet ? 'Testnet' : 'Mainnet'}</td>
                <td>${statusBadge}</td>
                <td>${actionButtons}</td>
            `;
            
            // Bind actions
            if (isActive) {
                const deactivateBtn = tr.querySelector('.btn-deactivate-key');
                if (deactivateBtn) {
                    deactivateBtn.addEventListener('click', async () => {
                        await deactivateKmsKey();
                    });
                }
            } else {
                const activateBtn = tr.querySelector('.btn-activate-key');
                if (activateBtn) {
                    activateBtn.addEventListener('click', async (e) => {
                        const kid = e.currentTarget.dataset.id;
                        await activateKmsKey(kid);
                    });
                }
            }
            tr.querySelector('.btn-delete-key').addEventListener('click', async (e) => {
                if (confirm("Are you sure you want to delete this credential?")) {
                    const kid = e.currentTarget.dataset.id;
                    await deleteKmsKey(kid);
                }
            });
            
            elements.kmsKeysTbody.appendChild(tr);
        });
    } catch (err) {
        console.error("List keys failed:", err);
    }
}

export async function addKmsKey() {
    const label = elements.kmsNewLabel.value.trim();
    const wallet_address = elements.kmsNewAddress.value.trim();
    const private_key = elements.kmsNewKey.value.trim();
    const testnet = elements.kmsNewTestnet.checked;
    const password = elements.kmsReverifyPwd.value || 'system_default';
    
    if (!label || !wallet_address || !private_key || !password) {
        alert("Please fill in all fields to authorize encryption.");
        return;
    }
    
    try {
        const resp = await fetch('/api/kms/keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label, wallet_address, private_key, testnet, password })
        });
        const res = await resp.json();
        
        elements.kmsReverifyPwd.value = 'system_default';
        
        if (resp.ok) {
            alert("Key encrypted and stored successfully!");
            elements.kmsNewLabel.value = '';
            elements.kmsNewAddress.value = '';
            elements.kmsNewKey.value = '';
            await checkKmsStatus();
            await loadKmsAuditLogs();
        } else {
            alert("Failed to save key: " + res.detail);
        }
    } catch (err) {
        console.error("Add key failed:", err);
        alert("Add key failed");
    }
}

export async function deleteKmsKey(keyId) {
    try {
        const resp = await fetch(`/api/kms/keys/${keyId}`, { method: 'DELETE' });
        if (resp.ok) {
            await checkKmsStatus();
            await loadKmsAuditLogs();
            await fetchConfig();
        } else {
            const res = await resp.json();
            alert("Failed to delete: " + res.detail);
        }
    } catch (err) {
        console.error("Delete key failed:", err);
    }
}

export async function activateKmsKey(keyId) {
    try {
        const resp = await fetch(`/api/kms/keys/${keyId}/activate`, { method: 'POST' });
        if (resp.ok) {
            await checkKmsStatus();
            await loadKmsAuditLogs();
            await fetchConfig();
        } else {
            const res = await resp.json();
            alert("Activation failed: " + res.detail);
        }
    } catch (err) {
        console.error("Activate key failed:", err);
    }
}

export async function deactivateKmsKey() {
    try {
        const resp = await fetch('/api/kms/keys/deactivate', { method: 'POST' });
        if (resp.ok) {
            await checkKmsStatus();
            await loadKmsAuditLogs();
            await fetchConfig();
        } else {
            const res = await resp.json();
            alert("Deactivation failed: " + res.detail);
        }
    } catch (err) {
        console.error("Deactivate key failed:", err);
    }
}

export async function loadKmsAuditLogs() {
    try {
        const resp = await fetch('/api/kms/audit');
        if (!resp.ok) return;
        const logs = await resp.json();
        
        elements.kmsAuditLogs.innerHTML = '';
        if (logs.length === 0) {
            elements.kmsAuditLogs.innerHTML = '<div class="text-tertiary">No security events logged.</div>';
            return;
        }
        
        logs.forEach(log => {
            const dateStr = new Date(log.timestamp * 1000).toLocaleString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
            const row = document.createElement('div');
            row.className = 'audit-log-row';
            row.innerHTML = `
                <span class="audit-log-time">[${dateStr}]</span>
                <span class="audit-log-action">${log.action}</span>
                <span class="audit-log-desc">${escapeHtml(log.details)}</span>
            `;
            elements.kmsAuditLogs.appendChild(row);
        });
    } catch (err) {
        console.error("Load audit logs failed:", err);
    }
}
