"""KMS / Secure Vault cryptographic module for AI Trader.

Uses Python standard library hashlib and hmac to perform secure PBKDF2-HMAC-SHA256
key derivation and HMAC-SHA256-CTR encryption/decryption, avoiding compiled dependencies.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

# Global in-memory cache of unlocked credentials
# Keyed by wallet_address: {"wallet_address": str, "private_key": str, "testnet": bool, "label": str}
_UNLOCKED_CREDENTIALS: Dict[str, Dict[str, Any]] = {}
_ACTIVE_WALLET_ADDRESS: Optional[str] = None
_IS_UNLOCKED: bool = False


def xor_bytes(b1: bytes, b2: bytes) -> bytes:
    """XOR two byte sequences of the same length."""
    return bytes(x ^ y for x, y in zip(b1, b2))


def derive_key(password: str, salt: bytes, iterations: int = 100000) -> bytes:
    """Derive a 32-byte key from a password and salt using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, 32)


def encrypt_data(data: str, password: str) -> Dict[str, str]:
    """Encrypt a plaintext string using HMAC-SHA256-CTR with derived key.
    
    Returns a dictionary containing base64 encoded strings:
    - ciphertext
    - salt
    - iv
    - mac
    """
    salt = os.urandom(16)
    iv = os.urandom(16)
    
    # Derive encryption key
    enc_key = derive_key(password, salt)
    
    data_bytes = data.encode("utf-8")
    length = len(data_bytes)
    
    # Generate keystream using HMAC-SHA256 in Counter Mode
    keystream = b""
    block_index = 0
    while len(keystream) < length:
        # Counter block: IV + 4-byte big-endian block index
        counter_block = iv + block_index.to_bytes(4, "big")
        h = hmac.new(enc_key, counter_block, hashlib.sha256)
        keystream += h.digest()
        block_index += 1
        
    ciphertext = xor_bytes(data_bytes, keystream[:length])
    
    # Compute MAC to verify integrity (Encrypt-then-MAC)
    # MAC key is derived from encryption key to keep it distinct
    mac_key = hashlib.sha256(enc_key + b"_mac").digest()
    mac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    
    return {
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
        "salt": base64.b64encode(salt).decode("utf-8"),
        "iv": base64.b64encode(iv).decode("utf-8"),
        "mac": base64.b64encode(mac).decode("utf-8")
    }


def decrypt_data(enc_dict: Dict[str, str], password: str) -> str:
    """Decrypt a base64 encrypted payload. Throws ValueError if decryption or MAC verification fails."""
    try:
        ciphertext = base64.b64decode(enc_dict["ciphertext"])
        salt = base64.b64decode(enc_dict["salt"])
        iv = base64.b64decode(enc_dict["iv"])
        mac = base64.b64decode(enc_dict["mac"])
    except Exception as e:
        raise ValueError(f"Invalid base64 payload: {e}")
        
    # Derive key
    enc_key = derive_key(password, salt)
    
    # Verify MAC first
    mac_key = hashlib.sha256(enc_key + b"_mac").digest()
    expected_mac = hmac.new(mac_key, iv + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected_mac):
        raise ValueError("Decryption failed: MAC verification failed (incorrect password or corrupted data)")
        
    # Decrypt
    length = len(ciphertext)
    keystream = b""
    block_index = 0
    while len(keystream) < length:
        counter_block = iv + block_index.to_bytes(4, "big")
        h = hmac.new(enc_key, counter_block, hashlib.sha256)
        keystream += h.digest()
        block_index += 1
        
    plaintext_bytes = xor_bytes(ciphertext, keystream[:length])
    return plaintext_bytes.decode("utf-8")


def set_unlocked_credentials(creds: Dict[str, Dict[str, Any]]):
    """Store unlocked credentials in-memory."""
    global _UNLOCKED_CREDENTIALS, _IS_UNLOCKED
    _UNLOCKED_CREDENTIALS = creds
    _IS_UNLOCKED = True


def get_unlocked_credentials() -> Dict[str, Dict[str, Any]]:
    """Retrieve in-memory unlocked credentials."""
    return _UNLOCKED_CREDENTIALS


def set_active_wallet(wallet_address: Optional[str]):
    """Set the active wallet address in memory."""
    global _ACTIVE_WALLET_ADDRESS
    _ACTIVE_WALLET_ADDRESS = wallet_address


def get_active_wallet() -> Optional[str]:
    """Get the active wallet address."""
    return _ACTIVE_WALLET_ADDRESS


def get_active_credential() -> Optional[Dict[str, Any]]:
    """Get the currently active credential info if unlocked and set."""
    if not _ACTIVE_WALLET_ADDRESS:
        return None
    return _UNLOCKED_CREDENTIALS.get(_ACTIVE_WALLET_ADDRESS)


def wipe_vault():
    """Wipe secure memory."""
    global _UNLOCKED_CREDENTIALS, _ACTIVE_WALLET_ADDRESS, _IS_UNLOCKED
    _UNLOCKED_CREDENTIALS.clear()
    _ACTIVE_WALLET_ADDRESS = None
    _IS_UNLOCKED = False


def is_unlocked() -> bool:
    """Check if the vault is currently unlocked."""
    return _IS_UNLOCKED
