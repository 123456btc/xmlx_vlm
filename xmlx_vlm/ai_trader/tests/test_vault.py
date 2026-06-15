import os
import json
import pytest
from pathlib import Path
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.store import vault


def test_encryption_decryption_roundtrip():
    password = "SuperSecretMasterPassword"
    plaintext = "0x819cf39943b1236187a2d48c0818bf5d398935c1029abcb3f28cf081bfec3a2b"
    
    # 1. Encrypt data
    enc_dict = vault.encrypt_data(plaintext, password)
    assert "ciphertext" in enc_dict
    assert "salt" in enc_dict
    assert "iv" in enc_dict
    assert "mac" in enc_dict
    
    # 2. Decrypt with correct password
    decrypted = vault.decrypt_data(enc_dict, password)
    assert decrypted == plaintext
    
    # 3. Decrypt with incorrect password should raise ValueError due to MAC mismatch
    with pytest.raises(ValueError) as excinfo:
        vault.decrypt_data(enc_dict, "WrongPassword")
    assert "decryption failed" in str(excinfo.value).lower() or "mac verification failed" in str(excinfo.value).lower()


def test_database_kms_vault_storage(tmp_path):
    # Setup temporary SQLite database path
    db_file = tmp_path / "test_trader_sessions.db"
    db = QuantSessionDB(db_path=db_file)
    
    # 1. Initialize vault in DB
    salt_hex = "abcdef1234567890"
    verifier_hex = "deadbeef0987654321"
    db.init_kms_vault(salt_hex, verifier_hex)
    
    assert db.get_kms_config("vault_initialized") == "true"
    assert db.get_kms_config("vault_salt") == salt_hex
    assert db.get_kms_config("vault_verifier") == verifier_hex
    
    # 2. Add keys to the database
    key_id = "test-key-uuid-1"
    label = "Test Key 1"
    wallet = "0x1234567890abcdef1234567890abcdef12345678"
    enc_payload = '{"ciphertext": "xyz", "salt": "s", "iv": "i", "mac": "m"}'
    
    db.add_kms_key(key_id, label, wallet, enc_payload, testnet=True)
    
    # Verify key is listed (masked)
    keys_list = db.list_kms_keys()
    assert len(keys_list) == 1
    assert keys_list[0]["key_id"] == key_id
    assert keys_list[0]["label"] == label
    assert keys_list[0]["wallet_address"] == wallet
    assert keys_list[0]["testnet"] == 1
    assert keys_list[0]["status"] == "active"
    assert "encrypted_private_key" not in keys_list[0] # Verify payload is not leaked in list view
    
    # Verify we can fetch the full encrypted payload
    full_key = db.get_encrypted_kms_key(key_id)
    assert full_key is not None
    assert full_key["encrypted_private_key"] == enc_payload
    
    # 3. Activate key
    activated = db.activate_kms_key(key_id)
    assert activated["status"] == "active"
    
    keys_list = db.list_kms_keys()
    assert keys_list[0]["status"] == "active"
    
    # Add a second key
    key_id_2 = "test-key-uuid-2"
    db.add_kms_key(key_id_2, "Test Key 2", wallet, enc_payload, testnet=False)
    
    # Activate second key, should set first to inactive
    db.activate_kms_key(key_id_2)
    
    assert db.get_encrypted_kms_key(key_id)["status"] == "inactive"
    assert db.get_encrypted_kms_key(key_id_2)["status"] == "active"
    
    # Deactivate all
    db.deactivate_all_kms_keys()
    assert db.get_encrypted_kms_key(key_id_2)["status"] == "inactive"
    
    # 4. Audit Logging
    db.log_kms_audit("TEST_ACTION", "Testing vault logs")
    audit_logs = db.get_kms_audit_logs()
    assert len(audit_logs) == 1
    assert audit_logs[0]["action"] == "TEST_ACTION"
    assert audit_logs[0]["details"] == "Testing vault logs"
    
    # 5. Delete key
    db.delete_kms_key(key_id)
    assert len(db.list_kms_keys()) == 1
    assert db.get_encrypted_kms_key(key_id) is None
