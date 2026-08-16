"""
LoRA Adapter Manager for AI Trading OS.

Features:
1. Manages trained LoRA adapter weights, metadata, and versioning.
2. Supports hot-activating / switching active adapters for live trading and backtesting.
3. Persistent JSON manifest storage.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.config import DEFAULT_MODEL, LOGS_DIR

logger = logging.getLogger(__name__)


@dataclass
class AdapterMetadata:
    """Metadata record of a trained LoRA adapter."""
    name: str
    adapter_path: str
    base_model: str = DEFAULT_MODEL
    target_symbol: str = "ALL"
    target_timeframe: str = "1h"
    train_mode: str = "sft"
    is_active: bool = False
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    created_at: float = field(default_factory=time.time)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdapterManager:
    """
    Manager for trained LoRA adapters in AI Trading OS.
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or (LOGS_DIR / "adapters_manifest.json")
        self.adapters: Dict[str, AdapterMetadata] = {}
        self._load_manifest()

    def register_adapter(
        self,
        name: str,
        adapter_path: str,
        base_model: str = DEFAULT_MODEL,
        target_symbol: str = "ALL",
        target_timeframe: str = "1h",
        train_mode: str = "sft",
        description: str = "",
        auto_activate: bool = False,
    ) -> AdapterMetadata:
        """Register a newly trained adapter into the manifest."""
        meta = AdapterMetadata(
            name=name,
            adapter_path=str(adapter_path),
            base_model=base_model,
            target_symbol=target_symbol,
            target_timeframe=target_timeframe,
            train_mode=train_mode,
            description=description,
            is_active=auto_activate,
        )
        if auto_activate:
            for ad in self.adapters.values():
                ad.is_active = False

        self.adapters[name] = meta
        self._save_manifest()
        logger.info("Registered LoRA adapter '%s' at %s", name, adapter_path)
        return meta

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters."""
        return [ad.to_dict() for ad in self.adapters.values()]

    def get_active_adapter(self) -> Optional[AdapterMetadata]:
        """Return the currently activated adapter, if any."""
        for ad in self.adapters.values():
            if ad.is_active:
                return ad
        return None

    def activate_adapter(self, name: str) -> bool:
        """Activate a specific adapter by name."""
        if name not in self.adapters:
            logger.warning("Adapter '%s' not found.", name)
            return False

        for k, ad in self.adapters.items():
            ad.is_active = (k == name)

        self._save_manifest()
        logger.info("Activated LoRA adapter '%s'.", name)
        return True

    def deactivate_adapter(self) -> bool:
        """Deactivate all adapters (revert to base model)."""
        for ad in self.adapters.values():
            ad.is_active = False
        self._save_manifest()
        logger.info("Deactivated all LoRA adapters.")
        return True

    def delete_adapter(self, name: str) -> bool:
        """Remove an adapter from the registry."""
        if name in self.adapters:
            del self.adapters[name]
            self._save_manifest()
            return True
        return False

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self.adapters.items()}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.adapters = {k: AdapterMetadata(**v) for k, v in data.items()}
        except Exception as e:
            logger.warning("Failed to load adapter manifest: %s", e)
