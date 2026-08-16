"""Unit tests for MLX K-Line Dataset Builder, Trainer, and Adapter Manager."""

import json
import os
import tempfile
import pytest

from xmlx_vlm.ai_trader.training.kline_dataset_builder import DatasetBuildConfig, KlineDatasetBuilder
from xmlx_vlm.ai_trader.training.mlx_kline_trainer import KlineTrainingConfig, MLXKlineTrainer, train_kline_sft
from xmlx_vlm.ai_trader.training.adapter_manager import AdapterManager


class TestKlineTrainingPipeline:
    """Test suite for K-Line dataset construction and MLX fine-tuning pipeline."""

    def _generate_mock_bars(self, count: int = 60) -> list:
        bars = []
        base_price = 50000.0
        for i in range(count):
            p = base_price + i * 50.0
            bars.append({
                "symbol": "BTC",
                "timeframe": "1h",
                "timestamp_ms": 1700000000000 + i * 3600000,
                "open": p - 10.0,
                "high": p + 40.0,
                "low": p - 20.0,
                "close": p,
                "volume": 100.0 + i * 2.0,
                "buy_volume": 60.0 + i,
                "sell_volume": 40.0 + i,
            })
        return bars

    def test_kline_dataset_builder_sft_and_orpo(self):
        bars = self._generate_mock_bars(count=50)
        builder = KlineDatasetBuilder(config=DatasetBuildConfig(window_size=15, forward_window=5))

        # 1. SFT Samples
        sft_samples = builder.build_sft_samples(bars, symbol="BTC/USDT", timeframe="1h")
        assert len(sft_samples) > 0
        first_sample = sft_samples[0]
        assert "messages" in first_sample
        assert len(first_sample["messages"]) == 2
        assert first_sample["messages"][0]["role"] == "user"
        assert first_sample["messages"][1]["role"] == "assistant"
        assert "Quantitative Market Analysis" in first_sample["messages"][1]["content"]

        # 2. ORPO Samples
        orpo_samples = builder.build_orpo_samples(bars, symbol="BTC/USDT", timeframe="1h")
        assert len(orpo_samples) == len(sft_samples)
        first_orpo = orpo_samples[0]
        assert "prompt" in first_orpo
        assert "chosen" in first_orpo
        assert "rejected" in first_orpo

    def test_dataset_jsonl_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "train_sft.jsonl")
            bars = self._generate_mock_bars(count=40)
            builder = KlineDatasetBuilder(config=DatasetBuildConfig(window_size=10, forward_window=5))
            samples = builder.build_sft_samples(bars)

            builder.save_dataset_jsonl(samples, out_file)
            assert os.path.exists(out_file)

            with open(out_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == len(samples)
                parsed = json.loads(lines[0])
                assert "messages" in parsed

    def test_mlx_kline_trainer_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_dataset = os.path.join(tmpdir, "train.jsonl")
            with open(dummy_dataset, "w") as f:
                f.write(json.dumps({"messages": []}) + "\n")

            out_dir = os.path.join(tmpdir, "output_adapter")
            res = train_kline_sft(
                dataset_path=dummy_dataset,
                output_dir=out_dir,
                iters=10,
                batch_size=1,
            )

            assert res["status"] == "completed"
            assert os.path.exists(os.path.join(out_dir, "adapter_config.json"))
            assert os.path.exists(os.path.join(out_dir, "training_summary.json"))

    def test_adapter_manager_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            manifest_file = Path(tmpdir) / "adapters.json"
            mgr = AdapterManager(manifest_path=manifest_file)

            # 1. Register adapter
            meta = mgr.register_adapter(
                name="kline_cot_v1",
                adapter_path=f"{tmpdir}/kline_cot_v1",
                base_model="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
                target_symbol="BTC",
                auto_activate=True,
            )
            assert meta.name == "kline_cot_v1"
            assert meta.is_active is True

            # 2. Get active
            active = mgr.get_active_adapter()
            assert active is not None
            assert active.name == "kline_cot_v1"

            # 3. Register second and switch
            mgr.register_adapter(
                name="breakout_v2",
                adapter_path=f"{tmpdir}/breakout_v2",
                auto_activate=False,
            )
            assert len(mgr.list_adapters()) == 2

            mgr.activate_adapter("breakout_v2")
            assert mgr.get_active_adapter().name == "breakout_v2"

            # 4. Deactivate
            mgr.deactivate_adapter()
            assert mgr.get_active_adapter() is None
