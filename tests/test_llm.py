"""llm.py 基础单元测试

运行: pytest tests/ 或 python -m pytest tests/
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm import ModelConfig, GPTLanguageModel, FeedForward, Block


class TestModelConfig:
    """ModelConfig 配置验证"""

    def test_default_creation(self):
        cfg = ModelConfig(vocab_size=65)
        assert cfg.head_size == 32
        assert cfg.d_ff == 512

    def test_validation_nembd_divisible(self):
        with pytest.raises(ValueError):
            ModelConfig(n_embd=100, n_head=3)

    def test_validation_gqa(self):
        with pytest.raises(ValueError):
            ModelConfig(n_head=4, n_kv_head=3)

    def test_gqa_head_size(self):
        cfg = ModelConfig(n_embd=128, n_head=4, n_kv_head=2, vocab_size=65)
        assert cfg.head_size == 32

    def test_estimate_params_positive(self):
        cfg = ModelConfig(vocab_size=65)
        assert cfg.estimate_params() > 0

    def test_to_dict(self):
        cfg = ModelConfig(vocab_size=65)
        d = cfg.to_dict()
        assert 'n_layer' in d
        assert 'vocab_size' in d

    def test_summary(self):
        cfg = ModelConfig(vocab_size=65)
        s = cfg.summary()
        assert 'n_layer' in s


class TestGPTLanguageModel:
    """GPTLanguageModel 核心测试"""

    @pytest.fixture
    def cfg(self):
        return ModelConfig(
            vocab_size=65,
            n_layer=2,
            n_embd=64,
            n_head=4,
            block_size=32,
            batch_size=4,
        )

    @pytest.fixture
    def model(self, cfg):
        return GPTLanguageModel(cfg)

    def test_forward_shape(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_forward_with_targets(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x, y)
        assert loss is not None
        assert loss.item() > 0

    def test_generate_shape(self, model, cfg):
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        model.eval()
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=10)
        assert out.shape == (1, 14)

    def test_count_params(self, model):
        assert model.count_params() > 0

    def test_gqa_forward(self):
        cfg = ModelConfig(
            vocab_size=65, n_layer=2, n_embd=64,
            n_head=4, n_kv_head=2, block_size=32
        )
        model = GPTLanguageModel(cfg)
        x = torch.randint(0, 65, (2, 8))
        logits, loss = model(x)
        assert logits.shape[-1] == 65

    def test_different_configs_independent(self):
        """验证两个不同配置的模型互不干扰（消融前提）"""
        cfg_a = ModelConfig(vocab_size=65, n_layer=2, n_embd=32, n_head=2)
        cfg_b = ModelConfig(vocab_size=65, n_layer=4, n_embd=64, n_head=4)
        model_a = GPTLanguageModel(cfg_a)
        model_b = GPTLanguageModel(cfg_b)

        assert model_a.count_params() != model_b.count_params()
        # 前向独立
        x = torch.randint(0, 65, (1, 8))
        out_a, _ = model_a(x)
        out_b, _ = model_b(x)
        assert out_a.shape != out_b.shape


class TestFeedForward:
    """FeedForward 修复测试"""

    def test_shape(self):
        cfg = ModelConfig(n_embd=64)
        ff = FeedForward(cfg)
        x = torch.randn(2, 8, 64)
        out = ff(x)
        assert out.shape == (2, 8, 64)

    def test_scaled_init(self):
        """验证残差投影层被 scaled init"""
        cfg = ModelConfig(n_embd=64)
        ff = FeedForward(cfg)
        # 投影层权重应接近 0
        proj_weight = ff.net[-2].weight
        assert proj_weight.abs().max().item() < 0.1
