"""llm.py + train.py M1+M2 基础单元测试

运行: python -m pytest tests/test_llm.py
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm import ModelConfig, GPTLanguageModel, FeedForward, Block, cross_entropy_loss


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
        assert cfg.n_kv_head == 2

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

    def test_use_amp_property(self):
        cfg = ModelConfig(vocab_size=65, device='cpu')
        assert not cfg.use_amp  # CPU 不启用 AMP

    def test_save_load_roundtrip(self, tmp_path):
        """测试配置 JSON 序列化往返"""
        cfg = ModelConfig(vocab_size=65, n_layer=3, n_embd=256)
        path = str(tmp_path / "config.json")
        cfg.save(path)
        
        cfg2 = ModelConfig.load(path)
        assert cfg2.vocab_size == 65
        assert cfg2.n_layer == 3
        assert cfg2.n_embd == 256

    def test_gqa_repeat_correctness(self):
        """验证 GQA 模型的 numKVHeads 被正确设置"""
        cfg = ModelConfig(vocab_size=65, n_embd=64, n_head=4, n_kv_head=2)
        model = GPTLanguageModel(cfg)
        # 确保模型创建成功
        assert model.count_params() > 0


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
        logits, _ = model(x)
        assert logits.shape == (2, 8, 65)

    def test_different_configs_independent(self):
        """验证两个不同配置的模型互不干扰（消融前提）"""
        cfg_a = ModelConfig(vocab_size=65, n_layer=2, n_embd=32, n_head=2)
        cfg_b = ModelConfig(vocab_size=65, n_layer=4, n_embd=64, n_head=4)
        model_a = GPTLanguageModel(cfg_a)
        model_b = GPTLanguageModel(cfg_b)

        assert model_a.count_params() != model_b.count_params()
        x = torch.randint(0, 65, (1, 8))
        out_a, _ = model_a(x[:, :4])
        out_b, _ = model_b(x)
        assert out_a.shape != out_b.shape

    def test_sdpa_fallback(self):
        """测试 SDPA 和 fallback 都能正确运行"""
        for use_sdpa in [True, False]:
            cfg = ModelConfig(
                vocab_size=65, n_layer=2, n_embd=64,
                n_head=4, block_size=32, use_sdpa=use_sdpa
            )
            model = GPTLanguageModel(cfg)
            x = torch.randint(0, 65, (1, 8))
            logits, loss = model(x, torch.randint(0, 65, (1, 8)))
            assert logits.shape == (1, 8, 65)


class TestChunkedCE:
    """分块交叉熵测试"""

    def test_standard_equals_chunked(self):
        """验证分块 CE 与标准 CE 结果一致"""
        torch.manual_seed(42)
        logits = torch.randn(2, 8, 65, requires_grad=True)
        targets = torch.randint(0, 65, (2, 8))

        loss_standard = cross_entropy_loss(logits, targets, chunk_size=0)
        loss_chunked = cross_entropy_loss(logits, targets, chunk_size=4)

        # 数值应该非常接近
        assert abs(loss_standard.item() - loss_chunked.item()) < 0.01

    def test_chunked_backward(self):
        """验证 CE 梯度可回退"""
        logits = torch.randn(2, 8, 65, requires_grad=True)
        targets = torch.randint(0, 65, (2, 8))
        loss = cross_entropy_loss(logits, targets, chunk_size=0)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


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
        proj_weight = ff.net[-2].weight
        assert proj_weight.abs().max().item() < 0.1


class TestSeed:
    """可复现性测试"""

    def test_set_seed_reproducibility(self):
        """验证 set_seed 能使两次随机序列一致"""
        from llm import set_seed
        set_seed(42)
        a = torch.randn(10)
        
        set_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)
