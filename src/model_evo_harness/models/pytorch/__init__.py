"""PyTorch reference implementations; output contracts vary by model."""

from .architectures import DCN, DIN, FM, MMoE, DeepFM, TwoTower

__all__ = ["FM", "DeepFM", "DCN", "DIN", "MMoE", "TwoTower"]
