"""Inversion and reconstruction attack models.

Names are resolved lazily (PEP 562) so numpy-only attack modules work in
environments without torch, such as the experiment container.
"""
from __future__ import annotations

import importlib

_LAZY = {
    "invert_linear_template": "ijiss_btp.attacks.analytical",
    "evaluate_proxy_match_rate": "ijiss_btp.attacks.analytical",
    "TemplateInverterMLP": "ijiss_btp.attacks.neural_inversion",
    "train_neural_inverter": "ijiss_btp.attacks.neural_inversion",
    "coordinate_wise_inversion_attack": "ijiss_btp.attacks.coordinate_wise",
    "ps_iom_known_key_inversion": "ijiss_btp.attacks.coordinate_wise",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'ijiss_btp.attacks' has no attribute {name!r}")
