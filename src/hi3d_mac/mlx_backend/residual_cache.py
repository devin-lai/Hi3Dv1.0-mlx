"""Opt-in, bounded reuse of the transformer tail during late denoising.

This is an approximation, separate from the FP16 scheduling/kernel optimizations.
Block zero always runs. Only unguided evaluations below sigma 0.7 can reuse the
previous full evaluation's remaining-block residual, once consecutively at most,
and only if both the first-block output and adaLN modulation change by less than
the configured relative-L1 threshold. State is reset for every sampling pass.
"""
from __future__ import annotations

import mlx.core as mx

from .transformer import Sparc3DTransformerMLX


class CachedTransformer(Sparc3DTransformerMLX):
    def __init__(self, *args, threshold: float, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < threshold < 1:
            raise ValueError("residual cache threshold must be between zero and one")
        self.threshold = threshold
        self.reset_cache()

    def reset_cache(self):
        self._anchor = self._mod_anchor = self._residual = self._first = None
        self._reuse = self._active = False
        self.cache_events = []
        self.cache_calls = 0

    def __call__(self, feats, t, geo, kvs):
        self._active = t < 0.7 and len(kvs) == 1 and kvs[0] is not None
        self._sigma = float(t)
        self.cache_calls += 1
        if not self._active:
            self._anchor = self._mod_anchor = self._residual = self._first = None
            self._reuse = False
        return super().__call__(feats, t, geo, kvs)

    def run_block(self, i, x, mod, *args, **kwargs):
        if not self._active:
            return super().run_block(i, x, mod, *args, **kwargs)
        if i == 0:
            y = super().run_block(i, x, mod, *args, **kwargs)
            valid = self._anchor is not None and self._anchor.shape == y.shape
            change = modulation_change = None
            if valid:
                a = self._anchor.astype(mx.float32)
                m = self._mod_anchor.astype(mx.float32)
                changes = mx.stack([
                    mx.mean(mx.abs(y.astype(mx.float32) - a)) / mx.maximum(mx.mean(mx.abs(a)), 1e-8),
                    mx.mean(mx.abs(mod.astype(mx.float32) - m)) / mx.maximum(mx.mean(mx.abs(m)), 1e-8),
                ])
                change, modulation_change = changes.tolist()
            reuse = valid and not self._reuse and max(change, modulation_change) < self.threshold
            self.cache_events.append({"evaluation": self.cache_calls - 1, "sigma": self._sigma,
                                      "first_block_change": change, "modulation_change": modulation_change,
                                      "reuse": reuse})
            self._reuse = reuse
            self._first = y
            if not reuse:
                self._anchor, self._mod_anchor = mx.array(y), mx.array(mod)
            return y
        if self._reuse:
            return (self._first.astype(mx.float32) + self._residual).astype(self.dt) if i == self.num_blocks - 1 else x
        y = super().run_block(i, x, mod, *args, **kwargs)
        if i == self.num_blocks - 1:
            self._residual = y.astype(mx.float32) - self._first.astype(mx.float32)
            mx.eval(y, self._residual, self._anchor, self._mod_anchor)
        return y
