# -*- coding: utf-8 -*-
"""NumPy 向量化蒙特卡洛模拟器(百万次级).

与 src/damage_model.py 的解析模型互验(报告 3.3 节方法:
解析解 × 蒙特卡洛互验, 误差 <0.01% 视为模型自洽)。

判定顺序(报告 3.2 节): 先会意 → 精准(未中擦伤) → 会心。
挤压规则: 会心+会意 ≥ 100% 时, 会意未中且精准命中的攻击强制会心(白字清零)。
"""

from __future__ import annotations

import numpy as np

from .damage_model import (
    B0, B1, HUIYI_MULT, HUIXIN_MULT, Panel, judgment_probs,
)

# 全局默认种子(所有实验固定种子以保证可复现)
DEFAULT_SEED = 20261227


def simulate_damage(panel: Panel, n: int, seed: int = DEFAULT_SEED,
                    b0: float = B0, b1: float = B1) -> np.ndarray:
    """向量化模拟 n 次攻击, 返回每次伤害 ndarray.

    判定按概率类别直接分层抽样(与顺序判定等价):
        会意 -> 1.35 × B1(固定最大攻击)
        擦伤 -> B0(固定最小攻击)
        会心 -> 1.5 × B(t), t~U(0,1)
        白字 -> B(t)
    类别概率由 judgment_probs 给出(已实现挤压规则)。
    """
    rng = np.random.default_rng(seed)
    p_yi, p_xin, p_ca, p_bai = judgment_probs(panel)

    t = rng.random(n)                      # 攻击区间取值比例
    b = b0 + (b1 - b0) * t                 # 随机攻击基础伤害
    u = rng.random(n)                      # 判定抽签
    dmg = np.empty(n, dtype=np.float64)

    m_yi = u < p_yi
    m_ca = (~m_yi) & (u < p_yi + p_ca)
    m_xin = (~m_yi) & (~m_ca) & (u < p_yi + p_ca + p_xin)
    m_bai = ~(m_yi | m_ca | m_xin)

    dmg[m_yi] = HUIYI_MULT * b1
    dmg[m_ca] = b0
    dmg[m_xin] = HUIXIN_MULT * b[m_xin]
    dmg[m_bai] = b[m_bai]
    return dmg


def outcome_proportions(panel: Panel, n: int,
                        seed: int = DEFAULT_SEED + 1) -> dict[str, float]:
    """模拟 n 次攻击, 返回四象限判定占比 {'会意','会心','擦伤','白字'}."""
    rng = np.random.default_rng(seed)
    p_yi, p_xin, p_ca, p_bai = judgment_probs(panel)
    u = rng.random(n)
    return {
        "会意": float(np.mean(u < p_yi)),
        "擦伤": float(np.mean((u >= p_yi) & (u < p_yi + p_ca))),
        "会心": float(np.mean((u >= p_yi + p_ca) & (u < p_yi + p_ca + p_xin))),
        "白字": float(np.mean(u >= p_yi + p_ca + p_xin)),
    }


def simulate_total_damage(panel: Panel, hits: int, n_rounds: int,
                          seed: int = DEFAULT_SEED + 2,
                          b0: float = B0, b1: float = B1) -> np.ndarray:
    """模拟 n_rounds 个 '60秒总伤'(每个由 hits 次攻击构成), 返回总伤数组."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_rounds, dtype=np.float64)
    for i in range(n_rounds):
        # 每次 60 秒窗口使用独立子种子, 保证可复现且窗口间独立
        out[i] = simulate_damage(panel, hits, seed=int(rng.integers(0, 2**31 - 1)),
                                 b0=b0, b1=b1).sum()
    return out


def coefficient_of_variation(x: np.ndarray) -> float:
    """变异系数 CV = std/mean(ddof=1)."""
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1) / x.mean())
