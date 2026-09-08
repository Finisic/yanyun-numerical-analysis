# -*- coding: utf-8 -*-
"""调律保底机制测试: 硬上限、均值单调性、锚点复现."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pity_system import (  # noqa: E402
    BIANYIN_CAP, MECHANISMS, N_ITEMS, hard_caps, simulate_players,
)

SEED = 20261227
N_TEST_PLAYERS = 3000   # 测试用较小样本, 控制运行时间


@pytest.fixture(scope="module")
def results():
    """三机制×两目标的模拟结果(模块级缓存, 避免重复模拟)."""
    return {(g, m): simulate_players(g, m, N_TEST_PLAYERS, N_ITEMS, seed=SEED)
            for g in ("small", "perfect") for m in MECHANISMS}


class TestHardCap:
    """变音硬保底性质."""

    def test_perfect_bianyin_hard_cap_48(self, results):
        """动态+变音完美毕业: 全身硬上限 ≤ 6×8=48 件(报告 4.3 节)."""
        r = results[("perfect", "pity_bianyin")]
        assert r["costs"].max() <= hard_caps("perfect") == 48
        assert r["max"] <= BIANYIN_CAP * N_ITEMS

    def test_pity_without_bianyin_no_hard_cap(self, results):
        """无变音时完美毕业无 48 硬上限(右尾存在, 大概率超过 48)."""
        r = results[("perfect", "pity")]
        assert r["max"] > hard_caps("perfect")


class TestMonotonicity:
    """保底机制的均值/尾部单调性: 纯随机 ≥ 动态保底 ≥ 动态+变音."""

    @pytest.mark.parametrize("goal", ["small", "perfect"])
    def test_mean_monotonic(self, goal, results):
        m_pure = results[(goal, "pure")]["mean"]
        m_pity = results[(goal, "pity")]["mean"]
        m_bian = results[(goal, "pity_bianyin")]["mean"]
        assert m_pure >= m_pity >= m_bian

    @pytest.mark.parametrize("goal", ["small", "perfect"])
    def test_p99_monotonic(self, goal, results):
        p_pure = results[(goal, "pure")]["p99"]
        p_pity = results[(goal, "pity")]["p99"]
        p_bian = results[(goal, "pity_bianyin")]["p99"]
        assert p_pure >= p_pity >= p_bian

    def test_pity_cuts_right_tail_more_than_mean(self, results):
        """报告核心结论: 动态保底砍右尾幅度 > 降均值幅度(小毕业)."""
        m_pure, m_pity = results[("small", "pure")]["mean"], results[("small", "pity")]["mean"]
        p_pure, p_pity = results[("small", "pure")]["p99"], results[("small", "pity")]["p99"]
        mean_cut = 1 - m_pity / m_pure
        tail_cut = 1 - p_pity / p_pure
        assert tail_cut > mean_cut


class TestAnchorReproduction:
    """报告锚点6复现(小样本, 容差放宽)."""

    def test_pure_small_mean(self, results):
        """纯随机小毕业均值 ≈ 3/0.3×8 = 80 件."""
        assert results[("small", "pure")]["mean"] == pytest.approx(80, rel=0.06)

    def test_pure_perfect_mean(self, results):
        """纯随机完美毕业均值 ≈ 8/(0.3/8) = 213 件."""
        assert results[("perfect", "pure")]["mean"] == pytest.approx(213, rel=0.06)

    def test_pity_bianyin_perfect_mean(self, results):
        """动态+变音完美毕业均值 ≈43 件(硬上限确定性化)."""
        assert results[("perfect", "pity_bianyin")]["mean"] == pytest.approx(43, rel=0.08)

    def test_item_cost_lower_bounds(self, results):
        """成本下界: 小毕业每件至少 3 次(全身≥24), 完美毕业每件至少 1 次(全身≥8)."""
        assert results[("small", "pure")]["costs"].min() >= 3 * N_ITEMS
        assert results[("perfect", "pure")]["costs"].min() >= 1 * N_ITEMS
