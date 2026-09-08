# -*- coding: utf-8 -*-
"""伤害模型测试: 解析解 vs 蒙特卡洛互验、判定概率归一性、锚点复现."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.damage_model import (  # noqa: E402
    B0, B1, PANEL_MID, PANEL_VALIDATION, Panel, expected_damage, judgment_probs,
)
from src.monte_carlo import outcome_proportions, simulate_damage  # noqa: E402

SEED = 20261227


class TestJudgmentProbs:
    """判定概率归一性与挤压规则."""

    @pytest.mark.parametrize("pj", [0.0, 0.5, 0.9, 0.95, 1.0])
    @pytest.mark.parametrize("px", [0.0, 0.4, 0.6, 0.8, 1.0])
    @pytest.mark.parametrize("py", [0.0, 0.1, 0.15, 0.3, 0.5])
    def test_probs_normalized(self, pj, px, py):
        """任意面板下四种判定概率和为 1."""
        probs = judgment_probs(Panel(pj, px, py))
        assert sum(probs) == pytest.approx(1.0, abs=1e-12)
        assert all(p >= -1e-12 for p in probs)

    def test_squeeze_zone_white_vanish(self):
        """挤压线上(会心+会意>=100%): 白字清零, 强制会心(报告 3.2 节)."""
        p_yi, p_xin, p_ca, p_bai = judgment_probs(Panel(1.00, 0.70, 0.30))
        assert p_bai == 0.0
        assert p_xin == pytest.approx((1 - 0.30) * 1.00)

    def test_below_squeeze_zone(self):
        """挤压线下: p心=(1-p易)·p精·p心率(报告 3.2 节)."""
        p_yi, p_xin, p_ca, p_bai = judgment_probs(PANEL_VALIDATION)
        assert p_yi == pytest.approx(0.15)
        assert p_xin == pytest.approx(0.85 * 0.95 * 0.60)   # 48.45%
        assert p_ca == pytest.approx(0.85 * 0.05)           # 4.25%
        assert p_bai == pytest.approx(0.85 * 0.95 * 0.40)   # 32.3%


class TestAnalyticVsMonteCarlo:
    """解析解 vs 蒙特卡洛互验(报告 3.3 节方法)."""

    def test_validation_panel_error_below_0p1pct(self):
        """验证面板(95/60/15): 200万次模拟与解析期望误差 <0.1%."""
        e_ana = expected_damage(PANEL_VALIDATION)
        dmg = simulate_damage(PANEL_VALIDATION, 2_000_000, seed=SEED)
        assert abs(dmg.mean() - e_ana) / e_ana < 1e-3

    def test_full_coverage_panel_error_below_0p1pct(self):
        """满覆盖面板(100/80/20, 挤压区)同样互验通过."""
        panel = Panel(1.00, 0.80, 0.20)
        e_ana = expected_damage(panel)
        dmg = simulate_damage(panel, 2_000_000, seed=SEED + 1)
        assert abs(dmg.mean() - e_ana) / e_ana < 1e-3

    def test_outcome_proportions(self):
        """判定占比与解析概率一致(容差 0.2pp)."""
        props = outcome_proportions(PANEL_VALIDATION, 1_000_000, seed=SEED + 2)
        p_ana = judgment_probs(PANEL_VALIDATION)
        for name, p in zip(("会意", "会心", "擦伤", "白字"), p_ana):
            assert props[name] == pytest.approx(p, abs=2e-3)


class TestAnchorNumbers:
    """报告锚点数字复现(解析解侧)."""

    def test_anchor1_analytic_expectation(self):
        """锚点1: 验证面板解析期望 ≈15909.9(报告 3.3 节)."""
        assert expected_damage(PANEL_VALIDATION) == pytest.approx(15909.9, abs=0.5)

    def test_anchor2_marginal_rates(self):
        """锚点2: 中期面板边际收益 会意+2.07% > 会心+1.68% > 精准+1.59%."""
        e0 = expected_damage(PANEL_MID)
        g_yi = expected_damage(Panel(0.90, 0.40, 0.15)) / e0 - 1
        g_xin = expected_damage(Panel(0.90, 0.45, 0.10)) / e0 - 1
        g_jing = expected_damage(Panel(0.95, 0.40, 0.10)) / e0 - 1
        assert g_yi * 100 == pytest.approx(2.07, abs=0.01)
        assert g_xin * 100 == pytest.approx(1.68, abs=0.01)
        assert g_jing * 100 == pytest.approx(1.59, abs=0.01)
        assert g_yi > g_xin > g_jing

    def test_anchor3_full_coverage_jump(self):
        """锚点3: 固定会心70%, 凑满覆盖(会意25%→30%)期望跃升约 +8%."""
        e_before = expected_damage(Panel(1.00, 0.70, 0.25))
        e_after = expected_damage(Panel(1.00, 0.70, 0.30))
        jump = e_after / e_before - 1
        assert 0.079 < jump < 0.085     # 报告口径 +7.9%~8.4%

    def test_anchor3_squeeze_slope_drops(self):
        """锚点3: 过线后会意边际收益显著下降(方向性结论)."""
        es = [expected_damage(Panel(1.00, 0.70, y / 100)) for y in range(0, 51)]
        slopes = np.diff(es)
        assert slopes[30] < slopes[10]            # 线上斜率低于线下
        assert slopes[30] == pytest.approx(18.8, abs=0.3)  # 线上斜率复现
