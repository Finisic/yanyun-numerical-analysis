# -*- coding: utf-8 -*-
"""伤害公式解析期望模型(四层乘区: 基础区/修正区/判定区/增伤区).

公式来源: 报告 3.1 节(社区实测逆推的 PVE 武器技能伤害公式):

    伤害 = [(外功倍率×(外功攻击−目标防御) + 固定外功伤害) × (1+外功穿透/200) × (1+外功伤害加成)
            + 外功倍率×非本系属攻 × (1+穿透/200) × (1+属伤加成)
            + 本系倍率×(本系属攻+80) × (1+穿透/200) × (1+属伤加成)]
           × (会心/会意倍率) × (1+增伤₁+…+增伤ₙ) × 特殊增伤 × (1+定音增伤)

判定机制(报告 3.2 节, 官方渠道描述口径, 判定顺序: 先会意→精准→会心):
    - 会意: 概率 p易, 按最大攻击 × 1.35 结算;
    - 擦伤: 会意未中且精准未中, 概率 (1−p易)(1−p精), 固定按最小攻击结算;
    - 会心/白字: 会意未中且精准命中后,
        * 挤压线上(会心+会意 ≥ 100%): 强制会心, p心=(1−p易)·p精, p白=0;
        * 挤压线下: p心=(1−p易)·p精·p心率, p白=(1−p易)·p精·(1−p心率).

参数标定说明(报告 7.3 节 + 校准规则):
    报告明确给出: 外功 1200~2000、本系属攻 500~900、目标防御 39、
    会意/会心倍率 1.35/1.5、中期面板 min/max 攻击比 r≈0.63(报告 4.1 节原文)。
    报告未给出: 武器技能倍率 M_w、固定外功伤害 F、穿透/各类伤害加成/增伤区数值。
    本模块的处理:
      1) 穿透、外功伤害加成、属伤加成、非本系属攻、特殊增伤、定音增伤、增伤区
         在验证面板上均取中性值(0 或 1), 仅保留基础区线性结构;
         (稀释演示见 experiments/exp2_affix_value.py, 单独设定增伤区)
      2) 由 r=0.63 与验证面板解析期望 15909.9(报告 3.3 节)反解 B0/B1,
         再反推武器倍率 M_w 与固定伤害 F, 使基础伤害量级与报告锚点完全一致。
      3) 假设外功与本系属攻共用同一随机取值比例 t~U(0,1)(完全相关),
         使基础伤害 B(t) 为 t 的线性函数, B0=B(0), B1=B(1), Bm=(B0+B1)/2。
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 报告给定参数(7.3 节)
# ---------------------------------------------------------------------------
WG_MIN, WG_MAX = 1200.0, 2000.0       # 外功攻击最小~最大
ATTR_MIN, ATTR_MAX = 500.0, 900.0     # 本系属攻最小~最大
TARGET_DEFENSE = 39.0                 # 目标防御(副本 boss, 报告 7.3)
HUIYI_MULT = 1.35                     # 会意倍率
HUIXIN_MULT = 1.5                     # 会心倍率
BENXI_BONUS = 1.5                     # 本系属攻流派额外倍率(+50%, 报告 2.1)
ATTR_FLAT = 80.0                      # 公式中本系属攻的 +80 固定项(报告 3.1)

# 报告给定: 中期面板 min/max 攻击比 r≈0.63(4.1 节)
R_MIN_MAX = 0.63
# 报告锚点: 验证面板(精准95/会心60/会意15)解析期望 15909.9(3.3 节)
E_VALIDATION_TARGET = 15909.9


@dataclass(frozen=True)
class Panel:
    """三率面板(率值取 0~1)."""

    jingzhun: float  # 精准率
    huixin: float    # 会心率
    huiyi: float     # 会意率


# 报告用到的三块面板
PANEL_VALIDATION = Panel(0.95, 0.60, 0.15)   # 验证面板(3.3 节)
PANEL_MID = Panel(0.90, 0.40, 0.10)          # 中期面板(4.1 节)
PANEL_GRAD = Panel(1.00, 0.80, 0.27)         # 毕业面板(4.2 节, 见下)
PANEL_FULL_COVERAGE = Panel(1.00, 0.80, 0.20)  # 满覆盖面板(4.1 稳定性)

# 毕业面板说明: 报告 4.2 未给出毕业面板率值与词条数值。
# 实测取 精准100/会心80/会意27(满覆盖、挤压区内, 会心+会意=107),
# 该面板下 ∂E/∂B0 : ∂E/∂B1 = 0.6003, 与报告四条词条收益隐含比值 0.602 一致;
# 若改用锚点4的 (100/80/20) 面板, 排序不变, 小攻击词条收益偏差约 0.05pp。


# ---------------------------------------------------------------------------
# 判定概率(报告 3.2 节解析解)
# ---------------------------------------------------------------------------
def judgment_probs(panel: Panel) -> tuple[float, float, float, float]:
    """返回 (p会意, p会心, p擦伤, p白字), 严格按报告 3.2 节公式."""
    p_yi = panel.huiyi
    p_ca = (1.0 - p_yi) * (1.0 - panel.jingzhun)
    if panel.huixin + panel.huiyi >= 1.0 - 1e-12:
        # 挤压线上: 白字清零, 会意未中且精准命中强制会心
        p_xin = (1.0 - p_yi) * panel.jingzhun
        p_bai = 0.0
    else:
        p_xin = (1.0 - p_yi) * panel.jingzhun * panel.huixin
        p_bai = (1.0 - p_yi) * panel.jingzhun * (1.0 - panel.huixin)
    return p_yi, p_xin, p_ca, p_bai


# ---------------------------------------------------------------------------
# 基础伤害(基础区+修正区)与参数反推标定
# ---------------------------------------------------------------------------
def base_damage(t, m_w: float, fixed: float) -> float:
    """基础伤害 B(t), t∈[0,1] 为攻击区间取值比例(会意/擦伤等价于 t=1/t=0).

    B(t) = 外功倍率×(外功(t)−防御) + 固定外功伤害 + 本系倍率×(本系属攻(t)+80)
    其中本系倍率 = 外功倍率 × 1.5; 修正区各乘子取中性值 1(见模块 docstring)。
    """
    wg = WG_MIN + (WG_MAX - WG_MIN) * t
    attr = ATTR_MIN + (ATTR_MAX - ATTR_MIN) * t
    return m_w * (wg - TARGET_DEFENSE) + fixed + BENXI_BONUS * m_w * (attr + ATTR_FLAT)


def _solve_calibration() -> tuple[float, float, float, float]:
    """由 r=0.63 与 E=15909.9 反解 (B0, B1, M_w, F)."""
    # E = p易·1.35·B1 + p心·1.5·Bm + p擦·B0 + p白·Bm, Bm=(B0+B1)/2, B0=r·B1
    p_yi, p_xin, p_ca, p_bai = judgment_probs(PANEL_VALIDATION)
    c1 = p_yi * HUIYI_MULT + (p_xin * HUIXIN_MULT + p_bai) / 2.0   # B1 系数
    c0 = p_ca + (p_xin * HUIXIN_MULT + p_bai) / 2.0                # B0 系数
    b1 = E_VALIDATION_TARGET / (c0 * R_MIN_MAX + c1)
    b0 = R_MIN_MAX * b1
    # B1−B0 = M_w×[(2000−1200) + 1.5×(900−500)] = 1400·M_w
    m_w = (b1 - b0) / ((WG_MAX - WG_MIN) + BENXI_BONUS * (ATTR_MAX - ATTR_MIN))
    fixed = b0 - m_w * (WG_MIN - TARGET_DEFENSE) - BENXI_BONUS * m_w * (ATTR_MIN + ATTR_FLAT)
    return b0, b1, m_w, fixed


B0, B1, WEAPON_MULT, FIXED_DMG = _solve_calibration()
BM = (B0 + B1) / 2.0


# ---------------------------------------------------------------------------
# 解析期望伤害(报告 3.2 节)
# ---------------------------------------------------------------------------
def expected_damage(panel: Panel, b0: float = B0, b1: float = B1,
                    dmg_bonus: float = 0.0) -> float:
    """期望伤害解析解.

    E[D] = p易·B1·1.35 + p心·Bm·1.5 + p擦·B0 + p白·Bm, 再乘增伤区 (1+dmg_bonus)。
    """
    bm = (b0 + b1) / 2.0
    p_yi, p_xin, p_ca, p_bai = judgment_probs(panel)
    base = (p_yi * HUIYI_MULT * b1 + p_xin * HUIXIN_MULT * bm
            + p_ca * b0 + p_bai * bm)
    return base * (1.0 + dmg_bonus)


def attack_bounds() -> tuple[float, float, float]:
    """返回 (B0, B1, Bm) 标定后的最小/最大/均值基础伤害."""
    return B0, B1, BM
