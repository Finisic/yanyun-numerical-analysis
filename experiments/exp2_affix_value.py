# -*- coding: utf-8 -*-
"""实验二: 词条边际收益与乘区稀释(报告 4.2 节).

产出:
    results/exp2_affix_marginal.csv  锚点5: 毕业面板单条词条边际收益
    results/exp2_dilution.json       锚点5: 加算增伤稀释演示
    figures/fig2_affix_value.png     图2: 单条词条边际收益对比

运行: python experiments/exp2_affix_value.py

参数标定说明:
    报告未给出毕业面板的词条数值, 只给出四条词条的相对收益
    (最大本系属攻+2.81% > 最大外功+1.96% > 最小本系属攻+1.69% > 最小外功+1.18%)。
    本脚本以最大外功/最大属攻两条为标定点, 反解词条数值 X(外功)、Y(属攻),
    再以同一组 X/Y 预测最小词条收益, 检验模型自洽性(结果见 reconciliation.md)。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.damage_model import (  # noqa: E402
    BENXI_BONUS, HUIYI_MULT, HUIXIN_MULT, PANEL_GRAD, WEAPON_MULT,
    expected_damage, judgment_probs,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

# 报告锚点(4.2 节)
TARGET_MAX_WG = 0.0196    # 最大外功 +1.96%
TARGET_MAX_ATTR = 0.0281  # 最大本系属攻 +2.81%
DILUTION_EXISTING = 0.45  # 已有 45% 增伤
DILUTION_ADD = 0.10       # 新增 +10% 增伤


def _setup_chinese_font():
    """中文字体探测(回退链), 仅在直接运行脚本时生效."""
    try:
        import matplotlib
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
                     "Microsoft YaHei", "PingFang SC"):
            if name in available:
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
    except Exception:
        pass
    return None


def grad_panel_coeffs():
    """毕业面板(满覆盖、挤压区)下 ∂E/∂B1 与 ∂E/∂B0.

    满覆盖时 E = p易·1.35·B1 + (1−p易)·p精·1.5·Bm, Bm=(B0+B1)/2。
    """
    p_yi, p_xin, p_ca, p_bai = judgment_probs(PANEL_GRAD)
    assert abs(p_bai) < 1e-12 and abs(p_ca) < 1e-12, "毕业面板应为满覆盖无白字无擦伤"
    c_b1 = p_yi * HUIYI_MULT + p_xin * HUIXIN_MULT / 2.0
    c_b0 = p_xin * HUIXIN_MULT / 2.0
    return c_b1, c_b0


def run_affix_marginal() -> tuple[list[dict], dict]:
    """锚点5: 词条边际收益排序 + 词条数值反推标定."""
    e_grad = expected_damage(PANEL_GRAD)
    c_b1, c_b0 = grad_panel_coeffs()

    # 反解词条数值: 使 最大外功/最大属攻 收益精确等于报告锚点
    x_wg = TARGET_MAX_WG * e_grad / (c_b1 * WEAPON_MULT)
    y_attr = TARGET_MAX_ATTR * e_grad / (c_b1 * BENXI_BONUS * WEAPON_MULT)

    def gain_max(mult):   # 最大攻击词条: 只抬 B1
        return c_b1 * mult / e_grad

    def gain_min(mult):   # 最小攻击词条: 只抬 B0
        return c_b0 * mult / e_grad

    rows = [
        {"词条": "最大本系属攻", "词条数值": round(y_attr, 1),
         "边际收益%": round(gain_max(BENXI_BONUS * WEAPON_MULT * y_attr) * 100, 4)},
        {"词条": "最大外功", "词条数值": round(x_wg, 1),
         "边际收益%": round(gain_max(WEAPON_MULT * x_wg) * 100, 4)},
        {"词条": "最小本系属攻", "词条数值": round(y_attr, 1),
         "边际收益%": round(gain_min(BENXI_BONUS * WEAPON_MULT * y_attr) * 100, 4)},
        {"词条": "最小外功", "词条数值": round(x_wg, 1),
         "边际收益%": round(gain_min(WEAPON_MULT * x_wg) * 100, 4)},
    ]
    # 率值词条参考值: 凑满覆盖瞬间的跃升(报告 4.1: +7.9%~8.4%)
    e_before = expected_damage(type(PANEL_GRAD)(1.00, 0.70, 0.25))
    e_after = expected_damage(type(PANEL_GRAD)(1.00, 0.70, 0.30))
    rate_affix = (e_after / e_before - 1.0)
    rows.append({"词条": "率值词条(凑满覆盖)", "词条数值": "+5%会意",
                 "边际收益%": round(rate_affix * 100, 4)})

    meta = {"毕业面板": "精准100/会心80/会意27(满覆盖, 报告未给率值, 标定值)",
            "E_grad": round(e_grad, 1),
            "反推词条数值": {"外功词条": round(x_wg, 2), "本系属攻词条": round(y_attr, 2)}}
    return rows, meta


def run_dilution() -> dict:
    """锚点5: 加算增伤区稀释演示(报告 4.2).

    已有 45% 加算增伤时, 再 +10% 增伤的实际收益 = 0.10/1.45 = 6.90%;
    独立乘区(如会意伤害加成、特殊增伤)的 +10% 足额生效。
    """
    actual = DILUTION_ADD / (1.0 + DILUTION_EXISTING)
    return {
        "已有加算增伤": DILUTION_EXISTING,
        "新增增伤": DILUTION_ADD,
        "加算区实际收益%": round(actual * 100, 4),
        "独立乘区实际收益%": round(DILUTION_ADD * 100, 4),
        "稀释比例%": round((1 - actual / DILUTION_ADD) * 100, 2),
    }


def plot_fig2(rows: list[dict], dilution: dict):
    """图2: 单条词条的边际收益对比 + 加算稀释."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    names = [r["词条"] for r in rows]
    gains = [r["边际收益%"] for r in rows]
    colors = ["#8E9E82", "#6E8CA0", "#A9BBA0", "#93A9BB", "#C0876B"]
    bars = ax.barh(names[::-1], gains[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.2f%%", fontsize=9, padding=3)
    ax.set_xlabel("单条词条边际收益(%)")
    ax.set_title("毕业面板词条边际收益排序")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(gains) * 1.18)

    ax = axes[1]
    bars = ax.bar(["名义+10%增伤", "加算区实际", "独立乘区实际"],
                  [10.0, dilution["加算区实际收益%"], dilution["独立乘区实际收益%"]],
                  color=["#C9C9C9", "#C0876B", "#6E8CA0"])
    ax.bar_label(bars, fmt="%.2f%%", fontsize=9)
    ax.set_ylabel("实际收益(%)")
    ax.set_title("加算稀释(已有45%增伤)")
    ax.grid(axis="y", alpha=0.25)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("图2 单条词条的边际收益对比", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_affix_value.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    font = _setup_chinese_font()
    if font:
        print(f"[字体] {font}")

    rows, meta = run_affix_marginal()
    with open(os.path.join(RESULTS_DIR, "exp2_affix_marginal.csv"), "w",
              encoding="utf-8") as f:
        f.write("词条,词条数值,边际收益%\n")
        for r in rows:
            f.write(f"{r['词条']},{r['词条数值']},{r['边际收益%']}\n")
    print("[锚点5] 词条标定:", meta["反推词条数值"])
    for r in rows:
        print(f"        {r['词条']}: {r['边际收益%']}%")

    dilution = run_dilution()
    with open(os.path.join(RESULTS_DIR, "exp2_dilution.json"), "w",
              encoding="utf-8") as f:
        json.dump(dilution, f, ensure_ascii=False, indent=2)
    print(f"[锚点5] 稀释: 已有45%增伤时+10%增伤实际收益 "
          f"{dilution['加算区实际收益%']}%(独立乘区 {dilution['独立乘区实际收益%']}%)")

    plot_fig2(rows, dilution)
    print("[完成] fig2_affix_value.png 已落盘")


if __name__ == "__main__":
    main()
