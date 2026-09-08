# -*- coding: utf-8 -*-
"""实验一: 三率边际收益 + 满覆盖断点 + 挤压区扫描 + 稳定性(报告 3.3/4.1 节).

产出:
    results/exp1_validation.json     锚点1: 解析解×蒙特卡洛互验 + 判定占比
    results/exp1_marginal_rates.csv  锚点2: 中期面板三率边际收益
    results/exp1_squeeze_scan.csv    锚点3: 固定会心70%扫会意(挤压区)
    results/exp1_stability.json      锚点4: 覆盖率与输出稳定性
    figures/fig1_three_rates.png     图1: 挤压区期望曲线与边际收益
    figures/fig3_stability.png       图3: 覆盖率与输出稳定性

运行: python experiments/exp1_three_rates.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.damage_model import (  # noqa: E402
    PANEL_FULL_COVERAGE, PANEL_MID, PANEL_VALIDATION, Panel,
    expected_damage, judgment_probs,
)
from src.monte_carlo import (  # noqa: E402
    coefficient_of_variation, outcome_proportions, simulate_damage,
    simulate_total_damage,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

N_MC = 10_000_000      # 互验模拟次数(千万级, 使误差稳定在 0.01% 以内)
SEED = 20261227
# 60 秒总伤假设: 60 秒 16 次武器技能(约 3.75 秒/次循环)。
# 该假设与报告自洽: 报告单发 CV 与 60 秒波动的比值隐含 √N≈4.05(见 README 参数表)。
HITS_PER_60S = 16
N_WINDOWS = 50_000     # 60 秒窗口模拟个数


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


# ---------------------------------------------------------------------------
def run_validation() -> dict:
    """锚点1: 验证面板(95/60/15)解析解 × 蒙特卡洛互验 + 判定占比."""
    e_analytic = expected_damage(PANEL_VALIDATION)
    dmg = simulate_damage(PANEL_VALIDATION, N_MC, seed=SEED)
    e_sim = float(dmg.mean())
    props = outcome_proportions(PANEL_VALIDATION, N_MC, seed=SEED + 1)
    p_ana = judgment_probs(PANEL_VALIDATION)
    result = {
        "面板": "精准95/会心60/会意15",
        "模拟次数": N_MC,
        "解析期望": round(e_analytic, 4),
        "模拟期望": round(e_sim, 4),
        "相对误差%": round(abs(e_sim - e_analytic) / e_analytic * 100, 5),
        "判定占比_模拟": {k: round(v * 100, 2) for k, v in props.items()},
        "判定占比_解析": dict(zip(("会意", "会心", "擦伤", "白字"),
                                  (round(p * 100, 2) for p in p_ana))),
    }
    return result


def run_marginal_rates() -> list[dict]:
    """锚点2: 中期面板(90/40/10)每条 +5% 率值的边际收益(解析解)."""
    e0 = expected_damage(PANEL_MID)
    rows = []
    for name, panel in (
        ("会意", Panel(PANEL_MID.jingzhun, PANEL_MID.huixin, PANEL_MID.huiyi + 0.05)),
        ("会心", Panel(PANEL_MID.jingzhun, PANEL_MID.huixin + 0.05, PANEL_MID.huiyi)),
        ("精准", Panel(PANEL_MID.jingzhun + 0.05, PANEL_MID.huixin, PANEL_MID.huiyi)),
    ):
        gain = expected_damage(panel) / e0 - 1.0
        rows.append({"词条": name, "+5%率值边际收益%": round(gain * 100, 4)})
    return rows


def run_squeeze_scan() -> tuple[list[dict], dict]:
    """锚点3: 固定会心70%、精准100%, 扫描会意率 0~50%.

    断点在 会意=30%(会心+会意=100%)。返回扫描表与关键指标。
    """
    scan = []
    for yi_pct in range(0, 51):
        e = expected_damage(Panel(1.00, 0.70, yi_pct / 100.0))
        scan.append({"会意率%": yi_pct, "期望伤害": round(e, 4)})
    es = np.array([r["期望伤害"] for r in scan])
    slopes = np.diff(es)                      # 每 +1% 会意的期望增益
    slope_below = float(slopes[10])           # 挤压线下代表点(10%→11%)
    slope_above = float(slopes[30])           # 挤压线上代表点(30%→31%)
    jump = float(es[30] / es[25] - 1.0)       # 最后一条 +5% 词条(25%→30%)
    summary = {
        "线下边际(每1%会意)": round(slope_below, 2),
        "线上边际(每1%会意)": round(slope_above, 2),
        "过线骤降幅度%": round((1 - slope_above / slope_below) * 100, 1),
        "凑满瞬间跃升%(25%->30%)": round(jump * 100, 2),
    }
    return scan, summary


def run_stability() -> dict:
    """锚点4: 覆盖率(90/40/10) vs (100/80/20) 的输出稳定性."""
    out = {}
    for tag, panel in (("低覆盖(90/40/10)", PANEL_MID),
                       ("满覆盖(100/80/20)", PANEL_FULL_COVERAGE)):
        single = simulate_damage(panel, 2_000_000, seed=SEED + 10)
        totals = simulate_total_damage(panel, HITS_PER_60S, N_WINDOWS,
                                       seed=SEED + 11)
        out[tag] = {
            "单发CV%": round(coefficient_of_variation(single) * 100, 2),
            "60秒总伤波动±%": round(coefficient_of_variation(totals) * 100, 2),
        }
    return out


# ---------------------------------------------------------------------------
def plot_fig1(scan: list[dict], summary: dict):
    """图1: 三率挤压区——会意率对期望伤害的边际收益."""
    import matplotlib.pyplot as plt

    yi = np.array([r["会意率%"] for r in scan])
    es = np.array([r["期望伤害"] for r in scan])
    slopes = np.diff(es)
    c_main, c_acc = "#6E8CA0", "#C0876B"     # 低饱和配色

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(yi, es, color=c_main, lw=2)
    ax.axvline(30, color=c_acc, ls="--", lw=1.2)
    ax.annotate("满覆盖断点\n(会心70+会意30=100%)", xy=(30, es[30]),
                xytext=(33, es[8]), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=c_acc))
    ax.set_xlabel("会意率(%)")
    ax.set_ylabel("期望伤害")
    ax.set_title("期望伤害曲线(固定会心70%、精准100%)")

    ax = axes[1]
    ax.plot(yi[:-1] + 0.5, slopes, color=c_main, lw=2)
    ax.axvline(30, color=c_acc, ls="--", lw=1.2)
    ax.annotate("凑满瞬间跳升\n(白字清零)", xy=(30, slopes[29]), xytext=(36, slopes[29] * 0.75),
                fontsize=9, arrowprops=dict(arrowstyle="->", color=c_acc), color=c_acc)
    ax.text(8, slopes[10] * 3.2, f"挤压线下 ≈{summary['线下边际(每1%会意)']:.1f}",
            fontsize=9, color=c_main)
    ax.text(38, slopes[10] * 3.2, f"挤压线上 ≈{summary['线上边际(每1%会意)']:.1f}",
            fontsize=9, color=c_main)
    ax.set_xlabel("会意率(%)")
    ax.set_ylabel("每+1%会意的期望增益")
    ax.set_title("会意边际收益: 过线骤降"
                 f"({summary['线下边际(每1%会意)']:.1f}→{summary['线上边际(每1%会意)']:.1f})")
    for a in axes:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("图1 三率挤压区: 会意率对期望伤害的边际收益", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_three_rates.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3(stability: dict):
    """图3: 三率覆盖率与输出稳定性."""
    import matplotlib.pyplot as plt

    labels = list(stability.keys())
    cv_single = [stability[k]["单发CV%"] for k in labels]
    cv_total = [stability[k]["60秒总伤波动±%"] for k in labels]

    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, cv_single, w, label="单发伤害变异系数",
                color="#6E8CA0")
    b2 = ax.bar(x + w / 2, cv_total, w, label="60秒总伤波动(±)",
                color="#C0876B")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("波动幅度(%)")
    ax.set_title("图3 三率覆盖率与输出稳定性")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_stability.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    font = _setup_chinese_font()
    if font:
        print(f"[字体] {font}")

    validation = run_validation()
    with open(os.path.join(RESULTS_DIR, "exp1_validation.json"), "w",
              encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)
    print("[锚点1] 解析 %.1f vs 模拟 %.1f (误差 %.4f%%); 判定占比 %s"
          % (validation["解析期望"], validation["模拟期望"],
             validation["相对误差%"], validation["判定占比_模拟"]))

    marginal = run_marginal_rates()
    with open(os.path.join(RESULTS_DIR, "exp1_marginal_rates.csv"), "w",
              encoding="utf-8") as f:
        f.write("词条,+5%率值边际收益%\n")
        for r in marginal:
            f.write(f"{r['词条']},{r['+5%率值边际收益%']}\n")
    print("[锚点2] " + " | ".join(f"{r['词条']} {r['+5%率值边际收益%']}%"
                                   for r in marginal))

    scan, squeeze = run_squeeze_scan()
    with open(os.path.join(RESULTS_DIR, "exp1_squeeze_scan.csv"), "w",
              encoding="utf-8") as f:
        f.write("会意率%,期望伤害\n")
        for r in scan:
            f.write(f"{r['会意率%']},{r['期望伤害']}\n")
    print("[锚点3]", squeeze)

    stability = run_stability()
    with open(os.path.join(RESULTS_DIR, "exp1_stability.json"), "w",
              encoding="utf-8") as f:
        json.dump({**stability,
                   "假设": {"60秒攻击次数": HITS_PER_60S, "窗口数": N_WINDOWS}},
                  f, ensure_ascii=False, indent=2)
    print("[锚点4]", stability)

    plot_fig1(scan, squeeze)
    plot_fig3(stability)
    print("[完成] fig1_three_rates.png / fig3_stability.png 已落盘")


if __name__ == "__main__":
    main()
