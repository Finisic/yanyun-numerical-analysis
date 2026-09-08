# -*- coding: utf-8 -*-
"""实验三: 2万玩家×8件装备毕业成本三机制对比(报告 4.3 节).

机制: 纯随机(无保底) / 动态保底 / 动态保底+变音自选(实现见 src/pity_system.py)
产出:
    results/exp3_pity_cost.csv   锚点6: 三机制×两目标 毕业成本统计
    results/exp3_pity_cost.json  同上(含成本分布, 供绘图复核)
    figures/fig4_pity_cost.png   图4: 全身装备毕业成本模拟

运行: python experiments/exp3_pity_cost.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pity_system import (  # noqa: E402
    GOAL_NAMES, MECHANISM_NAMES, MECHANISMS, N_ITEMS, N_PLAYERS, hard_caps,
    simulate_players,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
SEED = 20261227


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


def run_all() -> dict:
    """三机制 × 两目标, 每组 2 万玩家 × 8 件装备."""
    out = {}
    for goal in ("small", "perfect"):
        out[goal] = {}
        for mech in MECHANISMS:
            r = simulate_players(goal, mech, N_PLAYERS, N_ITEMS, seed=SEED)
            out[goal][mech] = r
    return out


def save_results(out: dict):
    with open(os.path.join(RESULTS_DIR, "exp3_pity_cost.csv"), "w",
              encoding="utf-8") as f:
        f.write("毕业目标,机制,均值,P99,最大值\n")
        for goal in ("small", "perfect"):
            for mech in MECHANISMS:
                r = out[goal][mech]
                f.write(f"{GOAL_NAMES[goal]},{MECHANISM_NAMES[mech]},"
                        f"{r['mean']:.1f},{r['p99']:.0f},{r['max']}\n")

    jsonable = {
        GOAL_NAMES[g]: {
            MECHANISM_NAMES[m]: {
                "均值": round(out[g][m]["mean"], 2),
                "P99": round(out[g][m]["p99"], 1),
                "最大值": out[g][m]["max"],
            } for m in MECHANISMS
        } for g in ("small", "perfect")
    }
    jsonable["硬上限(完美毕业, 动态+变音)"] = hard_caps("perfect")
    with open(os.path.join(RESULTS_DIR, "exp3_pity_cost.json"), "w",
              encoding="utf-8") as f:
        json.dump(jsonable, f, ensure_ascii=False, indent=2)


def plot_fig4(out: dict):
    """图4: 全身装备毕业成本模拟(均值/P99/最大值, 三机制×两目标)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    stats = ("mean", "p99", "max")
    stat_names = ("均值", "P99", "最大值")
    colors = {"mean": "#6E8CA0", "p99": "#C0876B", "max": "#8E9E82"}

    for ax, goal, title in (
        (axes[0], "small", "小毕业(每件3条任意推荐)"),
        (axes[1], "perfect", "完美毕业(每件1条指定词条)"),
    ):
        x = np.arange(len(MECHANISMS))
        w = 0.26
        for i, s in enumerate(stats):
            vals = [out[goal][m][s] for m in MECHANISMS]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=stat_names[i],
                          color=colors[s])
            ax.bar_label(bars, fmt="%.0f", fontsize=8)
        ax.set_xticks(x, [MECHANISM_NAMES[m] for m in MECHANISMS], fontsize=9)
        ax.set_ylabel("毕业成本(狗粮装备件数)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if goal == "perfect":
            ax.axhline(hard_caps("perfect"), color="#C0876B", ls=":", lw=1)
            ax.annotate("硬上限 48 件", xy=(1.45, 90), fontsize=9,
                        color="#C0876B")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"图4 全身装备毕业成本模拟({N_PLAYERS // 10000}万玩家×{N_ITEMS}件装备)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_pity_cost.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    font = _setup_chinese_font()
    if font:
        print(f"[字体] {font}")

    out = run_all()
    save_results(out)
    for goal in ("small", "perfect"):
        for mech in MECHANISMS:
            r = out[goal][mech]
            print(f"[锚点6] {GOAL_NAMES[goal]} | {MECHANISM_NAMES[mech]}: "
                  f"均值 {r['mean']:.1f} / P99 {r['p99']:.0f} / 最大 {r['max']}")
    plot_fig4(out)
    print("[完成] fig4_pity_cost.png 已落盘")


if __name__ == "__main__":
    main()
