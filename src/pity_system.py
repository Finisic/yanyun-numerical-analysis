# -*- coding: utf-8 -*-
"""调律随机养成模拟: 纯随机 / 动态保底 / 动态保底+变音自选 三机制.

机制参数(报告 7.3 节 + 4.3 节, 社区实测口径):
    - 推荐词条基础概率 30%;
    - 动态保底: 连续未出推荐词条, 概率每次 +10pp(30%→40%→…→100% 封顶),
      一旦出荐(任意推荐词条)即重置;
    - 推荐词条池 8 选 1: 指定词条的单次概率 = 推荐概率 / 8;
    - 变音硬保底: 单件装备变音(调律)累计 6 次可自选词条(周年庆新赛季机制)。

毕业目标(报告 4.3 节):
    - 小毕业: 每件装备 3 条任意推荐词条;
    - 完美毕业: 每件装备 1 条指定词条(推荐池 8 选 1)。

关键建模细节(经数值验证, 见 results/reconciliation.md):
    1) 完美毕业场景下, 出"非指定的推荐词条"同样会重置动态保底计数
       (官方机制描述为"出荐重置", 并非"出指定词条才重置"),
       这是动态保底完美毕业均值仍高达 ~157 件的原因;
    2) 小毕业场景下变音自选不提供实质帮助: 3 条任意推荐的期望成本仅 ~7.3 次/件,
       而变音需累计 6 次才能自选 1 条, 理性玩家会把每件仅一次的变音机会
       留给完美毕业的指定词条, 故小毕业在三机制中的成本分布一致
       (与报告"动态保底"和"动态+变音"两列小毕业数据均为 59 件一致)。

成本单位: 狗粮装备件数(每次调律消耗 1 件)。
"""

from __future__ import annotations

import numpy as np

P_BASE = 0.30      # 推荐词条基础概率
P_STEP = 0.10      # 动态保底递增步长(pp)
POOL_SIZE = 8      # 推荐词条池大小(8 选 1)
BIANYIN_CAP = 6    # 变音硬保底次数
N_ITEMS = 8        # 全身装备件数
N_PLAYERS = 20_000  # 虚拟玩家数
DEFAULT_SEED = 20261227

MECHANISMS = ("pure", "pity", "pity_bianyin")
MECHANISM_NAMES = {
    "pure": "纯随机(无保底)",
    "pity": "动态保底",
    "pity_bianyin": "动态保底+变音自选",
}
GOAL_NAMES = {"small": "小毕业(每件3条任意推荐)", "perfect": "完美毕业(每件1条指定词条)"}


def _simulate_item_costs(n: int, goal: str, mechanism: str,
                         rng: np.random.Generator) -> np.ndarray:
    """向量化模拟 n 件装备的毕业成本(每件装备的调律次数)."""
    use_pity = mechanism in ("pity", "pity_bianyin")
    use_bianyin = mechanism == "pity_bianyin"

    rolls = np.zeros(n, dtype=np.int64)      # 已调律次数
    fail = np.zeros(n, dtype=np.int64)       # 连续未出荐次数(动态保底计数)
    got = np.zeros(n, dtype=np.int64)        # 已获得推荐词条数(小毕业用)
    done = np.zeros(n, dtype=bool)

    while not done.all():
        active = ~done
        idx = np.nonzero(active)[0]
        p = np.full(idx.size, P_BASE)
        if use_pity:
            p = np.minimum(P_BASE + P_STEP * fail[idx], 1.0)
        u = rng.random(idx.size)

        if goal == "small":
            hit = u < p                       # 出任意推荐词条
            got[idx[hit]] += 1
            fail[idx] = np.where(hit, 0, fail[idx] + 1)
            done[idx] = got[idx] >= 3
        else:  # perfect: 指定词条(8选1), 出非指定推荐也会重置保底
            hit_specific = u < p / POOL_SIZE
            hit_other_rec = (~hit_specific) & (u < p)
            fail[idx] = np.where(hit_other_rec, 0,
                                 np.where(hit_specific, fail[idx], fail[idx] + 1))
            done[idx] = hit_specific

        rolls[idx] += 1

        # 变音硬保底: 累计 6 次可自选词条
        if use_bianyin and goal == "perfect":
            capped = active & (rolls >= BIANYIN_CAP) & (~done)
            done[capped] = True              # 第 6 次直接自选指定词条, 成本=6
        # 小毕业场景不使用变音(见模块 docstring 说明)

    return rolls


def simulate_players(goal: str, mechanism: str, n_players: int = N_PLAYERS,
                     n_items: int = N_ITEMS,
                     seed: int = DEFAULT_SEED) -> dict:
    """模拟 n_players 名玩家(每人 n_items 件装备)的毕业成本.

    返回 {'costs': 每玩家成本数组, 'mean': ..., 'p99': ..., 'max': ...}
    """
    if goal not in GOAL_NAMES:
        raise ValueError(f"未知毕业目标: {goal}")
    if mechanism not in MECHANISMS:
        raise ValueError(f"未知机制: {mechanism}")
    rng = np.random.default_rng(seed)
    item_costs = _simulate_item_costs(n_players * n_items, goal, mechanism, rng)
    costs = item_costs.reshape(n_players, n_items).sum(axis=1)
    return {
        "costs": costs,
        "mean": float(costs.mean()),
        "p99": float(np.percentile(costs, 99)),
        "max": int(costs.max()),
    }


def hard_caps(goal: str, n_items: int = N_ITEMS) -> float:
    """动态保底+变音机制下全身完美毕业的理论硬上限(每件 ≤6 次)."""
    if goal == "perfect":
        return BIANYIN_CAP * n_items
    return np.inf
