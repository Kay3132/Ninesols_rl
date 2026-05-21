"""
rewards.py —— 三層 reward（參考 Hollow Knight RL 論文）

  base         勝負結果（擊敗 boss / 死亡）
  sub          血量變化（boss 掉血給正、玩家掉血給負）
  instrumental 行為引導（存活、面向 boss…前期幫助收斂，後期可關）
"""

# reward 權重（之後可調）
W_WIN          = 1000.0   # 擊敗 boss
W_DEATH        = -100.0   # 玩家死亡
W_PLAYER_HURT  = 100.0    # 玩家每損失 1.0 (php_pct) 的懲罰
W_BOSS_HURT    = 200.0    # boss 每損失 1.0 (bhp_pct) 的獎勵
W_ALIVE        = 0.01     # 每步存活


def compute_reward(prev: dict | None, cur: dict, use_instrumental: bool = True) -> float:
    r = 0.0

    # ---- base：勝負 ----
    if cur.get("done"):
        if cur.get("php", 1) <= 0:
            r += W_DEATH
        elif cur.get("boss_present") and cur.get("bhp", 1) <= 0:
            r += W_WIN

    # ---- sub：血量變化 ----
    if prev is not None:
        dphp = cur.get("php_pct", 1.0) - prev.get("php_pct", 1.0)
        if dphp < 0:
            r += dphp * W_PLAYER_HURT          # dphp 為負 → 懲罰

        if cur.get("boss_present") and prev.get("boss_present"):
            dbhp = cur.get("bhp_pct", 1.0) - prev.get("bhp_pct", 1.0)
            if dbhp < 0:
                r += (-dbhp) * W_BOSS_HURT     # boss 掉血 → 獎勵

    # ---- instrumental：行為引導 ----
    if use_instrumental:
        r += W_ALIVE

    return r
