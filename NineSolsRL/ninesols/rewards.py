"""
rewards.py —— 三層 reward（參考 Hollow Knight RL 論文）

  base         勝負結果（擊敗 boss / 死亡）
  sub          血量變化（boss 掉血、玩家掉血）
  instrumental 行為引導（存活、主動接戰）

慣例：所有權重都是「正值」，公式裡明確用 +（獎勵）/ -（懲罰）。
"""

# ---- 權重（全部正值；正負號在公式裡決定）----
#
# 重點是各項之間的「比例」，不是絕對值 —— 訓練時 VecNormalize 會對 reward 做
# 跑動正規化（除以 std），整體放大/縮小同一倍數對訓練等價。這版重新設計了比例：
#   贏一場 ≈ +300(WIN) +300(打掉滿血 BOSS_HURT) = +600
#   輸一場 ≈ -100(DEATH) -100(掉滿血 PLAYER_HURT) = -200
# 舊版 W_WIN=1000 / W_BOSS_HURT=200 比例失衡（打傷 boss 的累積獎勵遠小於勝負），
# 這版讓「打掉滿血」與「勝利」同量級，sub reward 才推得動 actor。
W_WIN          = 300.0    # 擊敗 boss             → 獎勵 +
W_DEATH        = 100.0    # 玩家死亡              → 懲罰 -
W_BOSS_HURT    = 300.0    # boss 每損失 1.0 血量比例 → 獎勵 +
W_PLAYER_HURT  = 100.0    # 玩家每損失 1.0 血量比例 → 懲罰 -
W_ALIVE        = 0.02     # 每步存活              → 獎勵 +（很小，避免鼓勵 idle）

# 格檔：只獎勵成功格檔、不懲罰（避免 agent 學成不格檔）。
# 亂按格檔遊戲自己會讓窗口縮小、parry 失敗 → 自然拿不到獎勵。
W_PARRY_PRECISE   = 10.0  # 精確格檔              → 獎勵 +（大，主要目標）
W_PARRY_IMPRECISE = 3.0   # 不精確格檔            → 獎勵 +（小）

# instrumental：引導 agent 主動接戰（解決「進關卡發呆不打」）
# in_range / face 都要靠近 boss 才拿得到 → 比 idle 的 W_ALIVE 大得多，
# 梯度自然把 agent 拉向 boss。
W_APPROACH     = 0.02     # 每拉近 boss 1 單位距離 → 獎勵 +（拉遠則 -）
W_IN_RANGE     = 0.1      # 處於接戰距離內         → 獎勵 +
W_FACE_BOSS    = 0.05     # 面向 boss             → 獎勵 +

ENGAGE_RANGE   = 200.0    # 視為「接戰距離」的水平距離（遊戲單位）


def compute_reward(prev: dict | None, cur: dict, use_instrumental: bool = True) -> float:
    r = 0.0

    # ---- base：勝負 ----
    if cur.get("done"):
        if cur.get("php", 1) <= 0:
            r -= W_DEATH                                  # 死亡懲罰
        elif cur.get("boss_present") and cur.get("bhp", 1) <= 0:
            r += W_WIN                                    # 擊敗 boss

    # ---- sub：血量變化 ----
    if prev is not None:
        dphp = cur.get("php_pct", 1.0) - prev.get("php_pct", 1.0)
        if dphp < 0:
            r -= (-dphp) * W_PLAYER_HURT                  # 玩家掉血懲罰

        if cur.get("boss_present") and prev.get("boss_present"):
            dbhp = cur.get("bhp_pct", 1.0) - prev.get("bhp_pct", 1.0)
            if dbhp < 0:
                r += (-dbhp) * W_BOSS_HURT                # boss 掉血獎勵

        # 成功格檔 → 獎勵（用 parry_count 上升緣偵測，每次格檔只給一次）
        if cur.get("parry_count", 0) > prev.get("parry_count", 0):
            pr = cur.get("last_parry_result", 0)
            if pr >= 2:
                r += W_PARRY_PRECISE                      # 精確格檔
            elif pr == 1:
                r += W_PARRY_IMPRECISE                    # 不精確格檔

    # ---- instrumental：存活 + 引導接戰 ----
    if use_instrumental:
        r += W_ALIVE                                      # 存活獎勵

        if cur.get("boss_present"):
            cur_dist = abs(cur.get("bdx", 0.0))

            # 拉近與 boss 的距離 → 獎勵；拉遠 → 懲罰（鼓勵主動接戰）
            if prev is not None and prev.get("boss_present"):
                prev_dist = abs(prev.get("bdx", 0.0))
                r += (prev_dist - cur_dist) * W_APPROACH

            # 在接戰距離內 → 獎勵
            if cur_dist < ENGAGE_RANGE:
                r += W_IN_RANGE

            # 面向 boss → 獎勵
            bdx = cur.get("bdx", 0.0)
            facing = cur.get("facing", 1)
            if (bdx >= 0 and facing > 0) or (bdx < 0 and facing < 0):
                r += W_FACE_BOSS

    return r
