"""
rewards.py —— 三層 reward（參考 Hollow Knight RL 論文）

  base         勝負結果（擊敗 boss / 死亡）
  sub          血量變化（boss 掉血、玩家掉血）
  instrumental 行為引導（存活、主動接戰）

慣例：所有權重都是「正值」，公式裡明確用 +（獎勵）/ -（懲罰）。
"""

# ---- 權重----
# 壓縮極端值，將 Base/Sub 與 Instrumental 的落差控制在 100 倍以內。
W_WIN          = 50.0     # 擊敗 boss (降維，交給 PPO 的 gamma 折扣去累積)
W_DEATH        = 20.0     # 玩家死亡 (降維)
W_BOSS_HURT    = 50.0     # boss 每損失 1.0 血量比例
W_PLAYER_HURT  = 20.0     # 玩家每損失 1.0 血量比例 (早期調低，讓它敢換血)

W_MAX_HURT_PENALTY = 20.0  # 設置單局扣血懲罰上限 (剛好等於一條滿血的懲罰量)

W_PARRY_PRECISE   = 5.0   # 精確格檔
W_PARRY_IMPRECISE = 1.0   # 不精確格檔

# --- instrumental：per-step 項一律改「每秒速率」，compute_reward 內乘 dt ---
# 為什麼改 dt-based：原本「每步固定值」在遊戲加速 / fps 抖動時，「每秒幾步」
# 會變 → 每秒拿到的 reward 跟著漂。改成「每秒速率 × dt」後不管 1×/2× 加速、
# fps 怎麼抖，每「遊戲秒」拿到的 reward 都一樣 → 加速訓練→1× 跑可完美銜接。
W_TIME_PENALTY_PS = -0.3  # 每秒：時間懲罰（原 -0.01/步 @30Hz）
W_IN_RANGE_PS     = 3.0   # 每秒：處於接戰距離內
W_FACE_BOSS_PS    = 1.5   # 每秒：面向 boss
# W_COWARD 從原 -0.2/步(=-6/秒) 大幅調小：原值太大時，「衝過去送死」會比
# 「在遠處龜縮」總分更高 → agent 學會自殺。接戰動力應來自「打到 boss 的大獎勵」，
# 而不是「龜縮被重罰」。
W_COWARD_PS       = -0.6  # 每秒：龜縮（遠離戰區）懲罰

W_APPROACH     = 0.02     # 每拉近 1 單位（位移型，天生與步數無關，不乘 dt）
ENGAGE_RANGE   = 200.0    # 視為「接戰距離」

# 一次性接戰 bonus（實際在 env.py 套用，因為需要 per-episode 狀態）。
# boss 未出現時所有接戰 instrumental 都被 if boss_present 擋掉，那段只剩
# 沒有方向性的時間懲罰。給「本局首次 boss 出現」一筆一次性獎勵，當作
# 「去把 boss 觸發出來」的錨點。一次性 → 不會被 boss 偵測 flicker 刷分。
W_BOSS_ENGAGED = 15.0

def compute_reward(prev: dict | None, cur: dict, cumulative_hurt: float, use_instrumental: bool = True) -> tuple[float, float]:
    r = 0.0
    step_hurt_penalty = 0.0  # 記錄這一步實際扣了多少分

    # ---- base：勝負 ----
    if cur.get("done"):
        if cur.get("php", 1) <= 0:
            r -= W_DEATH
        elif cur.get("boss_present") and cur.get("bhp", 1) <= 0:
            r += W_WIN

    # ---- sub：血量變化 ----
    if prev is not None:
        dphp = cur.get("php_pct", 1.0) - prev.get("php_pct", 1.0)
        if dphp < 0:
            raw_penalty = (-dphp) * W_PLAYER_HURT
            
            # 【核心邏輯：計算剩餘的懲罰額度】
            remaining_penalty_allowance = W_MAX_HURT_PENALTY - cumulative_hurt
            if remaining_penalty_allowance > 0:
                # 確保扣的分數不會超過剩餘額度
                actual_penalty = min(raw_penalty, remaining_penalty_allowance)
                r -= actual_penalty
                step_hurt_penalty = actual_penalty  # 回傳給 env.py 去累加

        if cur.get("boss_present") and prev.get("boss_present"):
            dbhp = cur.get("bhp_pct", 1.0) - prev.get("bhp_pct", 1.0)
            if dbhp < 0:
                r += (-dbhp) * W_BOSS_HURT

        # 成功格檔 → 獎勵（用 parry_count 上升緣偵測，每次格檔只給一次）
        if cur.get("parry_count", 0) > prev.get("parry_count", 0):
            pr = cur.get("last_parry_result", 0)
            if pr >= 2:
                r += W_PARRY_PRECISE                      # 精確格檔
            elif pr == 1:
                r += W_PARRY_IMPRECISE                    # 不精確格檔

    # ---- instrumental：時間懲罰 + 引導接戰（per-step 項乘 dt）----
    if use_instrumental:
        # dt：距上一筆 state 經過的「遊戲時間」(秒)。夾住異常值
        # (第一筆 / 卡頓時 dt 可能異常大或為 0)。
        dt = cur.get("dt", 1.0 / 30.0)
        dt = min(max(dt, 0.0), 0.2)

        r += W_TIME_PENALTY_PS * dt                       # 時間懲罰

        if cur.get("boss_present"):
            cur_dist = abs(cur.get("bdx", 0.0))

            if prev is not None and prev.get("boss_present"):
                prev_dist = abs(prev.get("bdx", 0.0))
                dist_delta = prev_dist - cur_dist
                # 限制單步移動獎勵上限，避免 Dash 造成的梯度爆炸
                dist_delta = max(-10.0, min(10.0, dist_delta))
                r += dist_delta * W_APPROACH               # 位移型，不乘 dt

            if cur_dist < ENGAGE_RANGE:
                r += W_IN_RANGE_PS * dt
            else:
                r += W_COWARD_PS * dt                      # 龜縮懲罰

            bdx = cur.get("bdx", 0.0)
            facing = cur.get("facing", 1)
            if (bdx >= 0 and facing > 0) or (bdx < 0 and facing < 0):
                r += W_FACE_BOSS_PS * dt

    return r, step_hurt_penalty
