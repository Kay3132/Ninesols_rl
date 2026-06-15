"""
rewards.py —— 三層 reward（參考 Hollow Knight RL 論文）

  base         勝負結果（擊敗 boss / 死亡）
  sub          血量變化（boss 掉血、玩家掉血）
  instrumental 行為引導（存活、主動接戰）

慣例：所有權重都是「正值」，公式裡明確用 +（獎勵）/ -（懲罰）。
"""

# ---- 權重----
# 壓縮極端值，將 Base/Sub 與 Instrumental 的落差控制在 100 倍以內。
W_WIN          = 150.0  
W_DEATH        = 100.0     
W_BOSS_HURT    = 100.0    # boss 每損失 1.0 血量比例
W_TRUNCATION   = 300.0

# rebuilt 2026-06-14: 快速通關 bonus —— WIN 時依「用掉 curriculum-scaled max_steps 的比例」給分。
# env.py 套用(需 per-episode 的 _steps / _effective_max_steps)。linear 偏好快速:
# speed_bonus = W_SPEED_BONUS * (1 - steps/effective_max_steps)。值參考自舊 No-hit-goal 線。
W_SPEED_BONUS  = 200.0

# v1.20.2: hurt penalty 改 quadratic（取代原 linear + W_MAX_HURT_PENALTY cap）
# 動機：原 linear+cap 在「滿 cap 後被打死」情境下 reward = 0（phase 2 close-out 殞落）。
# 改 quadratic 之後:
#   - 小擊 (5%) = 0.4 點 (比舊 1.0 還輕 → 接戰退縮風險更低)
#   - 中擊 (25%) = 9.4 點
#   - 重擊 (50%) = 37.5 點（舊 cap=10）
#   - 致命 (100%) = 150 點
# 訊號連續、不需 cap、不需 cumulative_hurt 追蹤。burst penalty 仍保留抓 combo。
W_PLAYER_HURT_Q = 120.0   # 2026-06-15: 130 → 120(降挨打懲罰)。quadratic 係數,r -= W_PLAYER_HURT_Q * dphp^2

# v1.20.1: burst damage penalty —— 短時間連續挨打(連段)額外懲罰，獨立 stream
# quadratic 對「連 5 擊」跟「散 5 擊」給同樣分,burst 仍負責抓 combo 訊號
BURST_WINDOW_STEPS = 60   # rebuilt 75529b2 2026-06-13: 30 → 60 (~2s, rose R15)，抓長連段
W_BURST_PENALTY    = 20.0 # rebuilt 75529b2 2026-06-13: 10 → 20 (rose R15)，提高連段挨打訊號

W_PARRY_PRECISE   = 50.0  # 精確格檔（v1.18.0: 5→20，格檔是稀有事件，單次給足夠分量）
W_PARRY_IMPRECISE = 20.0   # 不精確格檔（v1.18.0: 1→5）
# rebuilt 2026-06-15 Round 23: 加回 heal reward(Round 1 原值 35 → Round 18 rebuilt 砍掉)。
# php_pct 增量 × 35:回 1% = +0.35,回 50% = +17.5(滿瓶 8 瓶,殘血滿喝 ~+140 上限)。
# env.py 已 mask 滿血(php_pct>=99%)attack=5,wasted heal 不會觸發,可直接用 dphp×W 結構。
W_HEAL_SUCCESS    = 35.0
# rebuilt 2026-06-15 Round 22: 復活 Round 3/5 attempt incentive。
# 純 success-only reward 對 fresh policy 是 chicken-and-egg(沒成功 → 不學 → 永遠不成功)。
# 2026-06-15: 0.5 → 2.0,加大「嘗試 parry」誘因。gate 已 spam-proof(只在 boss
# windup cat 1-5 + attack=3 edge + 非自己動作後搖期觸發),提高權重不會開出 spam
# local optimum;且仍遠低於成功 parry 的 20/50,主梯度依舊指向「擋成功」而非「亂按」。
W_PARRY_ATTEMPT   = 2.0

# v1.18.0 防禦引導（v1.19.0: evade 8→2，降 per-second 主導性）
W_EVADE_PS               = 2.0   # 每秒：boss 在 windup/attacking 且玩家在 range 內、這幀沒挨打 → 加分
W_PUNISH_HIT_DURING_WINDUP = 5.0  # 一次性：boss 有 windup 預警你還挨打 → 額外扣分（獨立於 hurt-cap）

# --- instrumental：per-step 項一律改「每秒速率」，compute_reward 內乘 dt ---
# 為什麼改 dt-based：原本「每步固定值」在遊戲加速 / fps 抖動時，「每秒幾步」
# 會變 → 每秒拿到的 reward 跟著漂。改成「每秒速率 × dt」後不管 1×/2× 加速、
# fps 怎麼抖，每「遊戲秒」拿到的 reward 都一樣 → 加速訓練→1× 跑可完美銜接。
W_TIME_PENALTY_PS = -0.3  # 每秒：時間懲罰（原 -0.01/步 @30Hz）
W_IN_RANGE_PS     = 1.0   # 每秒：處於接戰距離內（v1.19.0: 3→1，降 per-second 主導性）
W_FACE_BOSS_PS    = 0.5   # 每秒：面向 boss（v1.19.0: 1.5→0.5）
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
W_BOSS_ENGAGED = 20.0

def compute_reward(prev: dict | None, cur: dict, use_instrumental: bool = True) -> float:
    r = 0.0

    # ---- base：勝負 ----
    # boss_dead 優先：2 階段 boss 真死才算 win（換階段不會觸發）；與 env 的 won 一致
    if cur.get("done"):
        if cur.get("boss_dead"):
            r += W_WIN
        elif cur.get("php", 1) <= 0:
            r -= W_DEATH

    # ---- sub：血量變化 ----
    if prev is not None:
        dphp = cur.get("php_pct", 1.0) - prev.get("php_pct", 1.0)
        if dphp < 0:
            # v1.20.2: quadratic hurt penalty (取代 linear+cap)
            # dphp ∈ [-1, 0],平方後 ∈ [0, 1] → 自然 bounded、不需 cap
            # 小擊輕、重擊飆,訊號連續可學「招式 A 比招式 B 危險」相對排序
            r -= W_PLAYER_HURT_Q * dphp * dphp
            # step_hurt_penalty 保持 0（簽章相容,env.py 累 0 無作用）
        elif dphp > 0:
            # 2026-06-15 Round 23: heal 成功 → 線性獎勵
            # dphp ∈ (0, 1]:回 1% = +0.35,回 50% = +17.5,殘血滿瓶喝 = +35
            # env.py mask 已擋滿血(php_pct>=99%)attack=5,wasted heal 不會觸發。
            r += W_HEAL_SUCCESS * dphp

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

        # v1.18.0 防禦引導(v2.0.0:遷移到 attack_category)
        # category:0 idle / 1-4 windup(parryable/unparryable/grab/ranged) / 5 attacking
        #          6 phase_change / 7 stunned
        # 「躲過 boss 攻擊」:boss 構成威脅(任一 windup 或攻擊中,category 1-5)、玩家在
        # 接戰範圍內、這幀沒挨打 → +分。用 prev 的 threat 訊號(避免本幀 transition 邊界誤判)。
        prev_cat = int(prev.get("attack_category", 0))
        boss_threat = 1 <= prev_cat <= 5
        in_range    = abs(prev.get("bdx", 1e9)) < ENGAGE_RANGE
        got_hit     = dphp < 0
        if boss_threat and in_range and not got_hit:
            dt = cur.get("dt", 1.0 / 30.0)
            dt = min(max(dt, 0.0), 0.2)
            r += W_EVADE_PS * dt

        # 「被預警攻擊命中」:boss 有任一 windup 預警(category 1-4)你還挨打 → 額外扣分
        # (獨立 stream,不進 hurt-cap)。attacking (5) 沒 windup 預警階段,不算「被預警還挨打」。
        if 1 <= prev_cat <= 4 and got_hit:
            r -= W_PUNISH_HIT_DURING_WINDUP

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

    return r
