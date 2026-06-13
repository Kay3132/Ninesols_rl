"""
env.py —— NineSolsEnv：標準 Gymnasium 環境

時序模型：free-running 30Hz。遊戲不暫停，每個 step 約 33ms 遊戲時間。
  step()  送 action → 收下一筆 state（自動取最新）→ 算 reward
  reset() 等到玩家可操作且存活（重生完成）
"""
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .bridge import GameBridge
from .rewards import (compute_reward, W_BOSS_ENGAGED, W_TRUNCATION,
                      BURST_WINDOW_STEPS, W_BURST_PENALTY, W_SPEED_BONUS)

# observation 各維度的正規化尺度
POS_SCALE = 1000.0   # 相對座標
VEL_SCALE = 200.0    # 速度
INJ_SCALE = 100.0    # 內傷

# v2.0.0: boss 攻擊類別 one-hot 維度（取代 boss_windup/boss_attacking 兩個 boolean）
# 8 個通用 category 在 Plugin.cs _bossFsmCategories 用同樣編號定義。
# 0 idle / 1 parryable_windup / 2 unparryable_windup / 3 grab_windup
# 4 ranged_windup / 5 attacking / 6 phase_change / 7 stunned
N_ATTACK_CATEGORIES = 8

# v2.0.0: per-boss attack ID one-hot 容量(跟 Plugin.cs MAX_ATTACK_IDS 同步)。
# 給 policy 細粒度區分同 boss 內不同 attack# 的能力。slot 語意 per-boss 各自定義,
# 未登錄 FSM → mod 端送 -1 → 這裡 one-hot 全 0。
N_ATTACK_IDS = 20

# ---- curriculum：弱化 boss，讓「贏」變得可達 ----
# boss 有效 HP = boss_hp_scale × maxHP。從 START 開始，最近 WINDOW 場勝率
# ≥ WIN_RATE 就 +STEP，直到 MAX(1.0)。勝率不到就停在原難度（自我修正）。
CURRICULUM_START    = 0.15
CURRICULUM_MAX      = 1.00
CURRICULUM_STEP     = 0.15
CURRICULUM_WINDOW   = 20
CURRICULUM_WIN_RATE = 0.60

# v1.19.0: Boss HP 里程碑 reward —— 每階段獨立追蹤。
# 用途：在 episode 中段塞 outcome-correlated 訊號，補 gamma=0.98 horizon (~4s) 太短、
# WIN 訊號傳不到 episode 中段的問題。
# 注意 bhp_pct 是「對當前 curriculum cap 取比例」(mod 端 remain/cap)，所以 0.75/0.5/0.25
# 在任何 curriculum scale 下都代表「此階段血條剩餘 75%/50%/25%」，跨難度行為一致。
# 多階段 boss 用 PHASE_CHANGE_DELTA 偵測新血條 → 重設 _milestones_hit、下一階段重新累積。
MILESTONES = [
    (0.75, 15.0),    # 開始造成傷害
    (0.50, 25.0),    # 半血線
    (0.25, 40.0),    # 殘血線
    (0.001, 60.0),   # 此階段架勢清零（鼓勵 push 過 phase transition）
]
PHASE_CHANGE_DELTA = 0.5  # bhp_pct 一步上升 > 此值 → 視為 phase change (0→1 躍升)


class NineSolsEnv(gym.Env):
    metadata = {"render_modes": []}

    # v2.0.0: 25 → 45 (+20 per-boss attack ID one-hot)
    # rebuilt 1a41c9a/04fa407 2026-06-13: 45 → 56。+11 新 obs(由部署 dll 還原確認:
    # shield ×2 + hazard ×7 + attack_nearest ×2)。順序為災後 best-effort 重建,
    # 對齊 3.8M 步 checkpoint(policy 第一層 = 64×56)。
    OBS_DIM = 17 + N_ATTACK_CATEGORIES + N_ATTACK_IDS + 11   # = 17 + 8 + 20 + 11 = 56

    def __init__(self, host: str = "127.0.0.1", port: int = 19271,
                 max_steps: int = 1500):
        super().__init__()
        self.bridge = GameBridge(host, port)
        self.max_steps = max_steps

        # action：move(停/左/右) × dodge(無/跳/衝刺) × attack(無/近戰/遠程/格檔/貼符/喝藥)
        self.action_space = spaces.MultiDiscrete([3, 3, 6])

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.OBS_DIM,), dtype=np.float32)

        self._steps = 0
        self._prev_raw: dict | None = None
        # v1.20.2: hurt penalty 改 quadratic 後不再需要 cumulative_hurt cap 追蹤
        self._boss_engaged = False  # 本局是否已首次接戰 boss（一次性 bonus 用）
        self._milestones_hit: set = set()  # 本階段已觸發的 bhp_pct 門檻（phase change 會重設）
        self._effective_max_steps = max_steps  # v1.20: 本 episode 有效上限（reset 時依 curriculum 調整）
        self._last_hit_step = -999  # v1.20.1: 上次挨打的 step（burst penalty 用）
        # v1.20.1: train.py 啟動時的首次 reset() 會 Suicide() 一次當「測試」，
        # 之後跑出的第一個 episode 不計入 curriculum stats（避免初始設置狀態污染學習）
        self._first_episode_done = False
        # v2.0.0: 本 episode 已進入的 boss phase 數(reset=1,phase change 偵測時 +1)。
        # SIL NEAR 判定用 phase_count >= total_phases 決定是否在最終階段。
        self._phase_count = 1
        self._episode_return = 0.0   # 累計 raw reward,ep-end log 用
        # rebuilt 2026-06-14: 本回合開場的累積 parry 計數（ep-end 取 delta = 本場次數）
        self._ep_start_parry_atm = 0   # parry_enter_count（嘗試）
        self._ep_start_parry_ok = 0    # parry_count（成功）

        # curriculum 狀態
        self._boss_hp_scale = CURRICULUM_START
        self._recent_outcomes: deque = deque(maxlen=CURRICULUM_WINDOW)  # True=勝

    # ---- 編解碼 ----
    def _encode_obs(self, s: dict) -> np.ndarray:
        # v2.0.0: boss 攻擊類別 one-hot（取代舊的 boss_windup / boss_attacking 兩個 bit）
        # 未知 boss / 未登錄 FSM → mod 端 fallback 到 0 (idle)，這裡夾範圍防呆。
        cat = int(s.get("attack_category", 0))
        cat = max(0, min(N_ATTACK_CATEGORIES - 1, cat))
        cat_onehot = [1.0 if i == cat else 0.0 for i in range(N_ATTACK_CATEGORIES)]
        # v2.0.0: per-boss attack ID one-hot。mod 端送 -1 = 不在 attack ID 表內 → 全 0。
        # 0 ~ N_ATTACK_IDS-1 才會點亮對應 slot。語意 per-boss(見 Plugin.cs _bossAttackIds)。
        aid = int(s.get("attack_id", -1))
        id_onehot = [0.0] * N_ATTACK_IDS
        if 0 <= aid < N_ATTACK_IDS:
            id_onehot[aid] = 1.0
        return np.array([
            s.get("vx", 0.0) / VEL_SCALE,
            s.get("vy", 0.0) / VEL_SCALE,
            s.get("facing", 1.0),
            1.0 if s.get("grounded") else 0.0,
            s.get("php_pct", 1.0),
            s.get("internal_injury", 0.0) / INJ_SCALE,
            s.get("qi_pct", 0.0),
            1.0 if s.get("boss_present") else 0.0,
            s.get("bdx", 0.0) / POS_SCALE,
            s.get("bdy", 0.0) / POS_SCALE,
            s.get("bvx", 0.0) / VEL_SCALE,
            s.get("bvy", 0.0) / VEL_SCALE,
            s.get("b_facing", 0.0),
            s.get("bhp_pct", 0.0),
            1.0 if s.get("controllable") else 0.0,
            s.get("last_parry_result", 0) / 2.0,    # 0 無 / 0.5 不精確 / 1 精確
            1.0 if s.get("knocked_down") else 0.0,   # 玩家受傷/倒地（可用閃避起身）
            *cat_onehot,                             # v2.0.0: 8 個通用 attack category one-hot
            *id_onehot,                              # v2.0.0: 20 個 per-boss attack ID one-hot
            # rebuilt 2026-06-13: 45→56 的 11 個新維度(欄位由部署 dll 的 Plugin.cs 確認)。
            # 順序為 best-effort 重建,需用 3.8M checkpoint 實跑驗證(見 plan)。
            1.0 if s.get("shield_active") else 0.0,             # idx45 1a41c9a shield
            s.get("shield_hp_pct", 0.0),                        # idx46
            float(s.get("hazard_unparryable", 0)) / 5.0,        # idx47
            s.get("hazard_nearest_dx", 0.0) / POS_SCALE,        # idx48
            s.get("hazard_nearest_dy", 0.0) / POS_SCALE,        # idx49
            float(s.get("hazard_type_explosion", 0)),           # idx50
            float(s.get("hazard_type_damagearea", 0)),          # idx51
            s.get("attack_nearest_dx", 0.0) / POS_SCALE,        # idx52 04fa407 +parrable
            s.get("attack_nearest_dy", 0.0) / POS_SCALE,        # idx53
            float(s.get("hazard_count", 0)) / 5.0,              # idx54 rose R9 tail
            float(s.get("hazard_type_danger", 0)),              # idx55
        ], dtype=np.float32)

    @staticmethod
    def _decode_action(action) -> dict:
        # action = [move, dodge, attack]
        return {
            "move":   int(action[0]),   # 0停 1左 2右
            "dodge":  int(action[1]),   # 0無 1跳 2衝刺
            "attack": int(action[2]),   # 0無 1近戰 2遠程 3格檔 4貼符 5喝藥
        }

    # ---- curriculum ----
    def _maybe_advance_curriculum(self):
        """上一批 episode 勝率達標 → 提升 boss HP 倍率（curriculum 升級）。"""
        if (self._boss_hp_scale >= CURRICULUM_MAX
                or len(self._recent_outcomes) < CURRICULUM_WINDOW):
            return
        win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        if win_rate >= CURRICULUM_WIN_RATE:
            self._boss_hp_scale = min(self._boss_hp_scale + CURRICULUM_STEP,
                                      CURRICULUM_MAX)
            self._recent_outcomes.clear()
            print(f"[curriculum] 近 {CURRICULUM_WINDOW} 場勝率 {win_rate:.0%} 達標 "
                  f"→ boss HP 倍率提升到 {self._boss_hp_scale:.2f}")

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self._prev_raw = None

        # curriculum：上一批 episode 勝率達標 → 提升難度
        self._maybe_advance_curriculum()

        # v1.20: max_steps 隨 curriculum 等比放大，讓 truncation 永遠 = 真失敗，
        # 不會在 scale 升高後因為「時間天花板太低」誤罰認真打的 agent。
        # scale=0.10: 2800 (~93s)、scale=0.50: 6000 (~200s)、scale=1.00: 10000 (~333s)
        self._effective_max_steps = int(self.max_steps * (1.0 + 4.0 * self._boss_hp_scale))

        # 要求 mod 重置：呼叫 Player.Suicide() → memory mode 死亡會自動重啟此場 boss 戰
        self.bridge.send_action({"reset": True, "boss_hp_scale": self._boss_hp_scale})

        # 等死亡 + 戰鬥重啟完成、玩家可操作且存活
        # （mod 在 reset 後 3 秒內強制 controllable=false → reset 迴圈不會在死亡前提早 break）
        while True:
            s = self.bridge.recv_state()
            if s.get("controllable") and s.get("php", 0) > 0 and not s.get("done"):
                break
            # 死亡/重啟過場中：保持停止；boss_hp_scale 一起送讓 mod 知道當前難度
            self.bridge.send_action({"move": 0, "boss_hp_scale": self._boss_hp_scale})

        self._milestones_hit = set()  # 里程碑門檻歸零（新 episode + phase 1）
        self._last_hit_step = -999  # v1.20.1: burst penalty 狀態歸零
        self._phase_count = 1  # v2.0.0: 新 episode 從 phase 1 起算
        self._episode_return = 0.0  # 累計 raw reward 歸零
        # boss 若在 reset 當下已在場 → 視為本局已接戰（不發 bonus、不啟動資源限制）
        self._boss_engaged = bool(s.get("boss_present"))
        # rebuilt 2026-06-14: 記下開場的累積 parry 計數（遊戲端累積值，ep-end 取差）
        self._ep_start_parry_atm = int(s.get("parry_enter_count", 0))
        self._ep_start_parry_ok = int(s.get("parry_count", 0))
        self._prev_raw = s
        # 診斷：boss 起始 HP vs 未縮放滿血。ratio 應 ≈ scale（若縮放生效）
        _bhp = s.get("bhp", 0.0)
        _bmax = s.get("bhp_max", 0.0)
        _ratio = (_bhp / _bmax) if _bmax > 0 else 0.0
        print(f"[ep-start] Boss_HP_Scale={self._boss_hp_scale:.0%}")
        return self._encode_obs(s), {"raw": s, "boss_hp_scale": self._boss_hp_scale}

    def step(self, action):
        act = self._decode_action(action)
        act["boss_hp_scale"] = self._boss_hp_scale   # 側通道：告訴 mod 當前 curriculum 難度

        # 接戰前禁用消耗性資源動作：長程(ammo)/喝藥(potion) 都是有限數量且不在 obs 裡，
        # boss 戰前的長游走會把它們耗光。用 sticky 的 _boss_engaged（非即時 boss_present）
        # → boss 偵測 flicker 免疫，一旦接戰過就永久解除、戰鬥中絕不誤擋。
        if not self._boss_engaged and act["attack"] in (2, 3, 5):
            act["attack"] = 0

        # rebuilt 4f38deb 2026-06-13: 滿血(php_pct >= 99%)時禁止喝藥(attack=5)，
        # 省藥給 phase 2 close-out。獨立 gate，與 _boss_engaged 無關、戰鬥中也生效。
        # 用 self._prev_raw 的 php_pct（step 開頭尚未收新 state）。
        if act["attack"] == 5 and self._prev_raw is not None \
                and self._prev_raw.get("php_pct", 1.0) >= 0.99:
            act["attack"] = 0

        self.bridge.send_action(act)
        s = self.bridge.recv_state()
        self._steps += 1

        # v1.20.2: quadratic hurt penalty (取代 linear+cap),compute_reward 改單一回傳
        reward = compute_reward(self._prev_raw, s)

        # v1.20.1: burst damage penalty —— 短時間內連續挨打額外扣分（不進 hurt-cap）
        # 解 W_MAX_HURT_PENALTY 上限後致命一擊沒訊號的問題（phase 2 close-out 死亡）
        if self._prev_raw is not None:
            dphp = s.get("php_pct", 1.0) - self._prev_raw.get("php_pct", 1.0)
            if dphp < 0:  # 這幀挨打
                if self._steps - self._last_hit_step < BURST_WINDOW_STEPS:
                    reward -= W_BURST_PENALTY
                self._last_hit_step = self._steps

        # 一次性：本局首次讓 boss 出現/接戰 → 給接戰 bonus
        if s.get("boss_present") and not self._boss_engaged:
            reward += W_BOSS_ENGAGED
            self._boss_engaged = True
            #print(f"[engage] step={self._steps} 首次接戰 boss (+{W_BOSS_ENGAGED:.0f})")

        # v1.19.0: Boss HP 里程碑（per-phase，farming-proof）
        # 中段 outcome-correlated reward → 補 gamma=0.98 horizon 太短的 win 訊號傳遞問題
        if (self._prev_raw is not None and s.get("boss_present")
                and self._prev_raw.get("boss_present")):
            prev_bhp = self._prev_raw.get("bhp_pct", 1.0)
            cur_bhp  = s.get("bhp_pct", 1.0)
            # Phase change 偵測：清零後新的滿血條（bhp_pct 0→1 躍升）→ 重設讓下一階段
            # 重新累積里程碑（每階段獨立 ~+140 outcome 訊號）。boss 自癒幅度小、不會誤觸發。
            if cur_bhp - prev_bhp > PHASE_CHANGE_DELTA and self._milestones_hit:
                #print(f"[milestone] step={self._steps} phase change "
                #      f"(bhp_pct {prev_bhp:.2f}→{cur_bhp:.2f}) 里程碑重設")
                self._phase_count += 1  # v2.0.0: SIL NEAR 判定要知道目前在第幾階段
                print(f"[milestone] step={self._steps} Phase change → phase {self._phase_count}")
                self._milestones_hit.clear()
            # 觸發跨越本階段門檻的里程碑（一步可能跨多個）
            for threshold, bonus in MILESTONES:
                if (threshold not in self._milestones_hit
                        and prev_bhp >= threshold > cur_bhp):
                    reward += bonus
                    self._milestones_hit.add(threshold)
                    #print(f"[milestone] step={self._steps} bhp<{threshold:.2f} (+{bonus:.0f})")

        terminated = bool(s.get("done", False))
        truncated = self._steps >= self._effective_max_steps
        # v1.20: 撞 max_steps 沒贏沒死 → 額外懲罰，封堵「拖時間到 truncation 免死亡懲罰」exploit
        if truncated and not terminated:
            reward -= W_TRUNCATION
        # rebuilt 2026-06-14: 快速通關 bonus —— 只在真 WIN(boss 真死)觸發。
        # time_fraction 用 _effective_max_steps（已隨 curriculum 放大）→ bonus 自動依難度縮放。
        # linear：越快清越肥（用掉 30% 時間 → 0.7×；80% → 0.2×）。
        if terminated and s.get("boss_dead"):
            time_fraction = self._steps / max(1, self._effective_max_steps)
            reward += W_SPEED_BONUS * (1.0 - time_fraction)
        self._prev_raw = s
        self._episode_return += float(reward)  # 累計 raw reward,ep-end log 用

        # episode 結束 → 記錄勝負給 curriculum（勝 = boss 真死，2 階段全清）
        ep_summary = None
        if terminated or truncated:
            won = bool(terminated and s.get("boss_dead"))
            # rebuilt 2026-06-14: 本場 parry 次數（遊戲端累積值取 delta）+ chi / phase / bhp。
            parry_atm = int(s.get("parry_enter_count", 0)) - self._ep_start_parry_atm
            parry_ok  = int(s.get("parry_count", 0)) - self._ep_start_parry_ok
            total_phases = int(s.get("total_phases", 1))
            # ep-end 尾巴：每欄獨立 key=value（CSV 友善，不用斜線）
            ep_extra = (f"| parry={parry_ok} atm={parry_atm} "
                        f"chi={s.get('chi', 0):.0f} phase={self._phase_count}/{total_phases} "
                        f"| bhp={s.get('bhp', 0):.0f} bhp%={s.get('bhp_pct', 0):.0%} ")
            # per-episode CSV 用（EpisodeCSVCallback 會再補 total_timesteps）
            ep_summary = {
                "scale": round(float(self._boss_hp_scale), 4), "won": int(won),
                "steps": self._steps, "ret": round(float(self._episode_return), 2),
                "parry_ok": parry_ok, "parry_atm": parry_atm,
                "chi": int(s.get("chi", 0)), "phase": self._phase_count,
                "total_phases": total_phases,
                "bhp": round(float(s.get("bhp", 0)), 1),
                "bhp_pct": round(float(s.get("bhp_pct", 0)), 4),
            }
            if not self._first_episode_done:
                # v1.20.1: 首次 episode 是 train.py 啟動 Suicide 後的「測試 episode」，
                # 不計入 curriculum stats 避免初始狀態污染學習
                self._first_episode_done = True
                print(f"[ep-end] scale={self._boss_hp_scale:.2f} "
                      f"{'WIN' if won else 'lose'} steps={self._steps} "
                      f"ret={self._episode_return:.1f} "
                      f"{ep_extra}| [首次 episode, 不計入 curriculum]")
            else:
                self._recent_outcomes.append(won)
                n = len(self._recent_outcomes)
                wr = sum(self._recent_outcomes) / n if n else 0.0
                print(f"[ep-end] scale={self._boss_hp_scale:.2f} "
                      f"{'WIN' if won else 'lose'} steps={self._steps} "
                      f"ret={self._episode_return:.1f} "
                      f"{ep_extra}| 近{n}場勝率 {wr:.0%}")

        # v2.0.0: phase_count 是 env 端追蹤的,塞進 s 一起給 SIL recorder
        # (total_phases 已由 mod 端 JSON 帶在 s 裡,不用額外注入)
        s["phase_count"] = self._phase_count
        info = {"raw": s, "boss_hp_scale": self._boss_hp_scale,
                "raw_reward": float(reward)}  # SIL recorder 抓未 normalize 的 reward
        if ep_summary is not None:
            info["ep_summary"] = ep_summary   # rebuilt 2026-06-14: per-episode CSV 用
        return (self._encode_obs(s), reward, terminated, truncated, info)

    def close(self):
        self.bridge.close()
