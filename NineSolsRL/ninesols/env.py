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
from .rewards import compute_reward, W_BOSS_ENGAGED, W_TRUNCATION

# observation 各維度的正規化尺度
POS_SCALE = 1000.0   # 相對座標
VEL_SCALE = 200.0    # 速度
INJ_SCALE = 100.0    # 內傷

# ---- curriculum：弱化 boss，讓「贏」變得可達 ----
# boss 有效 HP = boss_hp_scale × maxHP。從 START 開始，最近 WINDOW 場勝率
# ≥ WIN_RATE 就 +STEP，直到 MAX(1.0)。勝率不到就停在原難度（自我修正）。
CURRICULUM_START    = 0.10
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

    OBS_DIM = 19

    def __init__(self, host: str = "127.0.0.1", port: int = 19271,
                 max_steps: int = 2000):
        super().__init__()
        self.bridge = GameBridge(host, port)
        self.max_steps = max_steps

        # action：move(停/左/右) × dodge(無/跳/衝刺) × attack(無/近戰/遠程/格檔/貼符/喝藥)
        self.action_space = spaces.MultiDiscrete([3, 3, 6])

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.OBS_DIM,), dtype=np.float32)

        self._steps = 0
        self._prev_raw: dict | None = None
        self.cumulative_hurt = 0.0  # 記錄單局總扣血懲罰
        self._boss_engaged = False  # 本局是否已首次接戰 boss（一次性 bonus 用）
        self._milestones_hit: set = set()  # 本階段已觸發的 bhp_pct 門檻（phase change 會重設）
        self._effective_max_steps = max_steps  # v1.20: 本 episode 有效上限（reset 時依 curriculum 調整）

        # curriculum 狀態
        self._boss_hp_scale = CURRICULUM_START
        self._recent_outcomes: deque = deque(maxlen=CURRICULUM_WINDOW)  # True=勝

    # ---- 編解碼 ----
    def _encode_obs(self, s: dict) -> np.ndarray:
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
            1.0 if s.get("boss_windup") else 0.0,    # boss 攻擊前置（預警）
            1.0 if s.get("boss_attacking") else 0.0, # boss 攻擊中
            1.0 if s.get("knocked_down") else 0.0,   # 玩家受傷/倒地（可用閃避起身）
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

        self.cumulative_hurt = 0.0  # 每次重置遊戲時歸零
        self._milestones_hit = set()  # 里程碑門檻歸零（新 episode + phase 1）
        # boss 若在 reset 當下已在場 → 視為本局已接戰（不發 bonus、不啟動資源限制）
        self._boss_engaged = bool(s.get("boss_present"))
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
        if not self._boss_engaged and act["attack"] in (2, 5):
            act["attack"] = 0

        self.bridge.send_action(act)
        s = self.bridge.recv_state()
        self._steps += 1

        # 將 cumulative_hurt 傳入，並接收這回合實際產生的懲罰
        reward, step_hurt_penalty = compute_reward(self._prev_raw, s, self.cumulative_hurt)

        # 更新累計懲罰
        self.cumulative_hurt += step_hurt_penalty

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
                print(f"[milestone] step={self._steps} phase change "
                      f"(bhp_pct {prev_bhp:.2f}→{cur_bhp:.2f}) 里程碑重設")
                self._milestones_hit.clear()
            # 觸發跨越本階段門檻的里程碑（一步可能跨多個）
            for threshold, bonus in MILESTONES:
                if (threshold not in self._milestones_hit
                        and prev_bhp >= threshold > cur_bhp):
                    reward += bonus
                    self._milestones_hit.add(threshold)
                    print(f"[milestone] step={self._steps} bhp<{threshold:.2f} (+{bonus:.0f})")

        terminated = bool(s.get("done", False))
        truncated = self._steps >= self._effective_max_steps
        # v1.20: 撞 max_steps 沒贏沒死 → 額外懲罰，封堵「拖時間到 truncation 免死亡懲罰」exploit
        if truncated and not terminated:
            reward -= W_TRUNCATION
        self._prev_raw = s

        # episode 結束 → 記錄勝負給 curriculum（勝 = boss 真死，2 階段全清）
        if terminated or truncated:
            won = bool(terminated and s.get("boss_dead"))
            self._recent_outcomes.append(won)
            n = len(self._recent_outcomes)
            wr = sum(self._recent_outcomes) / n if n else 0.0
            print(f"[ep-end] scale={self._boss_hp_scale:.2f} "
                  f"{'WIN' if won else 'lose'} steps={self._steps} "
                  f"| boss_present={s.get('boss_present')} bhp={s.get('bhp', 0):.0f} "
                  f"| 近{n}場勝率 {wr:.0%}")

        return (self._encode_obs(s), reward, terminated, truncated,
                {"raw": s, "boss_hp_scale": self._boss_hp_scale})

    def close(self):
        self.bridge.close()
