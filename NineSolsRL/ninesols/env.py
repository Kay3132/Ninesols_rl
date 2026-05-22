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
from .rewards import compute_reward, W_BOSS_ENGAGED

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

        # 等到玩家重生完成且可操作
        while True:
            s = self.bridge.recv_state()
            if s.get("controllable") and s.get("php", 0) > 0 and not s.get("done"):
                break
            # 死亡/過場中：保持停止；boss_hp_scale 一起送讓 mod 知道當前難度
            self.bridge.send_action({"move": 0, "boss_hp_scale": self._boss_hp_scale})

        self.cumulative_hurt = 0.0  # 每次重置遊戲時歸零
        self._boss_engaged = False  # 接戰旗標每局歸零
        self._prev_raw = s
        # 診斷：boss 起始 HP vs 未縮放滿血。ratio 應 ≈ scale（若縮放生效）
        _bhp = s.get("bhp", 0.0)
        _bmax = s.get("bhp_max", 0.0)
        _ratio = (_bhp / _bmax) if _bmax > 0 else 0.0
        print(f"[ep-start] scale={self._boss_hp_scale:.2f} | "
              f"boss bhp={_bhp:.0f}/{_bmax:.0f} (ratio={_ratio:.2f})")
        return self._encode_obs(s), {"raw": s, "boss_hp_scale": self._boss_hp_scale}

    def step(self, action):
        act = self._decode_action(action)
        act["boss_hp_scale"] = self._boss_hp_scale   # 側通道：告訴 mod 當前 curriculum 難度
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
            print(f"[engage] step={self._steps} 首次接戰 boss (+{W_BOSS_ENGAGED:.0f})")

        terminated = bool(s.get("done", False))
        truncated = self._steps >= self.max_steps
        self._prev_raw = s

        # episode 結束 → 記錄勝負給 curriculum（勝 = boss 被打死）
        if terminated or truncated:
            won = bool(terminated and s.get("boss_present")
                       and s.get("bhp", 1.0) <= 0)
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
