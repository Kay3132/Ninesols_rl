"""
env.py —— NineSolsEnv：標準 Gymnasium 環境

時序模型：free-running 10Hz。遊戲不暫停，每個 step 約 100ms 遊戲時間。
  step()  送 action → 收下一筆 state（自動取最新）→ 算 reward
  reset() 等到玩家可操作且存活（重生完成）
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .bridge import GameBridge
from .rewards import compute_reward

# observation 各維度的正規化尺度
POS_SCALE = 1000.0   # 相對座標
VEL_SCALE = 200.0    # 速度
INJ_SCALE = 100.0    # 內傷


class NineSolsEnv(gym.Env):
    metadata = {"render_modes": []}

    OBS_DIM = 18

    def __init__(self, host: str = "127.0.0.1", port: int = 19271,
                 max_steps: int = 2000):
        super().__init__()
        self.bridge = GameBridge(host, port)
        self.max_steps = max_steps

        # action：move(停/左/右) × dodge(無/跳/衝刺) × attack(無/近戰/遠程/格檔/貼符)
        self.action_space = spaces.MultiDiscrete([3, 3, 5])

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.OBS_DIM,), dtype=np.float32)

        self._steps = 0
        self._prev_raw: dict | None = None

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
        ], dtype=np.float32)

    @staticmethod
    def _decode_action(action) -> dict:
        # action = [move, dodge, attack]
        return {
            "move":   int(action[0]),   # 0停 1左 2右
            "dodge":  int(action[1]),   # 0無 1跳 2衝刺
            "attack": int(action[2]),   # 0無 1近戰 2遠程 3格檔 4貼符
        }

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self._prev_raw = None

        # 等到玩家重生完成且可操作
        while True:
            s = self.bridge.recv_state()
            if s.get("controllable") and s.get("php", 0) > 0 and not s.get("done"):
                break
            self.bridge.send_action({"move": 0})   # 死亡/過場中：保持停止

        self._prev_raw = s
        return self._encode_obs(s), {"raw": s}

    def step(self, action):
        self.bridge.send_action(self._decode_action(action))
        s = self.bridge.recv_state()
        self._steps += 1

        reward = compute_reward(self._prev_raw, s)
        terminated = bool(s.get("done", False))
        truncated = self._steps >= self.max_steps
        self._prev_raw = s

        return self._encode_obs(s), reward, terminated, truncated, {"raw": s}

    def close(self):
        self.bridge.close()
