"""
sil_buffer.py —— Self-Imitation Learning(Oh et al. 2018)transition buffer。

SIL 的核心是「把過去 high-return episode 存下來,SIL update 用
`-log π(a|s) · max(R - V(s), 0)` 反向傳播」。本檔只提供純資料層,
不依賴 SB3 / torch,方便獨立單測 + pickle 持久化。

存的 transition 欄位:
    obs       np.float32[19]  —— **已 VecNormalize** 的 obs(讓 policy update 一致)
    action    np.int64[3]     —— MultiDiscrete([3,3,6]) 三維動作
    ret_raw   float32         —— 未 normalize 的 episodic return(sample 時再除以
                                  current ret_std 轉到 value head 尺度)
    is_win    bool            —— 該 episode 是否勝出(WIN 受保護不被 FIFO evict)

Admission(三選一即收整場 episode):
  1. WIN
  2. Near-win(`bhp_at_end < 100`,專攻 phase 2 close-out)
  3. 高回報(episodic_return > 最近 N 場勝出 admit 的最大值)

Eviction(超過 cap 時):FIFO,但 WIN transitions 受保護;只有當整個 buffer
都是 WIN 才會動到它們。
"""
from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np


class SILBuffer:
    # v2.0.0: 跟 env.NineSolsEnv.OBS_DIM 同步(17 base + 8 attack_category one-hot)
    OBS_DIM = 25
    ACT_DIM = 3   # MultiDiscrete([3, 3, 6])

    def __init__(self, max_transitions: int = 100_000,
                 admit_top_window: int = 20):
        self.max_transitions = int(max_transitions)
        self.admit_top_window = int(admit_top_window)

        # 平面存放,index 對齊
        self._obs: list[np.ndarray] = []        # 每筆 shape (19,) float32
        self._act: list[np.ndarray] = []        # 每筆 shape (3,)  int64
        self._ret: list[float]      = []        # raw episodic return
        self._win: list[bool]       = []

        # 最近 N 場「有被 admit」的 episode return —— 用來判斷新一場是否夠高
        self._recent_returns: deque[float] = deque(maxlen=admit_top_window)

        self._n_episodes_seen = 0
        self._n_episodes_admitted = 0
        self._n_wins_admitted = 0

    # ---- API ----
    def __len__(self) -> int:
        return len(self._obs)

    @property
    def n_wins(self) -> int:
        # transition-level WIN 計數
        return int(sum(self._win))

    @property
    def n_episodes_admitted(self) -> int:
        return self._n_episodes_admitted

    def should_admit(self, episodic_return: float,
                     is_win: bool, is_near_win: bool) -> bool:
        if is_win or is_near_win:
            return True
        if not self._recent_returns:
            return True   # 冷啟動,先放幾場進來
        return episodic_return > max(self._recent_returns)

    def add_episode(self, obs_seq: np.ndarray, action_seq: np.ndarray,
                    episodic_return: float, is_win: bool,
                    is_near_win: bool = False) -> bool:
        """嘗試把整場 episode 加入 buffer。回傳是否被 admit。"""
        self._n_episodes_seen += 1
        if not self.should_admit(episodic_return, is_win, is_near_win):
            return False

        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        action_seq = np.asarray(action_seq, dtype=np.int64)
        if obs_seq.ndim != 2 or obs_seq.shape[1] != self.OBS_DIM:
            raise ValueError(f"obs_seq shape {obs_seq.shape}, want (T, {self.OBS_DIM})")
        if action_seq.ndim != 2 or action_seq.shape[1] != self.ACT_DIM:
            raise ValueError(f"action_seq shape {action_seq.shape}, want (T, {self.ACT_DIM})")
        if len(obs_seq) != len(action_seq):
            raise ValueError("obs_seq / action_seq length mismatch")

        T = len(obs_seq)
        for i in range(T):
            self._obs.append(obs_seq[i].copy())
            self._act.append(action_seq[i].copy())
            self._ret.append(float(episodic_return))
            self._win.append(bool(is_win))

        self._recent_returns.append(float(episodic_return))
        self._n_episodes_admitted += 1
        if is_win:
            self._n_wins_admitted += 1

        self._evict_to_cap()
        return True

    def _evict_to_cap(self) -> None:
        """FIFO,優先 evict 非 WIN transition;buffer 全 WIN 才動到 WIN。"""
        if len(self._obs) <= self.max_transitions:
            return

        overflow = len(self._obs) - self.max_transitions

        # 先找最舊的非 WIN index,從那邊砍
        # 簡單策略:linear scan;FIFO 對應 list 開頭,所以從左往右找非 WIN
        # 砍掉(or 全 WIN 才砍最舊)
        to_drop: list[int] = []
        n = len(self._obs)
        # pass 1: 收集最舊的非 WIN
        for i in range(n):
            if len(to_drop) >= overflow:
                break
            if not self._win[i]:
                to_drop.append(i)
        # 還不夠 → 全 WIN buffer 滿了,只能砍最舊的 WIN
        if len(to_drop) < overflow:
            for i in range(n):
                if len(to_drop) >= overflow:
                    break
                if i not in to_drop:
                    to_drop.append(i)

        # 從大 index 往回砍,避免 shift
        drop_set = set(to_drop)
        self._obs = [x for i, x in enumerate(self._obs) if i not in drop_set]
        self._act = [x for i, x in enumerate(self._act) if i not in drop_set]
        self._ret = [x for i, x in enumerate(self._ret) if i not in drop_set]
        self._win = [x for i, x in enumerate(self._win) if i not in drop_set]

    def sample(self, batch_size: int, prioritized: bool = True,
               current_value_fn=None,
               candidate_multiplier: int = 4,
               priority_eps: float = 1e-3,
               rng: np.random.Generator | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """回傳 (obs[B,19], actions[B,3], returns_raw[B])。

        Prioritized:先均勻抽 candidate(batch_size × multiplier)筆,
        用 current_value_fn 估 V(s),按 `max(R - V, 0) + eps` 權重再從中抽 B 筆。
        這樣每次 sample 只 forward candidate 大小,而非整個 buffer。
        """
        n = len(self._obs)
        if n == 0:
            raise RuntimeError("SILBuffer empty, cannot sample")

        rng = rng if rng is not None else np.random.default_rng()

        if prioritized and current_value_fn is not None and n > batch_size:
            cand_n = min(n, batch_size * candidate_multiplier)
            cand_idx = rng.choice(n, size=cand_n, replace=False)
            cand_obs = np.stack([self._obs[i] for i in cand_idx], axis=0)
            cand_ret = np.array([self._ret[i] for i in cand_idx], dtype=np.float32)
            v_pred = current_value_fn(cand_obs).astype(np.float32)
            # 注意:current_value_fn 預期回傳 value head 尺度的 V;ret 在 buffer 是 raw。
            # 這裡只用「相對」權重(max(R-V,0)+eps),所以尺度不一致也僅影響權重分佈,
            # 不影響 SIL update 的 advantage(那邊另外做 ret/ret_std 轉換)。
            adv = np.maximum(cand_ret - v_pred, 0.0) + priority_eps
            probs = adv / adv.sum()
            pick = rng.choice(cand_n, size=batch_size, replace=True, p=probs)
            sel = cand_idx[pick]
        else:
            sel = rng.integers(0, n, size=batch_size)

        obs = np.stack([self._obs[i] for i in sel], axis=0).astype(np.float32)
        act = np.stack([self._act[i] for i in sel], axis=0).astype(np.int64)
        ret = np.array([self._ret[i] for i in sel], dtype=np.float32)
        return obs, act, ret

    def purge_non_wins(self) -> int:
        """v1.21.1: curriculum advance 時呼叫 —— 砍掉所有非 WIN transitions,
        保留 WIN(WIN 在任何難度下都是完整成功軌跡,對 phase 2 close-out 仍有價值)。

        順便 clear `_recent_returns` —— 上一難度的 return 對新難度的 admit 判斷
        無參考價值(雖然 reward shape 大致 invariant,episode 長度差異會讓 return
        分佈漂)。回傳被砍掉的 transition 數量。
        """
        n_before = len(self._obs)
        keep_idx = [i for i in range(n_before) if self._win[i]]
        self._obs = [self._obs[i] for i in keep_idx]
        self._act = [self._act[i] for i in keep_idx]
        self._ret = [self._ret[i] for i in keep_idx]
        self._win = [self._win[i] for i in keep_idx]
        self._recent_returns.clear()
        return n_before - len(self._obs)

    def stats(self) -> dict:
        return {
            "buffer_size": len(self._obs),
            "buffer_wins": self.n_wins,
            "episodes_seen": self._n_episodes_seen,
            "episodes_admitted": self._n_episodes_admitted,
            "wins_admitted": self._n_wins_admitted,
            "mean_return": float(np.mean(self._ret)) if self._ret else 0.0,
            "max_return": float(np.max(self._ret)) if self._ret else 0.0,
        }

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "max_transitions": self.max_transitions,
            "admit_top_window": self.admit_top_window,
            # 用 numpy stack 壓得緊一點;空 buffer 用 None
            "obs": np.stack(self._obs, axis=0) if self._obs else None,
            "act": np.stack(self._act, axis=0) if self._act else None,
            "ret": np.asarray(self._ret, dtype=np.float32),
            "win": np.asarray(self._win, dtype=np.bool_),
            "recent_returns": list(self._recent_returns),
            "n_episodes_seen": self._n_episodes_seen,
            "n_episodes_admitted": self._n_episodes_admitted,
            "n_wins_admitted": self._n_wins_admitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SILBuffer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        buf = cls(
            max_transitions=state["max_transitions"],
            admit_top_window=state["admit_top_window"],
        )
        if state["obs"] is not None:
            buf._obs = [row.copy() for row in state["obs"]]
            buf._act = [row.copy() for row in state["act"]]
        buf._ret = [float(x) for x in state["ret"].tolist()]
        buf._win = [bool(x) for x in state["win"].tolist()]
        buf._recent_returns = deque(state["recent_returns"],
                                    maxlen=buf.admit_top_window)
        buf._n_episodes_seen = int(state["n_episodes_seen"])
        buf._n_episodes_admitted = int(state["n_episodes_admitted"])
        buf._n_wins_admitted = int(state["n_wins_admitted"])
        return buf
