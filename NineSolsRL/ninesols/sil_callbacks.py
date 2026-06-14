"""
sil_callbacks.py —— SIL 用的兩個 SB3 callback。

1. SILEpisodeRecorderCallback —— 每個 env step 抓 (obs_t, a_t, r_t_raw),
   episode 結束 flush 整場到 SILBuffer。obs 是「post-VecNormalize」的版本
   (從 model._last_obs 拿,跟 policy 看到的一致);reward 從 env.py info
   裡的 `raw_reward` 抓未 normalize 值,buffer 內存 raw return,sample 時
   再除以 current ret_std 轉到 value head 尺度。

2. SILBufferCheckpointCallback —— 繼承 SB3 CheckpointCallback,在同一個
   `n_calls % save_freq == 0` 節奏多存一份 sil_buffer_<N>_steps.pkl,
   配合 PPO zip / vecnormalize pkl 三者同步存檔,reset 後可一起載回。
"""
from __future__ import annotations

import os
import csv  # rebuilt 9046049: per-episode CSV
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from .sil_buffer import SILBuffer


class SILEpisodeRecorderCallback(BaseCallback):
    """單 env(DummyVecEnv num_envs=1)版本。"""

    # v2.0.0 NEAR-WIN 判定門檻:final phase 殘血比例 < 此值 → 算 NEAR(scale-invariant)
    # 舊版用絕對 HP=100,在 scale=0.10 時 phase 1 殘血會誤判 NEAR。
    NEAR_WIN_BHP_PCT = 0.05

    def __init__(self, sil_buffer: SILBuffer,
                 verbose: int = 0):
        super().__init__(verbose)
        self.buffer = sil_buffer

        # per-episode 收集器
        self._obs_seq: list[np.ndarray] = []
        self._act_seq: list[np.ndarray] = []
        self._return_raw: float = 0.0

        # 是 Suicide 後的「測試 episode」(前 ~90 步 controllable=false),不 flush
        # 進 buffer,避免污染 SIL 學到「這些 obs 下任何 action 都 ok」
        self._first_episode_done = False

        # v1.21.1: 監看 curriculum advance(env.py 改變 boss_hp_scale 時觸發
        # buffer purge_non_wins + SIL coef decay)。None = 還沒收到第一個 info
        self._last_scale: Optional[float] = None

    def _flush_episode(self, info: dict) -> None:
        if not self._obs_seq:
            return

        # 首次 episode 直接清掉不 admit(對齊 env.py:226 的 curriculum 排除)
        if not self._first_episode_done:
            self._first_episode_done = True
            if self.verbose >= 1:
                print(f"[sil] 首次 episode 不入 buffer (len={len(self._obs_seq)} "
                      f"ret={self._return_raw:.1f})")
            self._obs_seq.clear()
            self._act_seq.clear()
            self._return_raw = 0.0
            return

        raw = info.get("raw", {}) if isinstance(info, dict) else {}
        is_truncated = bool(info.get("TimeLimit.truncated", False))
        # WIN = boss 真死(N 階段全清)。truncation 不算贏。
        is_win = bool(raw.get("boss_dead", False)) and not is_truncated
        # v2.0.0 NEAR-WIN:必須三條件 AND
        #   1. 非 WIN 且非 truncation(truncation 已被 W_TRUNCATION 罰,不該再 NEAR admit)
        #   2. 已進入最終階段(N-phase 通用,phase_count >= total_phases)
        #   3. 最終階段殘血比例 < NEAR_WIN_BHP_PCT(scale-invariant)
        # 修了舊版用絕對 HP=100 在 scale=0.10 時 phase 1 殘血誤判 NEAR 的 bug。
        phase_count  = int(raw.get("phase_count", 1))
        total_phases = int(raw.get("total_phases", 1))
        bhp_pct_end  = float(raw.get("bhp_pct", 1.0))
        is_near_win = ((not is_win) and (not is_truncated)
                       and phase_count >= total_phases
                       and bhp_pct_end < self.NEAR_WIN_BHP_PCT)

        # 2026-06-15 Round 21: SIL 只學最終階段(phase_count >= total_phases)trajectory。
        # WIN 隱含最終階段(boss 死了),NEAR 已自帶此條件,HIGH 之前沒擋 → phase 1 ret 新高
        # 也會 admit,SIL 會學「達到 phase 1 終點就死」的垃圾 pattern。加 gate 排除。
        in_final_phase = phase_count >= total_phases
        if not is_win and not in_final_phase:
            if self.verbose >= 1:
                print(f"[sil] skip phase {phase_count}/{total_phases} "
                      f"ret={self._return_raw:.1f}(非最終階段,丟棄)")
            self._obs_seq.clear()
            self._act_seq.clear()
            self._return_raw = 0.0
            return

        obs_arr = np.stack(self._obs_seq, axis=0).astype(np.float32)
        act_arr = np.stack(self._act_seq, axis=0).astype(np.int64)
        admitted = self.buffer.add_episode(
            obs_seq=obs_arr,
            action_seq=act_arr,
            episodic_return=self._return_raw,
            is_win=is_win,
            is_near_win=is_near_win,
        )
        if self.verbose >= 1 and admitted:
            tag = "WIN" if is_win else ("NEAR" if is_near_win else "HIGH")
            print(f"[sil] +episode len={len(self._obs_seq)} "
                  f"ret={self._return_raw:.1f} tag={tag} "
                  f"buf={len(self.buffer)}")

        self._obs_seq.clear()
        self._act_seq.clear()
        self._return_raw = 0.0

    def _on_step(self) -> bool:
        # SB3 在呼叫 _on_step 之前已經跑過 env.step(),但還沒把 _last_obs
        # 換成 new_obs(見 on_policy_algorithm.py:223 vs 255)。所以這時:
        #   self.model._last_obs[0]   = obs_t (產生此 action 的觀察、post-VecNorm)
        #   self.locals["actions"][0] = a_t  (剛送出去的 action,shape (3,))
        #   self.locals["infos"][0]   = info(包含 raw_reward / raw / 等)
        #   self.locals["dones"][0]   = 此 step 後 episode 是否結束
        obs_t = self.model._last_obs[0].copy()   # shape (19,)
        action = self.locals["actions"][0]
        info   = self.locals["infos"][0]
        done   = bool(self.locals["dones"][0])

        # MultiDiscrete: action 已是 shape (3,) int array;Discrete 才會被 reshape
        if action.ndim == 0:
            action = np.asarray([action], dtype=np.int64)
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != SILBuffer.ACT_DIM:
            # 防呆,避免靜默壞掉
            raise RuntimeError(f"unexpected action shape: {action.shape}")

        raw_r = float(info.get("raw_reward", 0.0))

        # v1.21.1: 偵測 curriculum advance(env.py:_maybe_advance_curriculum 改了 scale)
        scale = info.get("boss_hp_scale")
        if scale is not None:
            scale = float(scale)
            if self._last_scale is None:
                self._last_scale = scale
            elif abs(scale - self._last_scale) > 1e-6:
                # 通知 PPOSIL 做 buffer purge + coef decay(model = PPOSIL 實例)
                if hasattr(self.model, "on_curriculum_advance"):
                    self.model.on_curriculum_advance(self._last_scale, scale)
                self._last_scale = scale

        self._obs_seq.append(obs_t)
        self._act_seq.append(action.copy())
        self._return_raw += raw_r

        if done:
            self._flush_episode(info)

        return True


class SILBufferCheckpointCallback(CheckpointCallback):
    """跟 PPO ckpt 同節奏存 SIL buffer。檔名 `sil_buffer_{N}_steps.pkl`。"""

    def __init__(self, sil_buffer: SILBuffer, *args, verbose: int = 0, **kwargs):
        super().__init__(*args, verbose=verbose, **kwargs)
        self.buffer = sil_buffer

    def _on_step(self) -> bool:
        ret = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"sil_buffer_{self.num_timesteps}_steps.pkl",
            )
            self.buffer.save(path)
            if self.verbose >= 1:
                stats = self.buffer.stats()
                print(f"[sil] saved buffer → {path} "
                      f"(size={stats['buffer_size']} wins={stats['buffer_wins']})")
        return ret


def find_latest_sil_buffer(ckpt_dir: str, steps: int) -> Optional[str]:
    """找對應 PPO ckpt 步數的 sil_buffer pkl;沒有回 None。"""
    path = os.path.join(ckpt_dir, f"sil_buffer_{steps}_steps.pkl")
    return path if os.path.exists(path) else None


class EpisodeCSVCallback(BaseCallback):
    """rebuilt 9046049 2026-06-14: 每場 episode 寫一列到 episodes.csv。

    env.py 在 episode 結束時把摘要放進 info["ep_summary"];這裡補上全域步數
    `total_timesteps`(= self.num_timesteps,續訓時從 ckpt 接續、正確)後寫一列。
    檔案放在跟 PPO progress.csv 同一個 logger 資料夾(self.model.logger.dir),
    兩檔都用步數當 key → 可直接 join 對齊「某 PPO iteration 區間的 episode 表現」。
    單 env(DummyVecEnv num_envs=1)版本。
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._writer = None
        self._fh = None
        self._path: Optional[str] = None

    def _ensure_writer(self, summary: dict) -> None:
        if self._writer is not None:
            return
        # logger.dir 在 train.py set_logger 後才有 → 第一場才 lazy-open
        log_dir = getattr(self.model.logger, "dir", None) or "logs"
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "episodes.csv")
        self._fh = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["total_timesteps", *summary.keys()])
        self._fh.flush()
        if self.verbose >= 1:
            print(f"[ep-csv] per-episode metrics → {self._path}")

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = bool(self.locals["dones"][0])
        if done and isinstance(info, dict) and "ep_summary" in info:
            summary = info["ep_summary"]
            self._ensure_writer(summary)
            self._writer.writerow([self.num_timesteps, *summary.values()])
            self._fh.flush()   # 即時落地，訓練中途崩潰也保住已寫的列
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
