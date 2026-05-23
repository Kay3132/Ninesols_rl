"""
ppo_sil.py —— PPO + Self-Imitation Learning(Oh et al. 2018)。

子類 SB3 PPO,覆寫 `train()`:跑完 vanilla PPO 更新後,再從 SILBuffer 抽
N_SIL_UPDATES 個 minibatch 做加權 BC + value update:

    A = clamp(R - V(s), 0, adv_clip)
    policy_loss = -log π(a|s) · A
    value_loss  = 0.5 · A² · I[A>0]
    sil_loss    = coef · (policy_loss + value_coef · value_loss)

「自動退化」:當 PPO value head 追上某筆舊軌跡的 return → A=0 → 該樣本
不再貢獻梯度 → 不會把 policy 拉回早期次優模式。

VecNormalize 處理:buffer 存 raw R,這裡用當前 env.ret_rms.var 把 R 轉到
value head 看的尺度(`R / sqrt(var)`),確保 R 與 V(s) 同尺度。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from .sil_buffer import SILBuffer


DEFAULT_SIL_CFG = dict(
    n_updates=4,         # 每次 PPO 更新後跑幾次 SIL minibatch
    batch_size=256,
    coef=0.1,            # SIL 損失整體權重
    value_coef=0.5,      # SIL 內部 value vs policy 配比
    adv_clip=5.0,        # max(R-V, 0) 上限
    warmup=2,            # 首 N 個 PPO iter 不做 SIL(等 buffer 累積)
    min_buffer=2048,     # buffer 不到此量就不更新
    prioritized=True,
)


class PPOSIL(PPO):
    """PPO 子類,在 train() 尾段加掛 SIL 更新。"""

    def attach_sil(self, sil_buffer: SILBuffer,
                   vecnormalize: VecNormalize,
                   config: Optional[dict] = None) -> None:
        cfg = dict(DEFAULT_SIL_CFG)
        if config:
            cfg.update(config)
        self._sil_buffer = sil_buffer
        self._sil_venv = vecnormalize
        self._sil_cfg = cfg
        self._sil_iter = 0

    # buffer prioritized sample 用的回呼;放 inference path
    @th.no_grad()
    def _sil_value_fn(self, obs_np: np.ndarray) -> np.ndarray:
        obs_t = th.as_tensor(obs_np, device=self.device, dtype=th.float32)
        return self.policy.predict_values(obs_t).flatten().cpu().numpy()

    def train(self) -> None:
        # 1) vanilla PPO 更新 ——————————————————————————————————
        super().train()

        # 2) SIL 更新 —————————————————————————————————————————
        buf: Optional[SILBuffer] = getattr(self, "_sil_buffer", None)
        if buf is None:
            return
        cfg = self._sil_cfg
        self._sil_iter += 1
        if self._sil_iter <= cfg["warmup"]:
            self.logger.record("sil/n_updates", 0)
            return
        if len(buf) < cfg["min_buffer"]:
            self.logger.record("sil/n_updates", 0)
            self.logger.record("sil/buffer_size", len(buf))
            return

        # SIL update 時 policy 也要在 train mode(共用 PPO 的 optimizer / lr)
        self.policy.set_training_mode(True)

        # VecNormalize 的 return 標準差:buffer 存 raw R,這裡轉到 value head 尺度
        ret_var = float(self._sil_venv.ret_rms.var) if self._sil_venv.norm_reward else 1.0
        ret_std = float(np.sqrt(ret_var + 1e-8))

        n_pos = 0
        n_tot = 0
        ploss_log: list[float] = []
        vloss_log: list[float] = []
        adv_mean_log: list[float] = []

        for _ in range(cfg["n_updates"]):
            obs_np, act_np, ret_np = buf.sample(
                cfg["batch_size"],
                prioritized=cfg["prioritized"],
                current_value_fn=self._sil_value_fn,
            )
            obs_t = th.as_tensor(obs_np, device=self.device, dtype=th.float32)
            act_t = th.as_tensor(act_np, device=self.device, dtype=th.long)
            ret_t = th.as_tensor(ret_np / ret_std,
                                 device=self.device, dtype=th.float32)

            values, log_prob, _ = self.policy.evaluate_actions(obs_t, act_t)
            values = values.flatten()

            adv = th.clamp(ret_t - values.detach(),
                           min=0.0, max=cfg["adv_clip"])
            mask = (adv > 0).float()

            policy_loss = -(log_prob * adv).mean()
            # value loss 只算「目標 R 高於 V」的樣本(SIL 不要把 V 往下拉)
            value_loss = 0.5 * (mask * (ret_t - values).pow(2)).mean()
            sil_loss = cfg["coef"] * (policy_loss + cfg["value_coef"] * value_loss)

            self.policy.optimizer.zero_grad()
            sil_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                        self.max_grad_norm)
            self.policy.optimizer.step()

            n_pos += int(mask.sum().item())
            n_tot += int(mask.numel())
            ploss_log.append(float(policy_loss.item()))
            vloss_log.append(float(value_loss.item()))
            adv_mean_log.append(float(adv.mean().item()))

        # 3) Logging —————————————————————————————————————————
        self.logger.record("sil/n_updates", cfg["n_updates"])
        self.logger.record("sil/policy_loss", float(np.mean(ploss_log)))
        self.logger.record("sil/value_loss", float(np.mean(vloss_log)))
        self.logger.record("sil/pos_adv_fraction",
                           float(n_pos / max(1, n_tot)))
        self.logger.record("sil/mean_adv", float(np.mean(adv_mean_log)))
        stats = buf.stats()
        self.logger.record("sil/buffer_size", stats["buffer_size"])
        self.logger.record("sil/buffer_wins", stats["buffer_wins"])
        self.logger.record("sil/episodes_admitted", stats["episodes_admitted"])
        self.logger.record("sil/wins_admitted", stats["wins_admitted"])
        self.logger.record("sil/mean_return_buf", stats["mean_return"])
        self.logger.record("sil/max_return_buf", stats["max_return"])
