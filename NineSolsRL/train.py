"""
train.py —— 用 stable-baselines3 PPO 訓練

自動續訓：啟動時掃 checkpoints/，有 checkpoint 就挑「步數最大」那組
（{prefix}_N_steps.zip 模型 + 對應的 vecnormalize .pkl）載入接著練；
沒有就從零開始。中斷後直接再跑一次 train.py 就會自動接上。

調整重點（解決「critic 會學、actor 卡在隨機」）：
  1. reward 尺度：rewards.py 權重重新設計比例 + VecNormalize 跑動正規化
  2. credit assignment：gamma 0.98（回報視野 ~50 步 ≈ 2 秒 @25Hz）
  3. 跑更久：TOTAL_TIMESTEPS 當「絕對目標步數」，CheckpointCallback 定期存檔

使用方式：
  1. 先啟動遊戲、把角色帶到 boss 場
  2. cd NineSolsRL && uv run python train.py
"""
import os
import re
import glob

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from ninesols import NineSolsEnv

TOTAL_TIMESTEPS = 1_000_000        # 絕對目標步數（續訓會練到這個總數為止）
GAMMA           = 0.98
NAME_PREFIX     = "ppo_ninesols"
SAVE_PATH       = "ppo_ninesols"
VECNORM_PATH    = "ppo_ninesols_vecnormalize.pkl"
CKPT_DIR        = "checkpoints"


def make_vecnormalize(venv):
    """全新的 VecNormalize（從零訓練時用）。"""
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )


def find_latest_checkpoint(ckpt_dir: str, prefix: str):
    """掃 ckpt_dir，回傳步數最大的 (模型.zip, vecnormalize.pkl)。沒有就 (None, None)。"""
    zips = glob.glob(os.path.join(ckpt_dir, f"{prefix}_*_steps.zip"))
    if not zips:
        return None, None

    def steps_of(path: str) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    latest_zip = max(zips, key=steps_of)
    steps = steps_of(latest_zip)
    pkl = os.path.join(ckpt_dir, f"{prefix}_vecnormalize_{steps}_steps.pkl")
    return latest_zip, (pkl if os.path.exists(pkl) else None)


def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[train] 使用 GPU：{torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[train] 找不到 CUDA，改用 CPU")

    venv = DummyVecEnv([lambda: Monitor(NineSolsEnv(max_steps=2000))])

    latest_zip, latest_pkl = find_latest_checkpoint(CKPT_DIR, NAME_PREFIX)
    resumed = latest_zip is not None

    if resumed:
        print(f"[train] 發現 checkpoint，續訓：{latest_zip}")
        # VecNormalize 統計一定要一起載回來，否則 obs/reward 尺度對不上、模型會崩
        if latest_pkl:
            env = VecNormalize.load(latest_pkl, venv)
            env.training = True          # 續訓 → 統計繼續更新
            env.norm_reward = True
        else:
            print("[train] ⚠ 找不到對應的 vecnormalize.pkl，改用全新統計（前期可能不穩）")
            env = make_vecnormalize(venv)
        model = PPO.load(latest_zip, env=env, device=device)
        print(f"[train] 已載入，目前步數 = {model.num_timesteps}")
    else:
        print("[train] 沒有 checkpoint，從零開始訓練")
        env = make_vecnormalize(venv)
        model = PPO(
            "MlpPolicy", env,
            device=device,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            gamma=GAMMA,
            learning_rate=3e-4,
        )

    # 每 N 步存一次（save_vecnormalize 把正規化統計一起存，續訓才對得上）
    ckpt_cb = CheckpointCallback(
        save_freq=20480,
        save_path=CKPT_DIR,
        name_prefix=NAME_PREFIX,
        save_vecnormalize=True,
    )

    try:
        if resumed:
            # reset_num_timesteps=False → SB3 內部把 remaining 加回現有步數，
            # 等於「練到 TOTAL_TIMESTEPS 這個絕對總數為止」。
            remaining = TOTAL_TIMESTEPS - model.num_timesteps
            if remaining <= 0:
                print(f"[train] 已達目標 {TOTAL_TIMESTEPS} 步，無需續訓")
                return
            print(f"[train] 續訓 {remaining} 步（目標總數 {TOTAL_TIMESTEPS}）")
            model.learn(total_timesteps=remaining,
                        callback=ckpt_cb, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=TOTAL_TIMESTEPS,
                        callback=ckpt_cb, reset_num_timesteps=True)
        model.save(SAVE_PATH)
        env.save(VECNORM_PATH)
        print(f"[train] 模型已儲存：{SAVE_PATH}.zip / {VECNORM_PATH}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
