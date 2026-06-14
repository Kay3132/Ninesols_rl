"""
train.py —— stable-baselines3 PPO + Self-Imitation Learning(SIL)

v1.21.0:加掛 SIL —— 在不丟現有 PPO checkpoint 的前提下,讓 agent 把過去
高回報 / WIN / 近 WIN(phase 2 bhp<100)episode 存到 SILBuffer,每次 PPO
更新後抽 4 個 minibatch 做加權 BC + value update。

自動續訓:啟動時掃 checkpoints/,有 ppo_ninesols_*_steps.zip 就挑步數
最大那組(模型 + vecnormalize.pkl + 同步驟數的 sil_buffer.pkl)載回。
找不到 sil_buffer pkl 沒事(冷啟空 buffer)。

使用方式:
  1. 啟動遊戲、把角色帶到 boss 場
  2. cd NineSolsRL && uv run python train.py
"""
import os
import re
import glob
from datetime import datetime

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure  # rebuilt 9046049: per-iteration CSV

from ninesols import NineSolsEnv
from ninesols.ppo_sil import PPOSIL, DEFAULT_SIL_CFG
from ninesols.sil_buffer import SILBuffer
from ninesols.sil_callbacks import (SILEpisodeRecorderCallback,
                                    SILBufferCheckpointCallback,
                                    EpisodeCSVCallback,
                                    find_latest_sil_buffer)

TOTAL_TIMESTEPS = 5_000_000        # 絕對目標步數  # rebuilt f3502a7 2026-06-13: 1M → 3M
GAMMA           = 0.98
NAME_PREFIX     = "ppo_ninesols"
SAVE_PATH       = "ppo_ninesols"
VECNORM_PATH    = "ppo_ninesols_vecnormalize.pkl"
SIL_BUFFER_PATH = "sil_buffer.pkl"   # 最終 (非 checkpoint) 存檔
CKPT_DIR        = "checkpoints"

# SIL 設定(可調)。warmup=2 → 前 2 個 PPO iter 不做 SIL,先讓 recorder 灌 buffer。
SIL_CFG = dict(DEFAULT_SIL_CFG)  # 走預設,後續手動微調 coef 再來這裡改


def make_vecnormalize(venv):
    """全新的 VecNormalize(從零訓練時用)。"""
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )


def find_latest_checkpoint(ckpt_dir: str, prefix: str):
    """掃 ckpt_dir,回傳 (zip_path, vecnormalize_pkl_path, steps)。"""
    zips = glob.glob(os.path.join(ckpt_dir, f"{prefix}_*_steps.zip"))
    if not zips:
        return None, None, 0

    def steps_of(path: str) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    latest_zip = max(zips, key=steps_of)
    steps = steps_of(latest_zip)
    pkl = os.path.join(ckpt_dir, f"{prefix}_vecnormalize_{steps}_steps.pkl")
    return latest_zip, (pkl if os.path.exists(pkl) else None), steps


def main():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[train] 使用 GPU:{torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[train] 找不到 CUDA,改用 CPU")

    venv = DummyVecEnv([lambda: Monitor(NineSolsEnv(max_steps=2000))])

    latest_zip, latest_pkl, latest_steps = find_latest_checkpoint(CKPT_DIR, NAME_PREFIX)
    resumed = latest_zip is not None

    if resumed:
        print(f"[train] 發現 checkpoint,續訓:{latest_zip}")
        if latest_pkl:
            env = VecNormalize.load(latest_pkl, venv)
            env.training = True
            env.norm_reward = True
        else:
            print("[train] ⚠ 找不到對應的 vecnormalize.pkl,改用全新統計(前期可能不穩)")
            env = make_vecnormalize(venv)
        # 用 PPOSIL.load 載入:SB3 honors caller class → instance 會是 PPOSIL,
        # weights 從 vanilla PPO zip 對得回來(class 名沒被存在 zip 內)。
        model = PPOSIL.load(latest_zip, env=env, device=device)
        model.ent_coef = 0.005
        print(f"[train] 已載入,目前步數 = {model.num_timesteps}, ent_coef={model.ent_coef}")
    else:
        print("[train] 沒有 checkpoint,從零開始訓練")
        env = make_vecnormalize(venv)
        model = PPOSIL(
            "MlpPolicy", env,
            device=device,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            gamma=GAMMA,
            learning_rate=3e-4,
            ent_coef=0.005,
        )

    # SIL buffer:對應 ckpt 步數的 pkl,沒有就空起跑
    sil_buf_path = find_latest_sil_buffer(CKPT_DIR, latest_steps) if resumed else None
    if sil_buf_path:
        sil_buffer = SILBuffer.load(sil_buf_path)
        print(f"[train] 載入 SIL buffer:{sil_buf_path} "
              f"(size={len(sil_buffer)}, wins={sil_buffer.n_wins})")
    else:
        sil_buffer = SILBuffer()
        print(f"[train] SIL buffer 冷啟動(空)")

    model.attach_sil(sil_buffer, env, SIL_CFG)
    print(f"[train] SIL config: {SIL_CFG}")

    # rebuilt 9046049 2026-06-14: PPO per-iteration CSV logging。
    # SB3 內建 logger：每個 PPO iteration dump 一列(ep_rew_mean/ep_len_mean/loss/
    # entropy/sil_* 等全部 metrics)→ logs/run_<時間>/progress.csv。
    # 每次啟動開新 run 資料夾，避免續訓覆蓋掉前一輪的 csv。
    log_dir = os.path.join("logs", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "csv"]))
    print(f"[train] PPO/SIL metrics → {os.path.join(log_dir, 'progress.csv')}")

    # callback:ckpt(含 vecnormalize + sil buffer)+ recorder
    ckpt_cb = SILBufferCheckpointCallback(
        sil_buffer=sil_buffer,
        save_freq=40960,
        save_path=CKPT_DIR,
        name_prefix=NAME_PREFIX,
        save_vecnormalize=True,
        verbose=1,
    )
    recorder_cb = SILEpisodeRecorderCallback(sil_buffer, verbose=1)
    # rebuilt 9046049 2026-06-14: per-episode CSV(logs/run_*/episodes.csv)，
    # 用 total_timesteps 與 PPO progress.csv 對齊
    episode_csv_cb = EpisodeCSVCallback(verbose=1)
    callbacks = CallbackList([ckpt_cb, recorder_cb, episode_csv_cb])

    try:
        if resumed:
            remaining = TOTAL_TIMESTEPS - model.num_timesteps
            if remaining <= 0:
                print(f"[train] 已達目標 {TOTAL_TIMESTEPS} 步,無需續訓")
                return
            print(f"[train] 續訓 {remaining} 步(目標總數 {TOTAL_TIMESTEPS})")
            model.learn(total_timesteps=remaining,
                        callback=callbacks, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=TOTAL_TIMESTEPS,
                        callback=callbacks, reset_num_timesteps=True)
        model.save(SAVE_PATH)
        env.save(VECNORM_PATH)
        sil_buffer.save(SIL_BUFFER_PATH)
        print(f"[train] 模型已儲存:{SAVE_PATH}.zip / {VECNORM_PATH} / {SIL_BUFFER_PATH}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
