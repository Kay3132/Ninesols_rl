"""
train.py —— 用 stable-baselines3 PPO 訓練

使用方式：
  1. 先啟動遊戲（python main.py 只會開遊戲＋舊測試腳本，訓練時請改用本檔）
  2. uv run python train.py

流程：建立 NineSolsEnv（會 bind port 等遊戲連入）→ PPO 訓練。
"""
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ninesols import NineSolsEnv

TOTAL_TIMESTEPS = 100_000
SAVE_PATH = "ppo_ninesols"


def main():
    env = Monitor(NineSolsEnv(max_steps=2000))

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        tensorboard_log="./tb_logs",
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(SAVE_PATH)
        print(f"[train] 模型已儲存：{SAVE_PATH}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
