"""
main.py —— Nine Sols RL 入口

職責：
  1. 啟動遊戲（Nine Sols，會自動載入 BepInEx + NineSolsRL mod）
  2. 啟動連線伺服器（test_connection.py）

之後 RL 架構（env / agent / train）再逐步拆分檔案。
"""
import os
import sys
import time
import subprocess

# ---- 設定 ----
GAME_EXE = r"E:\SteamLibrary\steamapps\common\Nine Sols\NineSols.exe"
HERE = os.path.dirname(os.path.abspath(__file__))        # e:\Ninesols_rl
PROJECT = os.path.join(HERE, "NineSolsRL")               # Python 檔所在子資料夾
SERVER_SCRIPT = os.path.join(PROJECT, "test_connection.py")

# 若用 Steam 啟動（避免 Steam 未開時遊戲報錯），改用這個：
#   STEAM_APP_ID = "1809540"
#   啟動指令： os.startfile(f"steam://rungameid/{STEAM_APP_ID}")
USE_STEAM = False
STEAM_APP_ID = "1809540"


def launch_game():
    if USE_STEAM:
        print(f"[main] 透過 Steam 啟動遊戲 (appid={STEAM_APP_ID}) ...")
        os.startfile(f"steam://rungameid/{STEAM_APP_ID}")
        return None
    if not os.path.exists(GAME_EXE):
        print(f"[main] 找不到遊戲執行檔：{GAME_EXE}")
        print("[main] 請修改 main.py 裡的 GAME_EXE，或設 USE_STEAM=True")
        return None
    print(f"[main] 啟動遊戲：{GAME_EXE}")
    return subprocess.Popen([GAME_EXE], cwd=os.path.dirname(GAME_EXE))


def main():
    # 1. 先啟動遊戲（載入時間較長，C# 端會自動重試連線）
    game = launch_game()

    # 2. 稍等一下再啟動連線伺服器
    #    test_connection.py 內含 while True，會持續等待遊戲連入
    time.sleep(2.0)
    print(f"[main] 啟動連線伺服器：{SERVER_SCRIPT}")
    server = subprocess.Popen([sys.executable, SERVER_SCRIPT])

    print("[main] 全部啟動完成。按 Ctrl+C 結束。")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n[main] 收到中斷，關閉中 ...")
    finally:
        for proc, name in ((server, "伺服器"), (game, "遊戲")):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    print(f"[main] 已關閉{name}")
                except Exception as e:
                    print(f"[main] 關閉{name}失敗：{e}")


if __name__ == "__main__":
    main()
