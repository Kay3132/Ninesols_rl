"""
main.py —— Nine Sols RL 入口

目前職責：啟動連線伺服器 NineSolsRL/test_connection.py。
遊戲請自行手動開啟。

TODO（之後）：判斷遊戲處於「主選單 / 回想 / 關卡內」，分別做不同設定。

實際訓練請用 NineSolsRL/train.py。
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))           # e:\Ninesols_rl
PROJECT = os.path.join(HERE, "NineSolsRL")
SERVER_SCRIPT = os.path.join(PROJECT, "test_connection.py")


def main():
    print("[main] （遊戲請自行手動開啟）")
    print(f"[main] 啟動連線伺服器：{SERVER_SCRIPT}")
    server = subprocess.Popen([sys.executable, SERVER_SCRIPT])

    print("[main] 啟動完成。按 Ctrl+C 結束。")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\n[main] 收到中斷，關閉中 ...")
    finally:
        if server.poll() is None:
            try:
                server.terminate()
                print("[main] 已關閉伺服器")
            except Exception as e:
                print(f"[main] 關閉伺服器失敗：{e}")


if __name__ == "__main__":
    main()
