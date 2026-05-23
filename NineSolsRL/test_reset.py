"""
test_reset.py —— 單測 mod 的 [reset] 重啟記憶戰鬥功能（不跑 PPO）。

v1.17.0：reset = mod 呼叫 Player.Suicide() → memory mode 下死亡 → 遊戲自動重啟
這場 boss 戰。本腳本反覆送 {"reset": true} 並觀察「死亡 → 重生 → 重新可操作」。

前置：
  1. 開遊戲，確認 BepInEx log 出現 `NineSolsRL 1.17.0`。
  2. 手動進介川戰鬥（打到 boss 出現在畫面上）。
  3. 確認沒有別的 train.py 在跑（會搶 port 19271）。

用法：
  cd NineSolsRL && uv run python test_reset.py
"""
import time

from ninesols.bridge import GameBridge

N_RESETS = 5  # 連續測幾次 reset（想多測改這裡）


def fmt(s: dict) -> str:
    return (f"controllable={s.get('controllable')} php={s.get('php')} "
            f"done={s.get('done')} boss_present={s.get('boss_present')} "
            f"boss='{s.get('boss_name', '')}' bhp={s.get('bhp')}")


def main():
    bridge = GameBridge()
    print("[test] 等遊戲連線 127.0.0.1:19271 ...")
    bridge.ensure_connected()

    s = bridge.recv_state()
    print(f"[test] 初始狀態: {fmt(s)}")
    if not s.get("boss_present"):
        print("[test] ⚠ boss_present=False —— 你可能不在 boss 戰鬥中。")

    for i in range(1, N_RESETS + 1):
        print(f"\n===== reset #{i} =====")
        t0 = time.time()
        bridge.send_action({"reset": True})

        # 等死亡 + 戰鬥重啟：玩家先變不可操作/php<=0，再重生回可操作且存活
        saw_dead = False
        while True:
            s = bridge.recv_state(timeout=30.0)
            if not s.get("controllable") or s.get("php", 1) <= 0 or s.get("done"):
                saw_dead = True
            ok = (s.get("controllable") and s.get("php", 0) > 0
                  and not s.get("done"))
            if ok and saw_dead:
                break
            bridge.send_action({"move": 0})

        print(f"[test] reset #{i} 重生完成，耗時 {time.time() - t0:.1f}s | {fmt(s)}")

        # 重生後觀察 8 秒（持續送 no-op），看是否回到 boss 戰鬥
        boss_seen = False
        for _ in range(240):
            bridge.send_action({"move": 0})
            s = bridge.recv_state()
            if s.get("boss_present"):
                boss_seen = True
        verdict = "✓ 回到 boss 戰" if boss_seen else "✗ 沒看到 boss（重生點離 boss 較遠？）"
        print(f"[test] reset #{i} 後 8 秒: {fmt(s)} | {verdict}")

    print("\n[test] 全部 reset 測試完成。")
    print("[test] 判讀：每次都應看到「死亡 → 重生」，mod log 每次出現 "
          "`[reset] Player.Suicide() → 重啟記憶戰鬥`。")
    bridge.close()


if __name__ == "__main__":
    main()
