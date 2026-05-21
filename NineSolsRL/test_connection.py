import socket, json, time

HOST, PORT = "127.0.0.1", 19271

# ---- Action 協定（遊戲 <- Python）----
#   move:   0=停, 1=左, 2=右
#   jump / attack / dodge / parry: 0/1
#
# ---- State 協定（遊戲 -> Python）----
#   玩家： px py vx vy facing grounded php php_pct internal_injury
#   boss： boss_present bx by bvx bvy b_facing bdx bdy bhp bhp_pct
#   其他： controllable  state  done


def send_action(conn, action: dict):
    conn.sendall((json.dumps(action) + "\n").encode("utf-8"))


def new_episode_state():
    """每個 episode 開始時的歸零狀態。"""
    return {
        "step": 0,
        "start": time.time(),
        "total_reward": 0.0,
        "prev_php": None,
    }


def run_session(conn):
    """處理單一遊戲連線；遊戲斷線時 return（外層會重新 accept）。"""
    buf = ""
    episode = 1
    ep = new_episode_state()
    dead = False
    print(f"=== Episode {episode} 開始 ===")

    while True:
        try:
            data = conn.recv(4096).decode("utf-8")
        except Exception as e:
            print(f"[PY] recv 錯誤: {e}")
            return
        if not data:
            print("[PY] 連線中斷")
            return
        buf += data

        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if not line.strip():
                continue
            try:
                state = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            php    = state.get("php", 1)
            done   = state.get("done", False)
            ctrl   = state.get("controllable", False)
            pstate = state.get("state", "?")

            # ---- 死亡偵測：結束目前 episode、歸零 state ----
            if not dead and (php <= 0 or done or pstate == "Dead"):
                dead = True
                survived = time.time() - ep["start"]
                print(f"=== Episode {episode} 結束 | 步數={ep['step']} "
                      f"存活={survived:.1f}s 累積獎勵={ep['total_reward']:.1f} ===")

            # ---- 重生偵測：開新 episode，state 歸零 ----
            if dead and php > 0 and not done and ctrl:
                dead = False
                episode += 1
                ep = new_episode_state()
                print(f"=== Episode {episode} 開始 ===")

            # ---- 死亡/過場中：送停止動作，不累積步數 ----
            if dead or not ctrl:
                send_action(conn, {"move": 0})
                continue

            # ---- 正常步進：簡易測試策略（每 3 秒輪換動作）----
            ep["step"] += 1
            phase = int(time.time() - ep["start"]) // 3 % 5
            if phase == 0:
                action, label = {"move": 2}, "右移"
            elif phase == 1:
                action, label = {"move": 1}, "左移"
            elif phase == 2:
                action, label = {"move": 0, "jump": 1}, "跳躍"
            elif phase == 3:
                action, label = {"move": 0, "attack": 1}, "攻擊"
            else:
                action, label = {"move": 0, "dodge": 1}, "閃避"

            # ---- 簡單獎勵：扣血給負獎勵（示範用）----
            if ep["prev_php"] is not None and php < ep["prev_php"]:
                ep["total_reward"] -= (ep["prev_php"] - php)
            ep["prev_php"] = php

            send_action(conn, action)

            if ep["step"] % 20 == 1:
                print(f"[E{episode} #{ep['step']:4}] {label} "
                      f"px={state.get('px'):.0f} php={php} "
                      f"bhp_pct={state.get('bhp_pct', 0):.2f} state={pstate}")


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    # 外層 while True：持續接受連線，遊戲重開也不用重跑腳本
    while True:
        print(f"[PY] 等待遊戲連線到 {HOST}:{PORT} ...")
        conn, addr = server.accept()
        print(f"[PY] 遊戲已連線！{addr}")
        try:
            run_session(conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        print("[PY] 連線結束，重新等待下一次連線 ...\n")


if __name__ == "__main__":
    main()
