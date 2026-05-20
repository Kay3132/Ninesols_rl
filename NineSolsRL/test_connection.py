import socket, json, time

HOST = "127.0.0.1"
PORT = 19271

# Action 協定（每個 key 都可同時送）：
#   move:   0=停, 1=左, 2=右
#   jump:   0/1   跳躍
#   attack: 0/1   攻擊
#   dodge:  0/1   閃避/衝刺
#   parry:  0/1   格檔


def send_action(conn, action: dict):
    msg = json.dumps(action) + "\n"
    conn.sendall(msg.encode("utf-8"))


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"[PY] 等待遊戲連線到 {HOST}:{PORT} ...")

conn, addr = server.accept()
print(f"[PY] 遊戲已連線！{addr}")

buf = ""
state_count = 0
start = time.time()
while True:
    try:
        data = conn.recv(4096).decode("utf-8")
    except Exception as e:
        print(f"[PY] recv 錯誤: {e}")
        break
    if not data:
        print("[PY] 連線中斷")
        break
    buf += data
    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        if not line.strip():
            continue
        try:
            state = json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f"[PY] JSON 錯誤: {e}")
            continue

        state_count += 1
        boss_present = state.get("bx", -1) != -1

        # ---- 動作測試腳本：每 3 秒換一個動作，逐項驗證 ----
        phase = int(time.time() - start) // 3 % 5
        if phase == 0:
            action = {"move": 2}                    # 右移
            label = "右移"
        elif phase == 1:
            action = {"move": 1}                    # 左移
            label = "左移"
        elif phase == 2:
            action = {"move": 0, "jump": 1}         # 跳躍
            label = "跳躍"
        elif phase == 3:
            action = {"move": 0, "attack": 1}       # 攻擊
            label = "攻擊"
        else:
            action = {"move": 0, "dodge": 1}        # 閃避
            label = "閃避"

        send_action(conn, action)

        if state_count % 20 == 1:
            print(f"[#{state_count}] {label:4} px={state.get('px'):.1f} "
                  f"php={state.get('php')} bhp={state.get('bhp')} action={action}")
