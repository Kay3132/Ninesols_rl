import socket, json, threading, time, os

HOST = "127.0.0.1"
PORT = 19271
LOG_PATH = r"E:\SteamLibrary\steamapps\common\Nine Sols\BepInEx\LogOutput.log"


def tail_bepinex_log():
    """在背景持續讀取 BepInEx LogOutput.log 的新內容"""
    if not os.path.exists(LOG_PATH):
        print(f"[LOG] 找不到 log 檔: {LOG_PATH}")
        return
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)  # 跳到檔案末尾，只看新內容
        while True:
            line = f.readline()
            if line:
                print("[LOG]", line, end="")
            else:
                time.sleep(0.05)


# 啟動 log 監控 thread
log_thread = threading.Thread(target=tail_bepinex_log, daemon=True)
log_thread.start()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"[PY] Python server 啟動，等待遊戲連線到 {HOST}:{PORT} ...")

conn, addr = server.accept()
print(f"[PY] 遊戲已連線！{addr}")

buf = ""
state_count = 0
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
        if line.strip():
            try:
                state = json.loads(line.strip())
                state_count += 1
                if state_count % 60 == 1:  # 每秒印一次（約 60fps）
                    print(f"[STATE #{state_count}]", state)
            except json.JSONDecodeError as e:
                print(f"[PY] JSON 解析錯誤: {e}  raw={line[:80]}")
