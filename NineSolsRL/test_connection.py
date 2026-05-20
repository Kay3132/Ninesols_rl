import socket, json

HOST = "127.0.0.1"
PORT = 19271

print(f"連線到 {HOST}:{PORT} ...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("已連線！")
    buf = ""
    while True:
        data = s.recv(4096).decode("utf-8")
        if not data:
            print("連線中斷")
            break
        buf += data
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if line.strip():
                state = json.loads(line.strip())
                print(state)
