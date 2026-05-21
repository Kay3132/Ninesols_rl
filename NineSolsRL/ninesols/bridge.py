"""
bridge.py —— 與遊戲 mod 的 TCP 通訊層。

Python 當 server（bind port），遊戲端 C# mod 當 client 連入。
  - send_action(dict)  送一筆 action
  - recv_state()       讀下一筆 state；buffer 有多筆時回傳「最新」一筆（避免延遲累積）
"""
import json
import socket


class GameBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 19271):
        self.host, self.port = host, port
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self._conn: socket.socket | None = None
        self._buf = ""

    # ---- 連線管理 ----
    def ensure_connected(self):
        if self._conn is None:
            print(f"[bridge] 等待遊戲連線 {self.host}:{self.port} ...")
            self._conn, addr = self._server.accept()
            print(f"[bridge] 遊戲已連線 {addr}")

    def _drop(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._buf = ""

    # ---- 收 / 發 ----
    def send_action(self, action: dict):
        self.ensure_connected()
        try:
            self._conn.sendall((json.dumps(action) + "\n").encode("utf-8"))
        except Exception:
            self._drop()
            raise ConnectionError("送出 action 失敗：遊戲斷線")

    def recv_state(self, timeout: float = 10.0) -> dict:
        """阻塞讀取下一筆 state；若同時有多筆，回傳最新一筆。"""
        self.ensure_connected()
        self._conn.settimeout(timeout)

        # 1. 讀到至少一筆完整資料（含換行）
        while "\n" not in self._buf:
            data = self._conn.recv(8192)
            if not data:
                self._drop()
                raise ConnectionError("遊戲斷線")
            self._buf += data.decode("utf-8", errors="ignore")

        # 2. 非阻塞吸乾 socket 內已到達的剩餘資料 → 確保拿到最新 state
        self._conn.setblocking(False)
        try:
            while True:
                data = self._conn.recv(8192)
                if not data:
                    break
                self._buf += data.decode("utf-8", errors="ignore")
        except (BlockingIOError, OSError):
            pass
        finally:
            self._conn.setblocking(True)

        # 3. 取最後一筆完整 JSON，殘留未完成的留在 buffer
        lines = self._buf.split("\n")
        self._buf = lines[-1]
        complete = [ln for ln in lines[:-1] if ln.strip()]
        return json.loads(complete[-1])

    def close(self):
        self._drop()
        try:
            self._server.close()
        except Exception:
            pass
