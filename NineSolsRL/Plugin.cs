using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using BepInEx;
using UnityEngine;
using HarmonyLib;

namespace NineSolsRL
{
    [BepInPlugin("com.ninesolsrl.plugin", "NineSolsRL", "1.0.0")]
    public class Plugin : BaseUnityPlugin
    {
        public static Plugin Instance;

        private TcpListener _server;
        private TcpClient _client;
        private NetworkStream _stream;
        private Thread _listenThread;
        private bool _running = false;

        private int _actionMove  = 0;
        private int _actionSkill = 0;

        void Awake()
        {
            Instance = this;
            DontDestroyOnLoad(this.gameObject);
            Logger.LogInfo("NineSolsRL 啟動...");
            _running = true;
            _listenThread = new Thread(ListenForPython);
            _listenThread.IsBackground = true;
            _listenThread.Start();

            var harmony = new Harmony("com.ninesolsrl.plugin");
            harmony.PatchAll();
        }

        [HarmonyPatch(typeof(GameCore), "Update")]
        public static class PatchGameCoreUpdate
        {
            static void Postfix()
            {
                Instance?.OnGameUpdate();
            }
        }

        private int _debugTick = 0;
        public void OnGameUpdate()
        {
            _debugTick++;
            if (_debugTick % 60 == 0)
            {
                var p = Player.i;
                var ph = FindObjectOfType<PlayerHealth>();
                Logger.LogInfo($"[RL Debug] tick={_debugTick} connected={_connected} Player={p != null} PlayerHealth={ph != null}");
            }
            if (!_connected) return;
            var state = GetGameState();
            if (state == null) return;
            SendState(state);
        }

        private string GetGameState()
        {
            try
            {
                var player = Player.i;
                if (player == null) return null;
                var playerHealth = player.GetComponent<PlayerHealth>();
                if (playerHealth == null) return null;

                float px  = player.transform.position.x;
                float py  = player.transform.position.y;
                float php = playerHealth.CurrentHealthValue;

                float bx = -1, by = -1, bhp = 0, bhpMax = 1;
                var boss = FindObjectOfType<MonsterBase>();
                if (boss != null)
                {
                    bx     = boss.transform.position.x;
                    by     = boss.transform.position.y;
                    bhp    = boss.health.currentValue;
                    bhpMax = 100f;
                }

                bool done = (php <= 0) || (bhp <= 0);

                return $"{{\"px\":{px},\"py\":{py}," +
                       $"\"php\":{php}," +
                       $"\"bx\":{bx},\"by\":{by}," +
                       $"\"bhp\":{bhp},\"bhp_max\":{bhpMax}," +
                       $"\"done\":{(done ? "true" : "false")}}}\n";
            }
            catch (Exception e)
            {
                Logger.LogError(e);
                return null;
            }
        }

        private void SendState(string json)
        {
            if (_stream == null) return;
            try
            {
                var bytes = Encoding.UTF8.GetBytes(json);
                _stream.Write(bytes, 0, bytes.Length);
            }
            catch { _stream = null; }
        }

        private volatile bool _connected = false;
        private void ListenForPython()
        {
            try
            {
                Logger.LogInfo("等待 5 秒...");
                System.Threading.Thread.Sleep(5000);
                Logger.LogInfo("開始啟動 TCP server...");

                _server = new TcpListener(IPAddress.Loopback, 19271);
                _server.Start();
                Logger.LogInfo("TCP server 啟動成功，port 19271");

                Logger.LogInfo("等待 Python 連線...");
                int pollHeartbeat = 0;
                while (_running)
                {
                    bool ready = false;
                    try { ready = _server.Server.Poll(0, System.Net.Sockets.SelectMode.SelectRead); }
                    catch (Exception ex) { Logger.LogError("Poll 失敗: " + ex.Message); System.Threading.Thread.Sleep(50); continue; }

                    pollHeartbeat++;
                    if (pollHeartbeat % 300 == 0)
                        Logger.LogInfo($"[Poll 心跳] iter={pollHeartbeat} ready={ready} server_bound={_server.Server.IsBound}");

                    if (!ready) { System.Threading.Thread.Sleep(10); continue; }

                    Logger.LogInfo("偵測到連線，嘗試 Accept...");
                    System.Net.Sockets.TcpClient tcpClient;
                    try { tcpClient = _server.AcceptTcpClient(); }
                    catch (Exception ex) { Logger.LogError("Accept 失敗: " + ex.Message); continue; }

                    _stream = tcpClient.GetStream();
                    Logger.LogInfo("Python 已連線！");
                    _connected = true;

                    var buf = new byte[4096];
                    while (_running)
                    {
                        try
                        {
                            int n = _stream.Read(buf, 0, buf.Length);
                            if (n == 0) break;
                            ParseAction(Encoding.UTF8.GetString(buf, 0, n).Trim());
                        }
                        catch { break; }
                    }
                    Logger.LogInfo("Python 斷線，重新等待...");
                    _stream = null;
                    _client = null;
                    _connected = false;
                }
            }
            catch (Exception e)
            {
                Logger.LogError("ListenForPython 錯誤: " + e.ToString());
            }
        }

        private void ParseAction(string json)
        {
            try
            {
                _actionMove  = ExtractInt(json, "move");
                _actionSkill = ExtractInt(json, "skill");
            }
            catch { }
        }

        private int ExtractInt(string json, string key)
        {
            string search = $"\"{key}\":";
            int idx = json.IndexOf(search);
            if (idx < 0) return 0;
            int start = idx + search.Length;
            int end = start;
            while (end < json.Length && (char.IsDigit(json[end]) || json[end] == '-'))
                end++;
            return int.Parse(json.Substring(start, end - start));
        }

        void OnDestroy()
        {
            _running = false;
            _stream?.Close();
            _client?.Close();
            _server?.Stop();
        }
    }
}
