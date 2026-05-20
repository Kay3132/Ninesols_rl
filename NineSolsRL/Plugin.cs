using System;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using BepInEx;
using UnityEngine;
using HarmonyLib;

namespace NineSolsRL
{
    [BepInPlugin("com.ninesolsrl.plugin", "NineSolsRL", "1.0.0")]
    public class Plugin : BaseUnityPlugin
    {
        public static Plugin Instance;

        private NetworkStream _stream;
        private TcpClient _pendingClient;
        private Task _connectTask;
        private bool _connected = false;

        private int _actionMove  = 0;
        private int _actionSkill = 0;

        void Awake()
        {
            Instance = this;
            DontDestroyOnLoad(this.gameObject);
            Logger.LogInfo("NineSolsRL 啟動...");
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
                var ph = p?.GetComponentInChildren<PlayerHealth>(true);
                Logger.LogInfo($"[RL Debug] tick={_debugTick} connected={_connected} Player={p != null} PH={ph != null}");
            }

            if (!_connected)
            {
                TryConnect();
                return;
            }

            // 讀取 Python 傳來的 action（非阻塞）
            try
            {
                if (_stream?.DataAvailable == true)
                {
                    var buf = new byte[4096];
                    int n = _stream.Read(buf, 0, buf.Length);
                    if (n > 0)
                        ParseAction(Encoding.UTF8.GetString(buf, 0, n).Trim());
                    else
                    { Disconnect(); return; }
                }
            }
            catch { Disconnect(); return; }

            // 每 6 tick 送一次 state（約 10Hz），避免 TCP 積壓
            if (_debugTick % 6 == 0)
            {
                var state = GetGameState();
                if (state != null) SendState(state);
            }
        }

        private void TryConnect()
        {
            // 等到 tick=300 才開始嘗試（讓遊戲完全載入）
            if (_debugTick < 300) return;

            // 檢查進行中的連線
            if (_connectTask != null)
            {
                if (!_connectTask.IsCompleted) return;

                if (!_connectTask.IsFaulted && _pendingClient != null && _pendingClient.Connected)
                {
                    Logger.LogInfo("已連線到 Python！");
                    _stream = _pendingClient.GetStream();
                    _connected = true;
                }
                else
                {
                    Logger.LogInfo($"connect 失敗 (tick={_debugTick})，重試中...");
                    try { _pendingClient?.Close(); } catch { }
                    _pendingClient = null;
                }
                _connectTask = null;
                return;
            }

            // 每 180 tick（約 3 秒）嘗試一次
            if (_debugTick % 180 != 0) return;

            Logger.LogInfo($"ConnectAsync 127.0.0.1:19271 (tick={_debugTick})...");
            _pendingClient = new TcpClient();
            try { _connectTask = _pendingClient.ConnectAsync("127.0.0.1", 19271); }
            catch (Exception ex)
            {
                Logger.LogInfo("ConnectAsync 例外: " + ex.Message);
                _pendingClient = null;
            }
        }

        private void Disconnect()
        {
            Logger.LogInfo("Python 斷線，重新嘗試連線...");
            try { _stream?.Close(); } catch { }
            try { _pendingClient?.Close(); } catch { }
            _stream = null;
            _pendingClient = null;
            _connectTask = null;
            _connected = false;
        }

        private string GetGameState()
        {
            try
            {
                var player = Player.i;
                if (player == null) return null;
                var playerHealth = player.GetComponentInChildren<PlayerHealth>(true);
                if (playerHealth == null) return null;

                float px  = player.transform.position.x;
                float py  = player.transform.position.y;
                float php = playerHealth.CurrentHealthValue;

                float bx = -1, by = -1, bhp = -1, bhpMax = 1;
                bool bossPresent = false;
                var boss = FindObjectOfType<MonsterBase>();
                if (boss != null)
                {
                    bx        = boss.transform.position.x;
                    by        = boss.transform.position.y;
                    bhp       = boss.health.currentValue;
                    bhpMax    = boss.health.maxValue > 0 ? boss.health.maxValue : 1f;
                    bossPresent = true;
                    if (_debugTick % 60 == 0)
                        Logger.LogInfo($"[Boss] {boss.name} hp={bhp}/{bhpMax}");
                }

                // done: 玩家死亡，或 boss 出現過但血量歸零
                bool done = (php <= 0) || (bossPresent && bhp <= 0);

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
            catch { Disconnect(); }
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

        private static int ExtractInt(string json, string key)
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
            Disconnect();
        }
    }
}
