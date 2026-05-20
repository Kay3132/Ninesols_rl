using System;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using BepInEx;
using UnityEngine;
using HarmonyLib;

namespace NineSolsRL
{
    [BepInPlugin("com.ninesolsrl.plugin", "NineSolsRL", "1.5.0")]
    public class Plugin : BaseUnityPlugin
    {
        public static Plugin Instance;
        internal static BepInEx.Logging.ManualLogSource Log;   // 純 managed logger，不受 Unity 物件狀態影響

        private NetworkStream _stream;
        private TcpClient _pendingClient;
        private Task _connectTask;
        private bool _connected = false;

        private float _prevPx = 0, _prevPy = 0;
        private float _prevStateTime = 0;

        // ---- RL action 狀態 ----
        private int _moveDir = 0;                       // -1 左, 0 停, 1 右
        private int _jumpPulse, _attackPulse, _dodgePulse, _parryPulse;
        private const int PULSE = 3;                    // 離散動作維持的幀數

        private int _debugTick = 0;

        void Awake()
        {
            Instance = this;
            Log = Logger;
            DontDestroyOnLoad(this.gameObject);
            Logger.LogInfo("NineSolsRL 啟動...");
            var harmony = new Harmony("com.ninesolsrl.plugin");
            harmony.PatchAll();

            // 手動 patch InControl.PlayerAction.WasPressed getter（注入跳/攻/閃/格檔）
            try
            {
                var paType = AccessTools.TypeByName("InControl.PlayerAction");
                if (paType != null)
                {
                    var getter = AccessTools.PropertyGetter(paType, "WasPressed");
                    harmony.Patch(getter, postfix: new HarmonyMethod(
                        typeof(Plugin).GetMethod(nameof(WasPressedPostfix),
                            BindingFlags.Static | BindingFlags.NonPublic)));
                    Logger.LogInfo("已 patch InControl.PlayerAction.WasPressed");
                }
                else
                {
                    Logger.LogError("找不到 InControl.PlayerAction 型別，離散動作將無法注入");
                }
            }
            catch (Exception e) { Logger.LogError("InControl patch 失敗: " + e); }
        }

        // ============ Harmony patches ============

        // 主迴圈：每幀跑一次
        [HarmonyPatch(typeof(GameCore), "Update")]
        public static class PatchGameCoreUpdate
        {
            static void Postfix() => Instance?.OnGameUpdate();
        }

        // HMC 被呼叫的次數（無條件計數，用於診斷）
        public static long _hmcCalls = 0;

        // 移動注入：在遊戲算完水平移動後，直接覆蓋 moveX / VelX
        [HarmonyPatch(typeof(Player), "HorizontalMoveCheck")]
        public static class PatchHorizontalMoveCheck
        {
            private static int _logTick = 0;
            static void Postfix(Player __instance)
            {
                _hmcCalls++;
                var inst = Instance;
                // 用 ReferenceEquals 做 null 檢查，避免 Unity 的 == 把已 destroy 的 plugin 當成 null
                if (ReferenceEquals(inst, null) || !inst._connected) return;
                if (ReferenceEquals(__instance, null)) return;

                // 守門：只有玩家「真正可操作」時才注入移動，
                // 過場/劇情/傳送時讓遊戲完全掌控
                if (!IsPlayerControllable(__instance)) return;

                int dir = inst._moveDir;
                float maxRun;
                try { maxRun = __instance.MaxRunStat.Value; } catch { maxRun = 8f; }
                if (dir != 0)
                {
                    __instance.moveX = dir;
                    __instance.VelX  = maxRun * dir;     // 直接給速度，繞過內部加速邏輯
                }
                else
                {
                    __instance.moveX = 0;                // 交給遊戲自然減速
                }

                if (++_logTick % 120 == 0)
                    Log?.LogInfo($"[HMC] dir={dir} VelX={__instance.VelX:F1} " +
                                 $"maxRun={maxRun:F1} state={__instance.CurrentStateType}");
            }
        }

        // 判斷玩家此刻是否真正可被玩家操作（過場/對話/劇情/傳送/死亡時為 false）
        internal static bool IsPlayerControllable(Player p)
        {
            try
            {
                if (ReferenceEquals(p, null) || p == null) return false;

                // 輸入狀態必須是 Action（排除 對話/過場/UI/轉場/抽符 等）
                var pib = p.playerInput;
                if (pib == null || pib.currentStateType != PlayerInputStateType.Action) return false;

                if (p.CurrentStateType != PlayerStateType.Normal) return false;
                if (p.IsScriptedMove) return false;
                if (p.lockMoving) return false;
                if (p.canMoveNode == null || !p.canMoveNode.gameObject.activeSelf) return false;
                return true;
            }
            catch { return false; }
        }

        // 離散動作注入：讓特定 PlayerAction 的 WasPressed 回傳 true
        private static void WasPressedPostfix(object __instance, ref bool __result)
        {
            var inst = Instance;
            if (ReferenceEquals(inst, null) || !inst._connected) return;
            if (!IsPlayerControllable(Player.i)) return;   // 過場/劇情時不注入動作
            var acts = inst.GetActions();
            if (acts == null) return;

            if      (inst._jumpPulse   > 0 && ReferenceEquals(__instance, acts.Jump))   __result = true;
            else if (inst._attackPulse > 0 && ReferenceEquals(__instance, acts.Attack)) __result = true;
            else if (inst._dodgePulse  > 0 && ReferenceEquals(__instance, acts.Dodge))  __result = true;
            else if (inst._parryPulse  > 0 && ReferenceEquals(__instance, acts.Parry))  __result = true;
        }

        private PlayerGamePlayActionSet _cachedActions;
        private PlayerGamePlayActionSet GetActions()
        {
            if (_cachedActions != null) return _cachedActions;
            try
            {
                var p = Player.i;
                if (p == null) return null;
                var pib = p.playerInput;
                if (pib == null) return null;
                _cachedActions = pib.gameplayActions;   // gameplayActions 在整場遊戲中固定，可永久快取
            }
            catch { }
            return _cachedActions;
        }

        // ============ 主迴圈 ============

        public void OnGameUpdate()
        {
            _debugTick++;

            // 遞減離散動作脈衝
            if (_jumpPulse   > 0) _jumpPulse--;
            if (_attackPulse > 0) _attackPulse--;
            if (_dodgePulse  > 0) _dodgePulse--;
            if (_parryPulse  > 0) _parryPulse--;

            if (!_connected) { TryConnect(); return; }

            // 讀取 Python 傳來的 action（非阻塞）
            try
            {
                if (_stream?.DataAvailable == true)
                {
                    var buf = new byte[4096];
                    int n = _stream.Read(buf, 0, buf.Length);
                    if (n > 0) ParseAction(Encoding.UTF8.GetString(buf, 0, n));
                    else { Disconnect(); return; }
                }
            }
            catch { Disconnect(); return; }

            // 每 6 tick 送一次 state（約 10Hz）
            if (_debugTick % 6 == 0)
            {
                var state = GetGameState();
                if (state != null) SendState(state);
            }

            if (_debugTick % 120 == 0)
            {
                var p = Player.i;
                string st = "null";
                try { if (p != null) st = p.CurrentStateType.ToString(); } catch { }
                Logger.LogInfo($"[RL] tick={_debugTick} move={_moveDir} " +
                               $"J={_jumpPulse} A={_attackPulse} D={_dodgePulse} P={_parryPulse} " +
                               $"hmcCalls={_hmcCalls} state={st}");
            }
        }

        private void TryConnect()
        {
            if (_debugTick < 300) return;

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
            _moveDir = 0;
            _jumpPulse = _attackPulse = _dodgePulse = _parryPulse = 0;
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
                var ph = player.GetComponentInChildren<PlayerHealth>(true);
                if (ph == null) return null;

                // ---- 玩家 ----
                float px = player.transform.position.x;
                float py = player.transform.position.y;
                float vx = player.VelX;                          // 真實速度（非差分）
                float vy = player.VelY;
                float facing  = player.Facing == Facings.Right ? 1f : -1f;
                bool  grounded = false;
                try { grounded = player.IsOnGround; } catch { }
                float php    = ph.CurrentHealthValue;
                float phpPct = 0f; try { phpPct = ph.percentage; } catch { }
                float injury = 0f; try { injury = ph.CurrentInternalInjury; } catch { }   // 內傷

                // ---- boss / 最近的怪 ----
                float bx = 0, by = 0, bvx = 0, bvy = 0, bFacing = 0, bhp = 0, bhpPct = 0;
                bool bossPresent = false;
                var boss = FindObjectOfType<MonsterBase>();
                if (boss != null)
                {
                    bx = boss.transform.position.x;
                    by = boss.transform.position.y;
                    try { bvx = boss.VelX; bvy = boss.VelY; } catch { }
                    try { bFacing = boss.Facing == Facings.Right ? 1f : -1f; } catch { }
                    try { bhp = boss.health.currentValue; bhpPct = boss.health.percentage; } catch { }
                    bossPresent = true;
                }
                float bdx = bossPresent ? bx - px : 0f;          // 相對位置
                float bdy = bossPresent ? by - py : 0f;

                bool done = (php <= 0) || (bossPresent && bhp <= 0);
                bool controllable = IsPlayerControllable(player);
                string state = player.CurrentStateType.ToString();

                return "{" +
                    $"\"px\":{px},\"py\":{py},\"vx\":{vx},\"vy\":{vy}," +
                    $"\"facing\":{facing},\"grounded\":{(grounded ? "true" : "false")}," +
                    $"\"php\":{php},\"php_pct\":{phpPct},\"internal_injury\":{injury}," +
                    $"\"boss_present\":{(bossPresent ? "true" : "false")}," +
                    $"\"bx\":{bx},\"by\":{by},\"bvx\":{bvx},\"bvy\":{bvy}," +
                    $"\"b_facing\":{bFacing},\"bdx\":{bdx},\"bdy\":{bdy}," +
                    $"\"bhp\":{bhp},\"bhp_pct\":{bhpPct}," +
                    $"\"controllable\":{(controllable ? "true" : "false")}," +
                    $"\"state\":\"{state}\"," +
                    $"\"done\":{(done ? "true" : "false")}" +
                    "}\n";
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
                // 一個封包可能含多筆 JSON，取最後一筆
                int last = json.LastIndexOf('{');
                if (last > 0) json = json.Substring(last);

                int move = ExtractInt(json, "move");
                _moveDir = move == 1 ? -1 : move == 2 ? 1 : 0;

                if (ExtractInt(json, "jump")   == 1) _jumpPulse   = PULSE;
                if (ExtractInt(json, "attack") == 1) _attackPulse = PULSE;
                if (ExtractInt(json, "dodge")  == 1) _dodgePulse  = PULSE;
                if (ExtractInt(json, "parry")  == 1) _parryPulse  = PULSE;

                // 相容舊欄位 skill（= attack）
                if (ExtractInt(json, "skill")  == 1) _attackPulse = PULSE;
            }
            catch { }
        }

        private static int ExtractInt(string json, string key)
        {
            string search = $"\"{key}\":";
            int idx = json.IndexOf(search);
            if (idx < 0) return 0;
            int start = idx + search.Length;
            while (start < json.Length && json[start] == ' ') start++;   // 跳過空白
            int end = start;
            while (end < json.Length && (char.IsDigit(json[end]) || json[end] == '-'))
                end++;
            if (end == start) return 0;
            return int.Parse(json.Substring(start, end - start));
        }

        void OnDestroy()
        {
            Disconnect();
        }
    }
}
