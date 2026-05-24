using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using BepInEx;
using UnityEngine;
using HarmonyLib;

namespace NineSolsRL
{
    [BepInPlugin("com.ninesolsrl.plugin", "NineSolsRL", "2.0.0")]
    public class Plugin : BaseUnityPlugin
    {
        public static Plugin Instance;
        internal static BepInEx.Logging.ManualLogSource Log;   // 純 managed logger，不受 Unity 物件狀態影響

        private NetworkStream _stream;
        private TcpClient _pendingClient;
        private Task _connectTask;
        private bool _connected = false;

        // ---- RL action 狀態 ----
        // move:   -1 左, 0 停, 1 右
        // dodge:  0 無 / 1 跳 / 2 衝刺
        // attack: 0 無 / 1 近戰 / 2 遠程 / 3 格檔 / 4 貼符 / 5 喝藥
        private int _moveDir = 0;
        private int _jumpPulse, _dashPulse, _meleePulse, _rangedPulse, _parryPulse, _talismanPulse, _healPulse;
        private const int PULSE = 3;                    // 離散動作維持的幀數

        // curriculum：boss 有效 HP 壓到 _bossHpScale × maxHP（1.0 = 不弱化）
        private float _bossHpScale = 1.0f;

        // hard reset：呼叫 Player.Suicide() → memory mode 下死亡會自動重啟此場 boss 戰
        private float _lastResetTime = 0f;

        // 診斷：上次 boss 偵測結果（給 [RL] log 用）
        private string _lastBossInfo = "none";

        // 格檔：_lastParryResult 0 無 / 1 不精確 / 2 精確；_parryCount 累計成功格檔次數（reward 用）
        private int _lastParryResult = 0;
        private int _parryCount = 0;

        // v2.0.0 dev flag：開著時每次 boss FSM 字串變動就印一行 [boss-fsm] log，
        // 用來收集介川 / 黃裳等 boss 的所有招式 FSM 字串，整理後填到
        // _bossFsmCategories 對映表。production build 應改回 false。
        // 邊訓邊熱補策略：保留 true，訓練期間 grep log 抓新字串。
        private const bool LOG_BOSS_FSM = true;
        private string _lastLoggedBossFsm = null;

        // v2.0.0 通用攻擊類別查表：每個 boss 一張 FSM 字串 → category 的表
        // 8 個通用 category（boss-agnostic）：
        //   0 idle / 1 parryable_windup / 2 unparryable_windup / 3 grab_windup
        //   4 ranged_windup / 5 attacking / 6 phase_change / 7 stunned
        // 未知 FSM fallback 到 0 (idle)。
        // Key 用 Unity 物件全名（含 StealthGameMonster_Boss_ 前綴），匹配最直接零歧義。
        private static readonly Dictionary<string, Dictionary<string, int>> _bossFsmCategories
            = new Dictionary<string, Dictionary<string, int>>
        {
            ["StealthGameMonster_Boss_JieChuan"] = new Dictionary<string, int>
            {
                // category:0 idle / 1 parryable_windup / 5 attacking / 7 stunned
                ["Attack1"]       = 5,  // attacking
                ["Attack2"]       = 5,  // attacking
                ["Attack3"]       = 5,  // attacking
                ["Attack4"]       = 5,  // attacking
                ["Attack5"]       = 5,  // attacking
                ["Attack6"]       = 5,  // attacking
                ["Attack7"]       = 5,  // attacking
                ["Attack8"]       = 5,  // attacking
                ["Attack9"]       = 5,  // attacking
                ["Attack10"]      = 5,  // attacking
                ["Attack11"]      = 5,  // attacking
                ["Attack12"]      = 5,  // attacking
                ["Attack13"]      = 5,  // attacking
                ["Attack14"]      = 5,  // attacking
                ["Engaging"]      = 0,  // idle:接近 / 走位
                ["Hurt_Big"]      = 7,  // stunned:boss 被大傷硬直
                ["PostureBreak"]  = 7,  // stunned:架勢崩潰,deathblow 機會 
                ["PreAttack"]     = 1,  // parryable_windup:前置攻擊
                ["TurnAround"]    = 0,  // idle:轉身
                ["Undefined"]     = 0,  // idle:FSM 過渡(顯式登錄)
                ["WanderingIdle"] = 0,  // idle:閒晃(顯式登錄)
                // ---- TBD----
                // ["BossAngry"] // 是否有無敵幀/紅光?選 0/1/6
                // ["LastHit"]   // 中場狀態,語意未明
            },
            // 未來新 boss：加 entry。例：
            // ["StealthGameMonster_Boss_HuangShang"] = new Dictionary<string, int> { ... },
        };

        // v2.0.0 每個 boss 的總階段數（NEAR 判定用：phase_count >= total_phases 才算最終階段）
        // 未知 boss fallback 1（無 phase gating，等同單階段 boss）。
        private static readonly Dictionary<string, int> _bossTotalPhases
            = new Dictionary<string, int>
        {
            ["StealthGameMonster_Boss_JieChuan"] = 2,
            // ["StealthGameMonster_Boss_HuangShang"] = 3,
        };

        private int _debugTick = 0;
        private float _lastStateTime = 0f;              // 上次送 state 的真實時間（穩定頻率用）
        private float _lastGameTime  = 0f;              // 上次送 state 的遊戲時間（算 dt 用）
        private const float SEND_INTERVAL = 1f / 30f;   // 固定 30Hz 送 state

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

        // 格檔結果偵測：玩家成功格檔時記錄精確/不精確
        [HarmonyPatch(typeof(ParriableAttackEffect), "EffectParried")]
        public static class PatchEffectParried
        {
            static void Postfix(ParryResultData data)
            {
                var inst = Instance;
                if (ReferenceEquals(inst, null)) return;
                try
                {
                    inst._lastParryResult = data.isAccurate ? 2 : 1;   // 精確=2 不精確=1
                    inst._parryCount++;                                // 累計成功格檔
                }
                catch { }
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

        // 受傷/倒地狀態：PlayerLieDownState / PlayerHurtState 的 OnStateUpdate 都會跑
        // DodgeCheck()，所以這些狀態下「閃避」可以快速起身/翻滾脫離。
        internal static bool IsHurtOrDown(Player p)
        {
            try
            {
                if (ReferenceEquals(p, null) || p == null) return false;
                var st = p.CurrentStateType;
                return st == PlayerStateType.LieDown
                    || st == PlayerStateType.Hurt
                    || st == PlayerStateType.HurtFlying;
            }
            catch { return false; }
        }

        // 離散動作注入：讓特定 PlayerAction 的 WasPressed 回傳 true
        private static void WasPressedPostfix(object __instance, ref bool __result)
        {
            var inst = Instance;
            if (ReferenceEquals(inst, null) || !inst._connected) return;
            var acts = inst.GetActions();
            if (acts == null) return;
            var p = Player.i;

            if (IsPlayerControllable(p))
            {
                // 正常可操作：所有動作都放行
                if      (inst._jumpPulse     > 0 && ReferenceEquals(__instance, acts.Jump))         __result = true;
                else if (inst._dashPulse     > 0 && ReferenceEquals(__instance, acts.Dodge))        __result = true;
                else if (inst._meleePulse    > 0 && ReferenceEquals(__instance, acts.Attack))       __result = true;
                else if (inst._rangedPulse   > 0 && ReferenceEquals(__instance, acts.WeaponAttack)) __result = true;
                else if (inst._parryPulse    > 0 && ReferenceEquals(__instance, acts.Parry))        __result = true;
                else if (inst._talismanPulse > 0 && ReferenceEquals(__instance, acts.FooAttack))    __result = true;
                else if (inst._healPulse     > 0 && ReferenceEquals(__instance, acts.Heal))         __result = true;
            }
            else if (IsHurtOrDown(p))
            {
                // 受傷/倒地：只放行「閃避」用來快速起身；其餘動作遊戲本來就會擋
                if (inst._dashPulse > 0 && ReferenceEquals(__instance, acts.Dodge)) __result = true;
            }
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
            if (_jumpPulse     > 0) _jumpPulse--;
            if (_dashPulse     > 0) _dashPulse--;
            if (_meleePulse    > 0) _meleePulse--;
            if (_rangedPulse   > 0) _rangedPulse--;
            if (_parryPulse    > 0) _parryPulse--;
            if (_talismanPulse > 0) _talismanPulse--;
            if (_healPulse     > 0) _healPulse--;

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

            // 綁真實時間送 state（固定 30Hz，不隨 fps 浮動）；
            // dt 用「遊戲時間」算 → 之後遊戲加速時，reward 仍能正確換算
            if (Time.unscaledTime - _lastStateTime >= SEND_INTERVAL)
            {
                float dt = Time.time - _lastGameTime;
                _lastGameTime  = Time.time;
                _lastStateTime = Time.unscaledTime;
                var state = GetGameState(dt);
                if (state != null) SendState(state);
            }

            if (_debugTick % 120 == 0)
            {
                var p = Player.i;
                string st = "null";
                try { if (p != null) st = p.CurrentStateType.ToString(); } catch { }
                Logger.LogInfo($"[RL] tick={_debugTick} move={_moveDir} " +
                               $"jump={_jumpPulse} dash={_dashPulse} melee={_meleePulse} " +
                               $"ranged={_rangedPulse} parry={_parryPulse} talis={_talismanPulse} " +
                               $"heal={_healPulse} bossHpScale={_bossHpScale:F2} " +
                               $"boss={_lastBossInfo} state={st}");
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
            _jumpPulse = _dashPulse = _meleePulse = _rangedPulse = _parryPulse = _talismanPulse = _healPulse = 0;
            Logger.LogInfo("Python 斷線，重新嘗試連線...");
            try { _stream?.Close(); } catch { }
            try { _pendingClient?.Close(); } catch { }
            _stream = null;
            _pendingClient = null;
            _connectTask = null;
            _connected = false;
        }

        private string GetGameState(float dt)
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

                // 氣（Qi）—— 回血與貼符共用的資源
                float qi = 0f, qiMax = 1f, qiPct = 0f;
                try
                {
                    var en = player.ammo;
                    if (en != null) { qi = en.Value; qiMax = en.MaxValue; qiPct = en.percentage; }
                }
                catch { }

                // ---- boss 偵測 ----
                float bx = 0, by = 0, bvx = 0, bvy = 0, bFacing = 0, bhp = 0, bhpPct = 0, bhpMax = 0;
                int bossFsm = 0;
                int attackCategory = 0;   // v2.0.0：通用攻擊類別（0-7），未知 boss/FSM 預設 idle
                int totalPhases = 1;      // v2.0.0：boss 總階段數，未知 boss 預設 1
                bool bossPresent = false, bossDead = false;

                // 主要：遊戲權威的「當前 boss 血條」→ PostureSystem → MonsterBase
                MonsterBase boss = null;
                string bossSrc = "none";
                try
                {
                    var hpUI = GameCore.Instance != null ? GameCore.Instance.monsterHpUI : null;
                    var bossHpUI = hpUI != null ? hpUI.CurrentBossHP : null;
                    if (bossHpUI != null && bossHpUI.bindingPosture != null)
                    {
                        boss = bossHpUI.bindingPosture.BindMonster;
                        if (boss != null) bossSrc = "ui";
                    }
                }
                catch { }
                // 後備：UI 沒給 boss 血條 → 掃場上等級最高的怪
                if (boss == null)
                {
                    int bestLevel = int.MinValue;
                    foreach (var m in FindObjectsOfType<MonsterBase>())
                    {
                        if (m == null) continue;
                        int lvl = 0;
                        try { lvl = (int)m.monsterStat.monsterLevel; } catch { }
                        if (lvl > bestLevel) { bestLevel = lvl; boss = m; }
                    }
                    if (boss != null) bossSrc = "scan";
                }
                _lastBossInfo = boss != null ? boss.name + "(" + bossSrc + ")" : "none";
                if (boss != null)
                {
                    bx = boss.transform.position.x;
                    by = boss.transform.position.y;
                    try { bvx = boss.VelX; bvy = boss.VelY; } catch { }
                    try { bFacing = boss.Facing == Facings.Right ? 1f : -1f; } catch { }
                    try
                    {
                        // boss 真實血量 = PostureSystem（架勢系統），不是 MonsterBase.health
                        var ps = boss.postureSystem;
                        float maxHp = ps.MaxPostureValue;
                        bhpMax = maxHp;                              // 未縮放滿血
                        float cap = _bossHpScale * maxHp;
                        // curriculum：把 posture 壓到 cap = scale × maxHP
                        if (_bossHpScale < 1f && ps.PostureValue > cap)
                            ps.SetPostureValue((int)cap);
                        float remain = ps.RemainTotal;               // posture + 內傷 = 目前血量
                        bhp = remain;
                        // bhp_pct 相對於 cap 回報 → 不論難度 boss 都「從 100% 開始」
                        bhpPct = cap > 0f ? Mathf.Clamp01(remain / cap) : 0f;
                    }
                    catch { }
                    // v2.0.0 通用攻擊類別查表：取代舊的 bWindup/bAttacking substring 啟發式。
                    // 未登錄 boss → fallback idle (0)、totalPhases=1
                    // 未登錄 FSM → fallback idle (0)
                    try
                    {
                        var cs = boss.CurrentState;
                        bossFsm = (int)cs;
                        string csName = cs.ToString();
                        string bossName = boss.name;

                        if (_bossFsmCategories.TryGetValue(bossName, out var fsmMap)
                            && fsmMap.TryGetValue(csName, out int cat))
                        {
                            attackCategory = cat;
                        }
                        _bossTotalPhases.TryGetValue(bossName, out totalPhases);
                        if (totalPhases <= 0) totalPhases = 1;

                        // dev：FSM 字串變動時印一行，邊訓邊收集新字串（未在 _bossFsmCategories 內 → 補表）
                        if (LOG_BOSS_FSM && csName != _lastLoggedBossFsm)
                        {
                            Logger.LogInfo($"[boss-fsm] name={bossName} fsm={csName} (int={bossFsm}) cat={attackCategory}");
                            _lastLoggedBossFsm = csName;
                        }
                    }
                    catch { }
                    // 真死才 true；換階段 (BossPhaseChangeState) 時 IsDead() 為 false
                    // → 2 階段 boss 不會在一階清掉時誤判 done
                    try { bossDead = boss.IsDead(); }
                    catch { }
                    bossPresent = true;
                }
                float bdx = bossPresent ? bx - px : 0f;          // 相對位置
                float bdy = bossPresent ? by - py : 0f;

                bool done = (php <= 0) || bossDead;
                // reset 後 3 秒內強制 controllable=false → 確保 Python reset 迴圈會等
                // 完整個場景重載（fade + load + 進場動畫），不會在舊場景提早 break
                bool controllable = IsPlayerControllable(player)
                                    && (Time.time - _lastResetTime > 3f);
                bool knockedDown  = IsHurtOrDown(player);   // 受傷/倒地（可用閃避起身）
                string state = player.CurrentStateType.ToString();

                return "{" +
                    $"\"px\":{px},\"py\":{py},\"vx\":{vx},\"vy\":{vy}," +
                    $"\"facing\":{facing},\"grounded\":{(grounded ? "true" : "false")}," +
                    $"\"php\":{php},\"php_pct\":{phpPct},\"internal_injury\":{injury}," +
                    $"\"qi\":{qi},\"qi_max\":{qiMax},\"qi_pct\":{qiPct}," +
                    $"\"last_parry_result\":{_lastParryResult},\"parry_count\":{_parryCount}," +
                    $"\"boss_present\":{(bossPresent ? "true" : "false")}," +
                    $"\"boss_name\":\"{(boss != null ? boss.name : "")}\"," +
                    $"\"bx\":{bx},\"by\":{by},\"bvx\":{bvx},\"bvy\":{bvy}," +
                    $"\"b_facing\":{bFacing},\"bdx\":{bdx},\"bdy\":{bdy}," +
                    $"\"bhp\":{bhp},\"bhp_pct\":{bhpPct},\"bhp_max\":{bhpMax}," +
                    $"\"boss_fsm\":{bossFsm}," +
                    $"\"attack_category\":{attackCategory}," +
                    $"\"total_phases\":{totalPhases}," +
                    $"\"controllable\":{(controllable ? "true" : "false")}," +
                    $"\"knocked_down\":{(knockedDown ? "true" : "false")}," +
                    $"\"dt\":{dt}," +
                    $"\"state\":\"{state}\"," +
                    $"\"boss_dead\":{(bossDead ? "true" : "false")}," +
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

                // reset 指令：要求重載 JieChuan 挑戰（確定性 episode reset）
                if (json.Replace(" ", "").Contains("\"reset\":true"))
                {
                    DoHardReset();
                    return;
                }

                // move: 0 停 / 1 左 / 2 右
                int move = ExtractInt(json, "move");
                _moveDir = move == 1 ? -1 : move == 2 ? 1 : 0;

                // dodge: 0 無 / 1 跳 / 2 衝刺
                int dodge = ExtractInt(json, "dodge");
                if      (dodge == 1) _jumpPulse = PULSE;
                else if (dodge == 2) _dashPulse = PULSE;

                // attack: 0 無 / 1 近戰 / 2 遠程 / 3 格檔 / 4 貼符 / 5 喝藥
                int atk = ExtractInt(json, "attack");
                if      (atk == 1) _meleePulse    = PULSE;
                else if (atk == 2) _rangedPulse   = PULSE;
                else if (atk == 3) _parryPulse    = PULSE;
                else if (atk == 4) _talismanPulse = PULSE;
                else if (atk == 5) _healPulse     = PULSE;

                // boss_hp_scale: curriculum 弱化倍率（封包沒帶就維持原值）
                _bossHpScale = ExtractFloat(json, "boss_hp_scale", _bossHpScale);
            }
            catch { }
        }

        // hard reset：呼叫 Player.Suicide()（= 暫停選單「重置挑戰」按鈕本體）。
        // memory mode 下玩家死亡 → 遊戲自動重啟此場 boss 戰。截斷/敗/勝皆適用。
        private void DoHardReset()
        {
            if (Time.time - _lastResetTime < 3f) return;   // 去抖：避免短時間重複觸發
            _lastResetTime = Time.time;
            _moveDir = 0;
            _jumpPulse = _dashPulse = _meleePulse = _rangedPulse = _parryPulse = _talismanPulse = _healPulse = 0;
            try
            {
                Player.i.Suicide();   // health=-1 + DeadCheck，硬殺、繞過無敵
                Logger.LogInfo("[reset] Player.Suicide() → 重啟記憶戰鬥");
            }
            catch (Exception e) { Logger.LogError("[reset] " + e); }
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

        // 解析浮點數（curriculum 的 boss_hp_scale 用）；key 不存在或解析失敗回傳 fallback。
        private static float ExtractFloat(string json, string key, float fallback)
        {
            string search = $"\"{key}\":";
            int idx = json.IndexOf(search);
            if (idx < 0) return fallback;
            int start = idx + search.Length;
            while (start < json.Length && json[start] == ' ') start++;
            int end = start;
            while (end < json.Length &&
                   (char.IsDigit(json[end]) || json[end] == '-' || json[end] == '+' ||
                    json[end] == '.' || json[end] == 'e' || json[end] == 'E'))
                end++;
            if (end == start) return fallback;
            return float.TryParse(json.Substring(start, end - start),
                                  System.Globalization.NumberStyles.Float,
                                  System.Globalization.CultureInfo.InvariantCulture,
                                  out float v) ? v : fallback;
        }

        void OnDestroy()
        {
            Disconnect();
        }
    }
}
