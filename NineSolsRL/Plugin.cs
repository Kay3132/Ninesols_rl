using System;
using System.Collections.Generic;
using System.Globalization;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using BepInEx;
using BepInEx.Logging;
using HarmonyLib;
using UnityEngine;
using Object = UnityEngine.Object;

namespace NineSolsRL;

[BepInPlugin("com.ninesolsrl.plugin", "NineSolsRL", "2.1.0")]
public class Plugin : BaseUnityPlugin
{
	[HarmonyPatch(typeof(GameCore), "Update")]
	public static class PatchGameCoreUpdate
	{
		private static void Postfix()
		{
			Instance?.OnGameUpdate();
		}
	}

	[HarmonyPatch(typeof(Player), "HorizontalMoveCheck")]
	public static class PatchHorizontalMoveCheck
	{
		private static int _logTick;

		private static void Postfix(Player __instance)
		{
			_hmcCalls++;
			// 2026-06-15 Round 21 fix: 跟 WasPressed/IsPressed 同問題 ——
			// `Plugin instance = Instance` 在 postfix 上下文被 Unity 判 destroyed(null),
			// 即使 Awake 已設好。改用 s_moveDir 靜態鏡像,完全繞過 Instance。
			// 2026-06-15 Round 21 fix v2: ctrl+c 後 s_connected=false → 不干預鍵盤輸入。
			if (!s_connected)
			{
				return;
			}
			if (__instance != null && IsPlayerControllable(__instance))
			{
				int moveDir = s_moveDir;
				float num;
				try
				{
					num = __instance.MaxRunStat.Value;
				}
				catch
				{
					num = 8f;
				}
				if (moveDir != 0)
				{
					__instance.moveX = moveDir;
					((PhysicsMover)__instance).VelX = num * (float)moveDir;
				}
				else
				{
					__instance.moveX = 0;
				}
			}
		}
	}

	[HarmonyPatch(typeof(ParriableAttackEffect), "EffectParried")]
	public static class PatchEffectParried
	{
		private static void Postfix(ParryResultData data)
		{
			//IL_000c: Unknown result type (might be due to invalid IL or missing references)
			// 2026-06-15 Round 21 fix: 跟 HorizontalMoveCheck 同 bug ——
			// `Plugin instance = Instance` 在 postfix 上下文被 Unity 判 destroyed → 從沒 ++。
			// 改寫靜態欄位完全繞過 Instance。
			try
			{
				s_lastParryResult = ((!data.isAccurate) ? 1 : 2);
				s_parryCount++;
			}
			catch
			{
			}
		}
	}

	[HarmonyPatch(typeof(PlayerParryState), "OnStateEnter")]
	public static class PatchParryEnter
	{
		private static void Postfix()
		{
			// 2026-06-15 Round 21 fix: 同 PatchEffectParried 改靜態欄位
			try
			{
				s_parryEnterCount++;
			}
			catch
			{
			}
		}
	}

	public static Plugin Instance;

	internal static ManualLogSource Log;

	private NetworkStream _stream;

	private TcpClient _pendingClient;

	private Task _connectTask;

	private bool _connected;

	private int _moveDir;

	private int _jumpPulse;

	private int _dashPulse;

	private int _meleePulse;

	private int _rangedPulse;

	private int _parryPulse;

	private int _talismanPulse;

	private int _healPulse;

	private const int PULSE = 10;

	private float _bossHpScale = 1f;

	private float _lastResetTime;

	private string _lastBossInfo = "none";

	// 2026-06-15 Round 21 fix: 改 static,讓 PatchParryEnter / PatchEffectParried postfix
	// 不用透過 Plugin.Instance 寫(Instance 在 postfix 上下文被 Unity 判 null → 從沒 ++)。
	private static int s_lastParryResult;

	private static int s_parryCount;

	private static int s_parryEnterCount;

	private const bool LOG_BOSS = true;

	private string _lastLoggedBossSig;

	private static readonly Dictionary<string, Dictionary<string, int>> _bossFsmCategories = new Dictionary<string, Dictionary<string, int>> { ["StealthGameMonster_Boss_JieChuan"] = new Dictionary<string, int>
	{
		["Attack1"] = 5,
		["Attack2"] = 5,
		["Attack3"] = 5,
		["Attack4"] = 5,
		["Attack5"] = 5,
		["Attack6"] = 5,
		["Attack7"] = 5,
		["Attack8"] = 5,
		["Attack9"] = 5,
		["Attack10"] = 5,
		["Attack11"] = 5,
		["Attack12"] = 5,
		["Attack13"] = 5,
		["Attack14"] = 5,
		["Engaging"] = 0,
		["Hurt_Big"] = 7,
		["PostureBreak"] = 7,
		["PreAttack"] = 1,
		["TurnAround"] = 0,
		["Undefined"] = 0,
		["WanderingIdle"] = 0,
		["LastHit"] = 5
	} };

	private static readonly Dictionary<string, int> _bossTotalPhases = new Dictionary<string, int> { ["StealthGameMonster_Boss_JieChuan"] = 2 };

	private static readonly Dictionary<string, Dictionary<string, int>> _bossStatePhaseIndex = new Dictionary<string, Dictionary<string, int>> { ["StealthGameMonster_Boss_JieChuan"] = new Dictionary<string, int>
	{
		["PreAttack"] = 1,
		["Attack1"] = 1,
		["Attack2"] = 1,
		["Attack3"] = 1,
		["Attack4"] = 1,
		["Attack5"] = 1,
		["Attack6"] = 1,
		["Attack7"] = 1,
		["Attack8"] = 1,
		["Attack9"] = 1,
		["Attack10"] = 1,
		["Engaging"] = 1,
		["TurnAround"] = 1,
		["WanderingIdle"] = 1,
		["Undefined"] = 1,
		["Hurt_Big"] = 1,
		["Attack11"] = 2,
		["Attack12"] = 2,
		["Attack13"] = 2,
		["Attack14"] = 2,
		["LastHit"] = 2,
		["PostureBreak"] = 2
	} };

	private static readonly HashSet<string> _warnedMissingTotalPhaseBoss = new HashSet<string>();

	private static readonly HashSet<string> _warnedMissingPhaseMapBoss = new HashSet<string>();

	private static readonly HashSet<string> _warnedMissingPhaseState = new HashSet<string>();

	public const int MAX_ATTACK_IDS = 20;

	private static readonly Dictionary<string, Dictionary<string, int>> _bossAttackIds = new Dictionary<string, Dictionary<string, int>> { ["StealthGameMonster_Boss_JieChuan"] = new Dictionary<string, int>
	{
		["PreAttack"] = 0,
		["Attack1"] = 1,
		["Attack2"] = 2,
		["Attack3"] = 3,
		["Attack4"] = 4,
		["Attack5"] = 5,
		["Attack6"] = 6,
		["Attack7"] = 7,
		["Attack8"] = 8,
		["Attack9"] = 9,
		["Attack11"] = 10,
		["Attack12"] = 11,
		["Attack14"] = 12,
		["Hurt_Big"] = 13,
		["PostureBreak"] = 14,
		["LastHit"] = 15
	} };

	private int _debugTick;

	private float _lastStateTime;

	private float _lastGameTime;

	private const float SEND_INTERVAL = 1f / 30f;

	public static long _hmcCalls = 0L;

	// 2026-06-14 fix: Plugin.Instance 在 WasPressed/IsPressed postfix 上下文被 Unity 視為 destroyed (=null),
	// 即使 OnGameUpdate 正常跑。改用 static 鏡像欄位,postfix 完全不依賴 Instance。
	// 2026-06-15 Round 21 fix: HorizontalMoveCheck postfix 上下文 Plugin.Instance 被 Unity 判 null,
	// 改用 s_moveDir 靜態鏡像
	private static int s_moveDir;

	// 2026-06-15 Round 21 fix v2: ctrl+c 後若 Postfix 仍跑(s_moveDir=0 強制覆寫 moveX=0)
	// 會吃掉玩家鍵盤輸入。s_connected=false 時 Postfix 整段跳過,讓 game 自然處理。
	private static bool s_connected;

	// 2026-06-15 Round 21: 蓄力擊 lock —— attack=6 觸發後 70 game frame 內,
	// 整個 attack switch 被跳過。雙重保護:case 1 Math.Max 防止覆寫,lock 防止
	// 其他 attack(parry/ranged/heal)中斷 charge。
	private int _chargeLockFrames;
	private static int s_jumpPulse;
	private static int s_dashPulse;
	private static int s_meleePulse;
	private static int s_rangedPulse;
	private static int s_parryPulse;
	private static int s_talismanPulse;
	private static int s_healPulse;
	private static PlayerGamePlayActionSet s_actions;

	// 2026-06-15 Round 20: edge detection
	// WasPressed 只在「從非按到按」的瞬間 true(1 game frame),IsPressed 持續按住期間 true。
	// 這樣 game 不會誤認 hold pattern 為連續多次按下(雙跳第一跳同時用掉兩個 jump 用量的 bug)。
	private int _prevDodge;
	private int _prevAttack;
	private int _jumpEdge;     // 1 game frame 後衰減
	private int _dashEdge;
	private int _meleeEdge;
	private int _rangedEdge;
	private int _parryEdge;
	private int _talismanEdge;
	private int _healEdge;
	private static int s_jumpEdge;
	private static int s_dashEdge;
	private static int s_meleeEdge;
	private static int s_rangedEdge;
	private static int s_parryEdge;
	private static int s_talismanEdge;
	private static int s_healEdge;

	private PlayerGamePlayActionSet _cachedActions;

	private static bool TryReadShieldInfo(MonsterBase boss, out bool hasShieldObj, out bool shieldActive, out float shieldCur, out float shieldMax)
	{
		hasShieldObj = false;
		shieldActive = false;
		shieldCur = 0f;
		shieldMax = 0f;
		if (boss == null || (Object)(object)boss == (Object)null)
		{
			return false;
		}
		try
		{
			MonsterShield currentShield = boss.currentShield;
			if (currentShield == null || (Object)(object)currentShield == (Object)null)
			{
				return true;
			}
			hasShieldObj = true;
			shieldActive = ((Behaviour)currentShield).isActiveAndEnabled && (Object)(object)((Component)currentShield).gameObject != (Object)null && ((Component)currentShield).gameObject.activeInHierarchy;
			shieldCur = currentShield.ShieldCurrentHP;
			shieldMax = currentShield.ShieldMaxHP;
			return true;
		}
		catch
		{
			return false;
		}
	}

	private void Awake()
	{
		//IL_0021: Unknown result type (might be due to invalid IL or missing references)
		//IL_0027: Expected O, but got Unknown
		//IL_0066: Unknown result type (might be due to invalid IL or missing references)
		//IL_0073: Expected O, but got Unknown
		Instance = this;
		Log = Logger;
		Logger.LogInfo((object)"[NineSolsRL] Awake v2.1.0 啟動");
		UnityEngine.Object.DontDestroyOnLoad((Object)(object)((Component)this).gameObject);
		Harmony val = new Harmony("com.ninesolsrl.plugin");
		val.PatchAll();
		try
		{
			Type type = AccessTools.TypeByName("InControl.OneAxisInputControl");
			if (type != null)
			{
				MethodInfo methodInfo = AccessTools.PropertyGetter(type, "WasPressed");
				val.Patch((MethodBase)methodInfo, (HarmonyMethod)null, new HarmonyMethod(typeof(Plugin).GetMethod("WasPressedPostfix", BindingFlags.Static | BindingFlags.NonPublic)), (HarmonyMethod)null, (HarmonyMethod)null, (HarmonyMethod)null);
				Logger.LogInfo((object)"[NineSolsRL] InControl.OneAxisInputControl.WasPressed patched");
				MethodInfo isPressedGetter = AccessTools.PropertyGetter(type, "IsPressed");
				if (isPressedGetter != null)
				{
					val.Patch((MethodBase)isPressedGetter, (HarmonyMethod)null, new HarmonyMethod(typeof(Plugin).GetMethod("IsPressedPostfix", BindingFlags.Static | BindingFlags.NonPublic)), (HarmonyMethod)null, (HarmonyMethod)null, (HarmonyMethod)null);
					Logger.LogInfo((object)"[NineSolsRL] InControl.OneAxisInputControl.IsPressed patched");
				}
			}
			else
			{
				Logger.LogError((object)"[NineSolsRL] 找不到 InControl.OneAxisInputControl 型別，離散動作將無法注入");
			}
		}
		catch (Exception ex)
		{
			Logger.LogError((object)("[NineSolsRL] InControl patch 失敗: " + ex));
		}
	}

	// 2026-06-16 Round 26 fix v2: 只擋 cutscene / scripted,不擋 player state machine
	// (在 attack/dodge state 期間仍允許輸入,模擬真實鍵盤可打斷自己動作的行為)。
	// 跟 IsPlayerControllable 差別:不檢查 p.CurrentStateType。
	internal static bool IsGameInteractable(Player p)
	{
		try
		{
			if (p == null || (Object)(object)p == (Object)null) return false;
			PlayerInputBinder playerInput = p.playerInput;
			if ((Object)(object)playerInput == (Object)null || (int)playerInput.currentStateType != 0) return false;
			if (p.IsScriptedMove) return false;
			if (p.lockMoving) return false;
			if ((Object)(object)p.canMoveNode == (Object)null || !((Component)p.canMoveNode).gameObject.activeSelf) return false;
			return true;
		}
		catch
		{
			return false;
		}
	}

	internal static bool IsPlayerControllable(Player p)
	{
		//IL_0021: Unknown result type (might be due to invalid IL or missing references)
		//IL_002d: Unknown result type (might be due to invalid IL or missing references)
		try
		{
			if (p == null || (Object)(object)p == (Object)null)
			{
				return false;
			}
			PlayerInputBinder playerInput = p.playerInput;
			if ((Object)(object)playerInput == (Object)null || (int)playerInput.currentStateType != 0)
			{
				return false;
			}
			if ((int)p.CurrentStateType != 0)
			{
				return false;
			}
			if (p.IsScriptedMove)
			{
				return false;
			}
			if (p.lockMoving)
			{
				return false;
			}
			if ((Object)(object)p.canMoveNode == (Object)null || !((Component)p.canMoveNode).gameObject.activeSelf)
			{
				return false;
			}
			return true;
		}
		catch
		{
			return false;
		}
	}

	internal static bool IsHurtOrDown(Player p)
	{
		//IL_0011: Unknown result type (might be due to invalid IL or missing references)
		//IL_0016: Unknown result type (might be due to invalid IL or missing references)
		//IL_0017: Unknown result type (might be due to invalid IL or missing references)
		//IL_001a: Invalid comparison between Unknown and I4
		//IL_001c: Unknown result type (might be due to invalid IL or missing references)
		//IL_001f: Invalid comparison between Unknown and I4
		//IL_0021: Unknown result type (might be due to invalid IL or missing references)
		//IL_0024: Invalid comparison between Unknown and I4
		try
		{
			if (p == null || (Object)(object)p == (Object)null)
			{
				return false;
			}
			PlayerStateType currentStateType = p.CurrentStateType;
			return (int)currentStateType == 33 || (int)currentStateType == 26 || (int)currentStateType == 41;
		}
		catch
		{
			return false;
		}
	}

	private static void WasPressedPostfix(object __instance, ref bool __result)
	{
		// 完全用 static 欄位,繞過 Plugin.Instance(在 postfix 上下文被 Unity 視為 destroyed)
		PlayerGamePlayActionSet actions = s_actions;
		if (actions == null)
		{
			return;
		}
		// 2026-06-16 Round 26 v2: 用 IsGameInteractable(只擋 cutscene、scripted move、
		// lockMoving、canMoveNode 不 active)而不是 IsPlayerControllable(會擋 player state)。
		// 王出場演出時 input 不會跑進去(cutscene gate),但 player 在 attack/dodge state
		// 期間 input 仍能 fake(讓 parry 取消輕擊動畫 → 進 ParryState)。
		Player p = Player.i;
		if (!IsGameInteractable(p))
		{
			return;
		}
		if (s_jumpEdge > 0 && __instance == actions.Jump)
		{
			__result = true;
		}
		else if (s_dashEdge > 0 && __instance == actions.Dodge)
		{
			__result = true;
		}
		else if (s_meleeEdge > 0 && __instance == actions.Attack)
		{
			__result = true;
		}
		else if (s_rangedEdge > 0 && __instance == actions.WeaponAttack)
		{
			__result = true;
		}
		else if (s_parryEdge > 0 && __instance == actions.Parry)
		{
			__result = true;
		}
		else if (s_talismanEdge > 0 && __instance == actions.FooAttack)
		{
			__result = true;
		}
		else if (s_healEdge > 0 && __instance == actions.Heal)
		{
			__result = true;
		}
	}

	// 2026-06-14: IsPressed patch（保守版,只 Jump + Attack）
	// Jump:解 in-air 雙跳(遊戲端 in-air jump 邏輯查 IsPressed 確認真的按住)
	// Attack:agent 連送 attack=1 時 IsPressed 持續 true → 觸發蓄力擊(破盾)
	// 其他動作維持只 patch WasPressed,避免遠程/貼符/喝藥意外變成「持續按住」
	private static void IsPressedPostfix(object __instance, ref bool __result)
	{
		PlayerGamePlayActionSet actions = s_actions;
		if (actions == null)
		{
			return;
		}
		// 2026-06-16 Round 26 v2: 加 IsGameInteractable gate 擋 cutscene(王出場演出時
		// 不應該送 IsPressed=true 進去)。Round 23 拿掉 IsPlayerControllable 是對的
		// (允許 attack state 期間 hold = 蓄力),但完全沒擋會在 cutscene 期間 leak。
		Player p = Player.i;
		if (!IsGameInteractable(p))
		{
			return;
		}
		// IsPressed 給 Jump(維持 variable jump height)+ Attack(蓄力擊破盾)
		if (s_jumpPulse > 0 && __instance == actions.Jump)
		{
			__result = true;
		}
		else if (s_meleePulse > 0 && __instance == actions.Attack)
		{
			__result = true;
		}
	}

	private PlayerGamePlayActionSet GetActions()
	{
		if (_cachedActions != null)
		{
			return _cachedActions;
		}
		try
		{
			Player i = Player.i;
			if ((Object)(object)i == (Object)null)
			{
				return null;
			}
			PlayerInputBinder playerInput = i.playerInput;
			if ((Object)(object)playerInput == (Object)null)
			{
				return null;
			}
			_cachedActions = playerInput.gameplayActions;
			s_actions = _cachedActions;
		}
		catch
		{
		}
		return _cachedActions;
	}

	public void OnGameUpdate()
	{
		_debugTick++;
		// 2026-06-14 fix: 主動 prime s_actions 給 postfix 用(postfix 上下文無法 call instance method)
		if (s_actions == null)
		{
			GetActions();
		}
		if (_jumpPulse > 0)
		{
			_jumpPulse--;
		}
		s_jumpPulse = _jumpPulse;
		if (_dashPulse > 0)
		{
			_dashPulse--;
		}
		s_dashPulse = _dashPulse;
		if (_meleePulse > 0)
		{
			_meleePulse--;
		}
		s_meleePulse = _meleePulse;
		if (_rangedPulse > 0)
		{
			_rangedPulse--;
		}
		s_rangedPulse = _rangedPulse;
		if (_parryPulse > 0)
		{
			_parryPulse--;
		}
		s_parryPulse = _parryPulse;
		if (_talismanPulse > 0)
		{
			_talismanPulse--;
		}
		s_talismanPulse = _talismanPulse;
		if (_healPulse > 0)
		{
			_healPulse--;
		}
		s_healPulse = _healPulse;
		// 2026-06-15 Round 21: 蓄力擊 lock 倒數
		if (_chargeLockFrames > 0) { _chargeLockFrames--; }
		// 2026-06-15 Round 20: edge fields 衰減(只活 1 game frame)
		if (_jumpEdge > 0) { _jumpEdge--; }
		s_jumpEdge = _jumpEdge;
		if (_dashEdge > 0) { _dashEdge--; }
		s_dashEdge = _dashEdge;
		if (_meleeEdge > 0) { _meleeEdge--; }
		s_meleeEdge = _meleeEdge;
		if (_rangedEdge > 0) { _rangedEdge--; }
		s_rangedEdge = _rangedEdge;
		if (_parryEdge > 0) { _parryEdge--; }
		s_parryEdge = _parryEdge;
		if (_talismanEdge > 0) { _talismanEdge--; }
		s_talismanEdge = _talismanEdge;
		if (_healEdge > 0) { _healEdge--; }
		s_healEdge = _healEdge;
		// 2026-06-15 Round 23: 蓄力 lock 期間每 game frame refresh _meleePulse=PULSE,
		// 模仿 test_charged_attack 的「每 step 重設 pulse=10」pattern,
		// 因為 game 端不認一次性大 pulse + 自然衰減的 IsPressed=true 軌跡。
		// lock 衰到 0 後不再 refresh,_meleePulse 自然從 PULSE 衰減 → 0,
		// game 看到 IsPressed 轉 false → 觸發釋放動畫。
		if (_chargeLockFrames > 0)
		{
			_meleePulse = PULSE;
			s_meleePulse = PULSE;
		}
		if (!_connected)
		{
			TryConnect();
			return;
		}
		try
		{
			NetworkStream stream = _stream;
			if (stream != null && stream.DataAvailable)
			{
				byte[] array = new byte[4096];
				int num = _stream.Read(array, 0, array.Length);
				if (num <= 0)
				{
					Disconnect();
					return;
				}
				ParseAction(Encoding.UTF8.GetString(array, 0, num));
			}
		}
		catch
		{
			Disconnect();
			return;
		}
		if (Time.unscaledTime - _lastStateTime >= 1f / 30f)
		{
			float dt = Time.time - _lastGameTime;
			_lastGameTime = Time.time;
			_lastStateTime = Time.unscaledTime;
			string gameState = GetGameState(dt);
			if (gameState != null)
			{
				SendState(gameState);
			}
		}
	}

	private void TryConnect()
	{
		if (_debugTick < 300)
		{
			return;
		}
		if (_connectTask != null)
		{
			if (!_connectTask.IsCompleted)
			{
				return;
			}
			if (!_connectTask.IsFaulted && _pendingClient != null && _pendingClient.Connected)
			{
				_stream = _pendingClient.GetStream();
				_connected = true;
				s_connected = true;   // 2026-06-15 Round 21 fix: sync static mirror for postfix
			}
			else
			{
				try
				{
					_pendingClient?.Close();
				}
				catch
				{
				}
				_pendingClient = null;
			}
			_connectTask = null;
		}
		else if (_debugTick % 180 == 0)
		{
			_pendingClient = new TcpClient();
			try
			{
				_connectTask = _pendingClient.ConnectAsync("127.0.0.1", 19271);
			}
			catch (Exception)
			{
				_pendingClient = null;
			}
		}
	}

	private void Disconnect()
	{
		_moveDir = 0;
		_jumpPulse = (_dashPulse = (_meleePulse = (_rangedPulse = (_parryPulse = (_talismanPulse = (_healPulse = 0))))));
		// 2026-06-15 Round 21 fix: ctrl+c 斷線後若沒同步靜態鏡像,HorizontalMoveCheck.Postfix
		// 仍會用最後一次的 s_moveDir → 角色自己走。InControl postfix 同理會繼續送 fake input。
		s_moveDir = 0;
		s_jumpPulse = s_dashPulse = s_meleePulse = s_rangedPulse =
			s_parryPulse = s_talismanPulse = s_healPulse = 0;
		s_jumpEdge = s_dashEdge = s_meleeEdge = s_rangedEdge =
			s_parryEdge = s_talismanEdge = s_healEdge = 0;
		_chargeLockFrames = 0;
		_prevDodge = 0;
		_prevAttack = 0;
		try
		{
			_stream?.Close();
		}
		catch
		{
		}
		try
		{
			_pendingClient?.Close();
		}
		catch
		{
		}
		_stream = null;
		_pendingClient = null;
		_connectTask = null;
		_connected = false;
		s_connected = false;   // 2026-06-15 Round 21 fix v2: 讓 Postfix 完全不干預鍵盤
	}

	private string GetGameState(float dt)
	{
		//IL_0036: Unknown result type (might be due to invalid IL or missing references)
		//IL_0047: Unknown result type (might be due to invalid IL or missing references)
		//IL_0063: Unknown result type (might be due to invalid IL or missing references)
		//IL_0069: Invalid comparison between Unknown and I4
		//IL_0373: Unknown result type (might be due to invalid IL or missing references)
		//IL_0379: Invalid comparison between Unknown and I4
		//IL_0403: Unknown result type (might be due to invalid IL or missing references)
		//IL_0408: Unknown result type (might be due to invalid IL or missing references)
		//IL_040a: Unknown result type (might be due to invalid IL or missing references)
		//IL_040e: Expected I4, but got Unknown
		//IL_033a: Unknown result type (might be due to invalid IL or missing references)
		//IL_034d: Unknown result type (might be due to invalid IL or missing references)
		//IL_02be: Unknown result type (might be due to invalid IL or missing references)
		//IL_02c5: Expected I4, but got Unknown
		//IL_098d: Unknown result type (might be due to invalid IL or missing references)
		//IL_0992: Unknown result type (might be due to invalid IL or missing references)
		//IL_0627: Unknown result type (might be due to invalid IL or missing references)
		//IL_062e: Invalid comparison between Unknown and I4
		//IL_0640: Unknown result type (might be due to invalid IL or missing references)
		//IL_0645: Unknown result type (might be due to invalid IL or missing references)
		try
		{
			Player i = Player.i;
			if ((Object)(object)i == (Object)null)
			{
				return null;
			}
			PlayerHealth componentInChildren = ((Component)i).GetComponentInChildren<PlayerHealth>(true);
			if ((Object)(object)componentInChildren == (Object)null)
			{
				return null;
			}
			float x = ((Component)i).transform.position.x;
			float y = ((Component)i).transform.position.y;
			float velX = ((PhysicsMover)i).VelX;
			float velY = ((PhysicsMover)i).VelY;
			float num = (((int)((Actor)i).Facing == 1) ? 1f : (-1f));
			bool flag = false;
			try
			{
				flag = ((Actor)i).IsOnGround;
			}
			catch
			{
			}
			float currentHealthValue = componentInChildren.CurrentHealthValue;
			float num2 = 0f;
			try
			{
				num2 = ((Health)componentInChildren).percentage;
			}
			catch
			{
			}
			float num3 = 0f;
			try
			{
				num3 = componentInChildren.CurrentInternalInjury;
			}
			catch
			{
			}
			float num4 = 0f;
			float num5 = 1f;
			float num6 = 0f;
			try
			{
				PlayerEnergy ammo = i.ammo;
				if ((Object)(object)ammo != (Object)null)
				{
					num4 = ammo.Value;
					num5 = ammo.MaxValue;
					num6 = ammo.percentage;
				}
			}
			catch
			{
			}
			float num7 = 0f;
			float num8 = 1f;
			float num9 = 0f;
			try
			{
				PlayerEnergy chiContainer = i.chiContainer;
				if ((Object)(object)chiContainer != (Object)null)
				{
					num7 = chiContainer.Value;
					num8 = chiContainer.MaxValue;
					num9 = chiContainer.percentage;
				}
			}
			catch
			{
			}
			int num10 = 0;
			int num11 = 1;
			float num12 = 0f;
			try
			{
				PotionContainer potion = i.potion;
				if ((Object)(object)potion != (Object)null)
				{
					num10 = potion.DrinkTimesLeft;
					num11 = potion.maxValue;
					if (num11 > 0)
					{
						num12 = (float)num10 / (float)num11;
					}
				}
			}
			catch
			{
			}
			float num13 = 0f;
			float num14 = 0f;
			float num15 = 0f;
			float num16 = 0f;
			float num17 = 0f;
			float num18 = 0f;
			float num19 = 0f;
			float num20 = 0f;
			int num21 = 0;
			int num22 = 0;
			int value = 1;
			int value2 = 1;
			int num23 = -1;
			bool flag2 = false;
			bool flag3 = false;
			bool flag4 = false;
			float num24 = 0f;
			int num25 = 0;
			float num26 = 0f;
			float num27 = 0f;
			int num28 = 0;
			int num29 = 0;
			float num30 = 0f;
			float num31 = 0f;
			int num32 = 0;
			int num33 = 0;
			MonsterBase val = null;
			string text = "none";
			try
			{
				UIMonsterHPManager val2 = (((Object)(object)SingletonBehaviour<GameCore>.Instance != (Object)null) ? SingletonBehaviour<GameCore>.Instance.monsterHpUI : null);
				UIBossHP val3 = (((Object)(object)val2 != (Object)null) ? val2.CurrentBossHP : null);
				if ((Object)(object)val3 != (Object)null && (Object)(object)val3.bindingPosture != (Object)null)
				{
					val = val3.bindingPosture.BindMonster;
					if ((Object)(object)val != (Object)null)
					{
						text = "ui";
					}
				}
			}
			catch
			{
			}
			if ((Object)(object)val == (Object)null)
			{
				int num34 = int.MinValue;
				MonsterBase[] array = Object.FindObjectsOfType<MonsterBase>();
				foreach (MonsterBase val4 in array)
				{
					if (!((Object)(object)val4 == (Object)null))
					{
						int num35 = 0;
						try
						{
							num35 = (int)val4.monsterStat.monsterLevel;
						}
						catch
						{
						}
						if (num35 > num34)
						{
							num34 = num35;
							val = val4;
						}
					}
				}
				if ((Object)(object)val != (Object)null)
				{
					text = "scan";
				}
			}
			_lastBossInfo = (((Object)(object)val != (Object)null) ? (((Object)val).name + "(" + text + ")") : "none");
			if ((Object)(object)val != (Object)null)
			{
				num13 = ((Component)val).transform.position.x;
				num14 = ((Component)val).transform.position.y;
				try
				{
					num15 = ((PhysicsMover)val).VelX;
					num16 = ((PhysicsMover)val).VelY;
				}
				catch
				{
				}
				try
				{
					num17 = (((int)((Actor)val).Facing == 1) ? 1f : (-1f));
				}
				catch
				{
				}
				try
				{
					PostureSystem postureSystem = val.postureSystem;
					float maxPostureValue = postureSystem.MaxPostureValue;
					num20 = maxPostureValue;
					float num36 = _bossHpScale * maxPostureValue;
					if (_bossHpScale < 1f && postureSystem.PostureValue > num36)
					{
						postureSystem.SetPostureValue((int)num36);
					}
					float remainTotal = postureSystem.RemainTotal;
					num18 = remainTotal;
					num19 = ((num36 > 0f) ? Mathf.Clamp01(remainTotal / num36) : 0f);
				}
				catch
				{
				}
				try
				{
					var currentState = val.CurrentState;
					num21 = (int)currentState;
					string text2 = currentState.ToString();
					string name = ((Object)val).name;
					if (_bossFsmCategories.TryGetValue(name, out var value3) && value3.TryGetValue(text2, out var value4))
					{
						num22 = value4;
					}
					if (!_bossTotalPhases.TryGetValue(name, out value) || value <= 0)
					{
						value = 1;
						if (_warnedMissingTotalPhaseBoss.Add(name))
						{
							Logger.LogWarning((object)("[phase] missing total_phases mapping for boss=" + name + ", fallback total_phases=1"));
						}
					}
					if (_bossAttackIds.TryGetValue(name, out var value5) && value5.TryGetValue(text2, out var value6))
					{
						num23 = value6;
					}
					if (_bossStatePhaseIndex.TryGetValue(name, out var value7))
					{
						if (!value7.TryGetValue(text2, out value2))
						{
							value2 = 1;
							string item = name + "::" + text2;
							if (_warnedMissingPhaseState.Add(item))
							{
								Logger.LogWarning((object)("[phase] missing phase state mapping boss=" + name + " state=" + text2 + ", fallback phase_index=1"));
							}
						}
					}
					else
					{
						value2 = 1;
						if (_warnedMissingPhaseMapBoss.Add(name))
						{
							Logger.LogWarning((object)("[phase] missing phase map for boss=" + name + ", fallback phase_index=1"));
						}
					}
					if (value2 < 1)
					{
						value2 = 1;
					}
					if (value2 > value)
					{
						value2 = value;
					}
					bool hasShieldObj = false;
					bool shieldActive = false;
					float shieldCur = 0f;
					float shieldMax = 0f;
					bool flag5 = TryReadShieldInfo(val, out hasShieldObj, out shieldActive, out shieldCur, out shieldMax);
					flag4 = hasShieldObj && shieldActive;
					num24 = ((hasShieldObj && shieldMax > 0f) ? Mathf.Clamp01(shieldCur / shieldMax) : 0f);
					bool flag6 = false;
					int num37 = 0;
					int num38 = 0;
					List<string> list = new List<string>();
					num26 = 0f;
					num27 = 0f;
					num28 = 0;
					num29 = 0;
					num30 = 0f;
					num31 = 0f;
					num32 = 0;
					num33 = 0;
					float num39 = float.MaxValue;
					float num40 = float.MaxValue;
					try
					{
						DamageDealer[] array2 = Object.FindObjectsOfType<DamageDealer>();
						foreach (DamageDealer val5 in array2)
						{
							if ((Object)(object)val5 == (Object)null || !((Behaviour)val5).isActiveAndEnabled || (int)val5.type != 16)
							{
								continue;
							}
							num37++;
							Vector3 position = ((Component)val5).transform.position;
							float num41 = position.x - x;
							float num42 = position.y - y;
							float num43 = num41 * num41 + num42 * num42;
							if (!val5.parriable)
							{
								num38++;
								string name2 = ((Object)((Component)val5).gameObject).name;
								list.Add(name2);
								flag6 = true;
								if (name2.IndexOf("Explos", StringComparison.OrdinalIgnoreCase) >= 0)
								{
									num28 = 1;
								}
								else if (name2.IndexOf("DamageArea", StringComparison.OrdinalIgnoreCase) >= 0)
								{
									num29 = 1;
								}
								else if (name2.IndexOf("Danger", StringComparison.OrdinalIgnoreCase) >= 0)
								{
									num33 = 1;
								}
								if (num43 < num39)
								{
									num39 = num43;
									num26 = num41;
									num27 = num42;
								}
							}
							else if (num43 < num40)
							{
								num40 = num43;
								num30 = num41;
								num31 = num42;
							}
						}
					}
					catch
					{
					}
					num32 = num38;
					num25 = ((num38 > 0) ? 1 : 0);
					if (flag6)
					{
						num22 = 2;
					}
					string text3 = $"{name}|fsm={text2}({num21})|cat={num22}|aid={num23}|" + $"shield=ok={flag5},has={hasShieldObj},act={shieldActive}," + $"hp={(shieldActive ? Mathf.Round(shieldCur) : 0f):F0}/{(shieldActive ? Mathf.Round(shieldMax) : 0f):F0}";
					if (text3 != _lastLoggedBossSig)
					{
						int num44 = 0;
						int num45 = 0;
						try
						{
							DamageDealer[] componentsInChildren = ((Component)val).GetComponentsInChildren<DamageDealer>(true);
							num44 = componentsInChildren.Length;
							DamageDealer[] array2 = componentsInChildren;
							foreach (DamageDealer val6 in array2)
							{
								if ((Object)(object)val6 != (Object)null && ((Behaviour)val6).isActiveAndEnabled)
								{
									num45++;
								}
							}
						}
						catch
						{
						}
						Logger.LogInfo((object)($"[boss] name={name} fsm={text2}(int={num21}) cat={num22} aid={num23} phase={value2}/{value} " + $"| shield=act={shieldActive} hp={shieldCur:F0}/{shieldMax:F0} (ok={flag5},has={hasShieldObj}) " + $"| DD bossAll={num44} bossAct={num45} sceneMonAct={num37} sceneMonUnp={num38} " + "unp=[" + string.Join(",", list) + "]"));
						_lastLoggedBossSig = text3;
					}
				}
				catch
				{
				}
				try
				{
					flag3 = val.IsDead();
				}
				catch
				{
				}
				flag2 = true;
			}
			float num46 = (flag2 ? (num13 - x) : 0f);
			float num47 = (flag2 ? (num14 - y) : 0f);
			bool flag7 = currentHealthValue <= 0f || flag3;
			bool flag8 = IsPlayerControllable(i) && Time.time - _lastResetTime > 3f;
			bool flag9 = IsHurtOrDown(i);
			PlayerStateType currentStateType = i.CurrentStateType;
			string text4 = currentStateType.ToString();
			if (_debugTick % 600 == 0)
			{
				string arg = (((Object)(object)val != (Object)null) ? ((Object)val).name : "(none)");
				Logger.LogInfo((object)($"[state] tick={_debugTick}\n" + $"  player    : hp={currentHealthValue:F0}({num2 * 100f:F1}%) injury={num3:F0} grounded={flag} facing={num:F0}\n" + $"  resources : ammo(蒼砂/射箭)={num4:F0}/{num5:F0}({num6 * 100f:F0}%) " + $"chi(氣)={num7:F0}/{num8:F0}({num9 * 100f:F0}%) " + $"potion(藥水)={num10}/{num11}({num12 * 100f:F0}%)\n" + $"  boss      : present={flag2} name={arg} " + $"hp={num18:F0}/{num20:F0}({num19 * 100f:F1}%) cat={num22} aid={num23} phase={value2}/{value} " + $"shield=act={flag4} hp%={num24:F2}\n" + $"  scene     : hazard_unparryable={num25} count={num32} " + $"type=[exp={num28} dmgArea={num29} danger={num33}] " + $"haz_nearest=({num26:F0},{num27:F0}) " + $"atk_nearest=({num30:F0},{num31:F0})\n" + $"  events    : last_parry={s_lastParryResult} parry_count={s_parryCount} " + $"controllable={flag8} knocked_down={flag9} boss_dead={flag3} done={flag7}"));
			}
			return "{" + $"\"px\":{x},\"py\":{y},\"vx\":{velX},\"vy\":{velY}," + string.Format("\"facing\":{0},\"grounded\":{1},", num, flag ? "true" : "false") + $"\"php\":{currentHealthValue},\"php_pct\":{num2},\"internal_injury\":{num3}," + $"\"qi\":{num4},\"qi_max\":{num5},\"qi_pct\":{num6}," + $"\"chi\":{num7},\"chi_max\":{num8},\"chi_pct\":{num9}," + $"\"potion_left\":{num10},\"potion_max\":{num11},\"potion_pct\":{num12}," + $"\"last_parry_result\":{s_lastParryResult},\"parry_count\":{s_parryCount},\"parry_enter_count\":{s_parryEnterCount}," + "\"boss_present\":" + (flag2 ? "true" : "false") + ",\"boss_name\":\"" + (((Object)(object)val != (Object)null) ? ((Object)val).name : "") + "\"," + $"\"bx\":{num13},\"by\":{num14},\"bvx\":{num15},\"bvy\":{num16}," + $"\"b_facing\":{num17},\"bdx\":{num46},\"bdy\":{num47}," + $"\"bhp\":{num18},\"bhp_pct\":{num19},\"bhp_max\":{num20}," + $"\"boss_fsm\":{num21}," + $"\"attack_category\":{num22}," + $"\"attack_id\":{num23}," + $"\"phase_index\":{value2}," + $"\"total_phases\":{value}," + "\"controllable\":" + (flag8 ? "true" : "false") + ",\"knocked_down\":" + (flag9 ? "true" : "false") + ",\"shield_active\":" + (flag4 ? "true" : "false") + "," + $"\"shield_hp_pct\":{num24}," + $"\"hazard_unparryable\":{num25}," + $"\"hazard_nearest_dx\":{num26},\"hazard_nearest_dy\":{num27}," + $"\"hazard_type_explosion\":{num28},\"hazard_type_damagearea\":{num29}," + $"\"attack_nearest_dx\":{num30},\"attack_nearest_dy\":{num31}," + $"\"hazard_count\":{num32},\"hazard_type_danger\":{num33}," + $"\"dt\":{dt}," + "\"state\":\"" + text4 + "\",\"boss_dead\":" + (flag3 ? "true" : "false") + ",\"done\":" + (flag7 ? "true" : "false") + "}\n";
		}
		catch (Exception ex)
		{
			Logger.LogError((object)ex);
			return null;
		}
	}

	private void SendState(string json)
	{
		if (_stream == null)
		{
			return;
		}
		try
		{
			byte[] bytes = Encoding.UTF8.GetBytes(json);
			_stream.Write(bytes, 0, bytes.Length);
		}
		catch
		{
			Disconnect();
		}
	}

	private void ParseAction(string json)
	{
		try
		{
			int num = json.LastIndexOf('{');
			if (num > 0)
			{
				json = json.Substring(num);
			}
			if (json.Replace(" ", "").Contains("\"reset\":true"))
			{
				DoHardReset();
				return;
			}
			int num2 = ExtractInt(json, "move");
			_moveDir = ((num2 == 1) ? (-1) : ((num2 == 2) ? 1 : 0));
			s_moveDir = _moveDir;   // 2026-06-15 Round 21 fix: sync to static mirror for postfix
			int dodge = ExtractInt(json, "dodge");
			int attack = ExtractInt(json, "attack");
			// 2026-06-15 Round 20: edge detection. WasPressed 只在「從非按到按」那 1 frame true;
			// IsPressed/pulse 維持按住期間 true。避免 hold pattern 被 game 誤認連續按多次。
			switch (dodge)
			{
			case 1:
				_jumpPulse = PULSE;
				s_jumpPulse = _jumpPulse;
				if (_prevDodge != 1)
				{
					_jumpEdge = 1;
					s_jumpEdge = 1;
				}
				break;
			case 2:
				_dashPulse = PULSE;
				s_dashPulse = _dashPulse;
				if (_prevDodge != 2)
				{
					_dashEdge = 1;
					s_dashEdge = 1;
				}
				break;
			}
			_prevDodge = dodge;
			// 2026-06-15 Round 21: 蓄力擊 lock 進行中 → 整個 attack switch 跳過。
			// 避免 case 1(_meleePulse=10)、case 2/3/4/5(自己的 edge)中斷 charge。
			// _prevAttack 也不更新,lock 結束後 edge detection 仍正常運作。
			if (_chargeLockFrames > 0)
			{
				_bossHpScale = ExtractFloat(json, "boss_hp_scale", _bossHpScale);
				return;
			}
			switch (attack)
			{
			case 1:
				// 雙重保護:Math.Max 不覆寫較大 _meleePulse(防 lock 邊界 race)
				if (_meleePulse < PULSE)
				{
					_meleePulse = PULSE;
					s_meleePulse = _meleePulse;
				}
				if (_prevAttack != 1)
				{
					_meleeEdge = 1;
					s_meleeEdge = 1;
				}
				break;
			case 2:
				_rangedPulse = PULSE;
				s_rangedPulse = _rangedPulse;
				if (_prevAttack != 2)
				{
					_rangedEdge = 1;
					s_rangedEdge = 1;
				}
				break;
			case 3:
				_parryPulse = PULSE;
				s_parryPulse = _parryPulse;
				if (_prevAttack != 3)
				{
					_parryEdge = 1;
					s_parryEdge = 1;
				}
				break;
			case 4:
				_talismanPulse = PULSE;
				s_talismanPulse = _talismanPulse;
				if (_prevAttack != 4)
				{
					_talismanEdge = 1;
					s_talismanEdge = 1;
				}
				break;
			case 5:
				_healPulse = PULSE;
				s_healPulse = _healPulse;
				if (_prevAttack != 5)
				{
					_healEdge = 1;
					s_healEdge = 1;
				}
				break;
			case 6:
				// 2026-06-15 Round 23 v4: case 6 = case 1 alias(defensive fallback)。
				// env.py 端把 attack=6 macro 拆 attack=1 × 40 + attack=0 × 50 給 Plugin,
				// Plugin 走已驗證 work 的 case 1 path。case 6 留著只是萬一 raw socket 直接送
				// attack=6(例如老 test script)時行為合理。
				if (_meleePulse < PULSE)
				{
					_meleePulse = PULSE;
					s_meleePulse = _meleePulse;
				}
				if (_prevAttack != 6)
				{
					_meleeEdge = 1;
					s_meleeEdge = 1;
				}
				break;
			}
			_prevAttack = attack;
			_bossHpScale = ExtractFloat(json, "boss_hp_scale", _bossHpScale);
		}
		catch
		{
		}
	}

	private void DoHardReset()
	{
		if (Time.time - _lastResetTime < 3f)
		{
			return;
		}
		_lastResetTime = Time.time;
		_moveDir = 0;
		s_moveDir = 0;   // 2026-06-15 Round 21 fix: keep static mirror in sync
		_chargeLockFrames = 0;   // 2026-06-15 Round 21: 釋放蓄力 lock
		_jumpPulse = (_dashPulse = (_meleePulse = (_rangedPulse = (_parryPulse = (_talismanPulse = (_healPulse = 0))))));
		try
		{
			Player.i.Suicide();
		}
		catch (Exception ex)
		{
			Logger.LogError((object)("[reset] " + ex));
		}
	}

	private static int ExtractInt(string json, string key)
	{
		string text = "\"" + key + "\":";
		int num = json.IndexOf(text);
		if (num < 0)
		{
			return 0;
		}
		int i;
		for (i = num + text.Length; i < json.Length && json[i] == ' '; i++)
		{
		}
		int j;
		for (j = i; j < json.Length && (char.IsDigit(json[j]) || json[j] == '-'); j++)
		{
		}
		if (j == i)
		{
			return 0;
		}
		return int.Parse(json.Substring(i, j - i));
	}

	private static float ExtractFloat(string json, string key, float fallback)
	{
		string text = "\"" + key + "\":";
		int num = json.IndexOf(text);
		if (num < 0)
		{
			return fallback;
		}
		int i;
		for (i = num + text.Length; i < json.Length && json[i] == ' '; i++)
		{
		}
		int j;
		for (j = i; j < json.Length && (char.IsDigit(json[j]) || json[j] == '-' || json[j] == '+' || json[j] == '.' || json[j] == 'e' || json[j] == 'E'); j++)
		{
		}
		if (j == i)
		{
			return fallback;
		}
		if (!float.TryParse(json.Substring(i, j - i), NumberStyles.Float, CultureInfo.InvariantCulture, out var result))
		{
			return fallback;
		}
		return result;
	}

	private void OnDestroy()
	{
		Disconnect();
	}
}
