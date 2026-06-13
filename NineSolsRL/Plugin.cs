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

namespace NineSolsRL;

[BepInPlugin("com.ninesolsrl.plugin", "NineSolsRL", "2.0.0")]
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
			Plugin instance = Instance;
			if (instance != null && instance._connected && __instance != null && IsPlayerControllable(__instance))
			{
				int moveDir = instance._moveDir;
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
			Plugin instance = Instance;
			if (instance == null)
			{
				return;
			}
			try
			{
				instance._lastParryResult = ((!data.isAccurate) ? 1 : 2);
				instance._parryCount++;
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
			Plugin instance = Instance;
			if (instance == null)
			{
				return;
			}
			try
			{
				instance._parryEnterCount++;
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

	private const int PULSE = 3;

	private float _bossHpScale = 1f;

	private float _lastResetTime;

	private string _lastBossInfo = "none";

	private int _lastParryResult;

	private int _parryCount;

	public int _parryEnterCount;

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
		Log = ((BaseUnityPlugin)this).Logger;
		Object.DontDestroyOnLoad((Object)(object)((Component)this).gameObject);
		Harmony val = new Harmony("com.ninesolsrl.plugin");
		val.PatchAll();
		try
		{
			Type type = AccessTools.TypeByName("InControl.OneAxisInputControl");
			if (type != null)
			{
				MethodInfo methodInfo = AccessTools.PropertyGetter(type, "WasPressed");
				val.Patch((MethodBase)methodInfo, (HarmonyMethod)null, new HarmonyMethod(typeof(Plugin).GetMethod("WasPressedPostfix", BindingFlags.Static | BindingFlags.NonPublic)), (HarmonyMethod)null, (HarmonyMethod)null, (HarmonyMethod)null);
			}
			else
			{
				((BaseUnityPlugin)this).Logger.LogError((object)"找不到 InControl.OneAxisInputControl 型別，離散動作將無法注入");
			}
		}
		catch (Exception ex)
		{
			((BaseUnityPlugin)this).Logger.LogError((object)("InControl patch 失敗: " + ex));
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
		Plugin instance = Instance;
		if (instance == null || !instance._connected)
		{
			return;
		}
		PlayerGamePlayActionSet actions = instance.GetActions();
		if (actions == null)
		{
			return;
		}
		Player i = Player.i;
		if (IsPlayerControllable(i))
		{
			if (instance._jumpPulse > 0 && __instance == actions.Jump)
			{
				__result = true;
			}
			else if (instance._dashPulse > 0 && __instance == actions.Dodge)
			{
				__result = true;
			}
			else if (instance._meleePulse > 0 && __instance == actions.Attack)
			{
				__result = true;
			}
			else if (instance._rangedPulse > 0 && __instance == actions.WeaponAttack)
			{
				__result = true;
			}
			else if (instance._parryPulse > 0 && __instance == actions.Parry)
			{
				__result = true;
			}
			else if (instance._talismanPulse > 0 && __instance == actions.FooAttack)
			{
				__result = true;
			}
			else if (instance._healPulse > 0 && __instance == actions.Heal)
			{
				__result = true;
			}
		}
		else if (IsHurtOrDown(i) && instance._dashPulse > 0 && __instance == actions.Dodge)
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
		}
		catch
		{
		}
		return _cachedActions;
	}

	public void OnGameUpdate()
	{
		_debugTick++;
		if (_jumpPulse > 0)
		{
			_jumpPulse--;
		}
		if (_dashPulse > 0)
		{
			_dashPulse--;
		}
		if (_meleePulse > 0)
		{
			_meleePulse--;
		}
		if (_rangedPulse > 0)
		{
			_rangedPulse--;
		}
		if (_parryPulse > 0)
		{
			_parryPulse--;
		}
		if (_talismanPulse > 0)
		{
			_talismanPulse--;
		}
		if (_healPulse > 0)
		{
			_healPulse--;
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
					States currentState = val.CurrentState;
					num21 = (int)currentState;
					string text2 = ((object)(States)(ref currentState)).ToString();
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
							((BaseUnityPlugin)this).Logger.LogWarning((object)("[phase] missing total_phases mapping for boss=" + name + ", fallback total_phases=1"));
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
								((BaseUnityPlugin)this).Logger.LogWarning((object)("[phase] missing phase state mapping boss=" + name + " state=" + text2 + ", fallback phase_index=1"));
							}
						}
					}
					else
					{
						value2 = 1;
						if (_warnedMissingPhaseMapBoss.Add(name))
						{
							((BaseUnityPlugin)this).Logger.LogWarning((object)("[phase] missing phase map for boss=" + name + ", fallback phase_index=1"));
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
						((BaseUnityPlugin)this).Logger.LogInfo((object)($"[boss] name={name} fsm={text2}(int={num21}) cat={num22} aid={num23} phase={value2}/{value} " + $"| shield=act={shieldActive} hp={shieldCur:F0}/{shieldMax:F0} (ok={flag5},has={hasShieldObj}) " + $"| DD bossAll={num44} bossAct={num45} sceneMonAct={num37} sceneMonUnp={num38} " + "unp=[" + string.Join(",", list) + "]"));
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
			string text4 = ((object)(PlayerStateType)(ref currentStateType)).ToString();
			if (_debugTick % 600 == 0)
			{
				string arg = (((Object)(object)val != (Object)null) ? ((Object)val).name : "(none)");
				((BaseUnityPlugin)this).Logger.LogInfo((object)($"[state] tick={_debugTick}\n" + $"  player    : hp={currentHealthValue:F0}({num2 * 100f:F1}%) injury={num3:F0} grounded={flag} facing={num:F0}\n" + $"  resources : ammo(蒼砂/射箭)={num4:F0}/{num5:F0}({num6 * 100f:F0}%) " + $"chi(氣)={num7:F0}/{num8:F0}({num9 * 100f:F0}%) " + $"potion(藥水)={num10}/{num11}({num12 * 100f:F0}%)\n" + $"  boss      : present={flag2} name={arg} " + $"hp={num18:F0}/{num20:F0}({num19 * 100f:F1}%) cat={num22} aid={num23} phase={value2}/{value} " + $"shield=act={flag4} hp%={num24:F2}\n" + $"  scene     : hazard_unparryable={num25} count={num32} " + $"type=[exp={num28} dmgArea={num29} danger={num33}] " + $"haz_nearest=({num26:F0},{num27:F0}) " + $"atk_nearest=({num30:F0},{num31:F0})\n" + $"  events    : last_parry={_lastParryResult} parry_count={_parryCount} " + $"controllable={flag8} knocked_down={flag9} boss_dead={flag3} done={flag7}"));
			}
			return "{" + $"\"px\":{x},\"py\":{y},\"vx\":{velX},\"vy\":{velY}," + string.Format("\"facing\":{0},\"grounded\":{1},", num, flag ? "true" : "false") + $"\"php\":{currentHealthValue},\"php_pct\":{num2},\"internal_injury\":{num3}," + $"\"qi\":{num4},\"qi_max\":{num5},\"qi_pct\":{num6}," + $"\"chi\":{num7},\"chi_max\":{num8},\"chi_pct\":{num9}," + $"\"potion_left\":{num10},\"potion_max\":{num11},\"potion_pct\":{num12}," + $"\"last_parry_result\":{_lastParryResult},\"parry_count\":{_parryCount},\"parry_enter_count\":{_parryEnterCount}," + "\"boss_present\":" + (flag2 ? "true" : "false") + ",\"boss_name\":\"" + (((Object)(object)val != (Object)null) ? ((Object)val).name : "") + "\"," + $"\"bx\":{num13},\"by\":{num14},\"bvx\":{num15},\"bvy\":{num16}," + $"\"b_facing\":{num17},\"bdx\":{num46},\"bdy\":{num47}," + $"\"bhp\":{num18},\"bhp_pct\":{num19},\"bhp_max\":{num20}," + $"\"boss_fsm\":{num21}," + $"\"attack_category\":{num22}," + $"\"attack_id\":{num23}," + $"\"phase_index\":{value2}," + $"\"total_phases\":{value}," + "\"controllable\":" + (flag8 ? "true" : "false") + ",\"knocked_down\":" + (flag9 ? "true" : "false") + ",\"shield_active\":" + (flag4 ? "true" : "false") + "," + $"\"shield_hp_pct\":{num24}," + $"\"hazard_unparryable\":{num25}," + $"\"hazard_nearest_dx\":{num26},\"hazard_nearest_dy\":{num27}," + $"\"hazard_type_explosion\":{num28},\"hazard_type_damagearea\":{num29}," + $"\"attack_nearest_dx\":{num30},\"attack_nearest_dy\":{num31}," + $"\"hazard_count\":{num32},\"hazard_type_danger\":{num33}," + $"\"dt\":{dt}," + "\"state\":\"" + text4 + "\",\"boss_dead\":" + (flag3 ? "true" : "false") + ",\"done\":" + (flag7 ? "true" : "false") + "}\n";
		}
		catch (Exception ex)
		{
			((BaseUnityPlugin)this).Logger.LogError((object)ex);
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
			switch (ExtractInt(json, "dodge"))
			{
			case 1:
			{
				bool flag = false;
				try
				{
					flag = (Object)(object)Player.i != (Object)null && ((Actor)Player.i).IsOnGround;
				}
				catch
				{
				}
				if (flag || _jumpPulse == 0)
				{
					_jumpPulse = 3;
				}
				break;
			}
			case 2:
				_dashPulse = 3;
				break;
			}
			switch (ExtractInt(json, "attack"))
			{
			case 1:
				_meleePulse = 3;
				break;
			case 2:
				_rangedPulse = 3;
				break;
			case 3:
				_parryPulse = 3;
				break;
			case 4:
				_talismanPulse = 3;
				break;
			case 5:
				_healPulse = 3;
				break;
			}
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
		_jumpPulse = (_dashPulse = (_meleePulse = (_rangedPulse = (_parryPulse = (_talismanPulse = (_healPulse = 0))))));
		try
		{
			Player.i.Suicide();
		}
		catch (Exception ex)
		{
			((BaseUnityPlugin)this).Logger.LogError((object)("[reset] " + ex));
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
