# NineSols RL — 用強化學習打《九日 / Nine Sols》Boss

此專案讓 Agent 透過 **強化學習 (PPO + Self-Imitation Learning)** 學會擊敗《Nine Sols》的 Boss。

## 0. 架構
分成兩半,中間用 **TCP** 橋接:

```
                                    Transmitting using TCP
┌─────────────────────────┐            127.0.0.1:19271           ┌──────────────────────────┐
│  Nine Sols (Game)       │  ──── state  (Player/Boss)      ───> │  Python                  │
│  + BepInEx              │                                      │ (Gymnasium env + PPO/SIL)│
│  with (NineSolsRL.dll)  │  <─── action (movement/control) ───  │                          │
└─────────────────────────┘                                      └──────────────────────────┘
       TCP client                                                       TCP server
```

- **遊戲端**：一個 C# 的 BepInEx mod([Plugin.cs](NineSolsRL/Plugin.cs)),負責讀取遊戲內部狀態(玩家/Boss 位置、血量、Boss 攻擊類別…)、執行 Python 送來的操作,並當作 TCP **client** 連到 Python。
- **Python 端**：把遊戲包成標準 [Gymnasium](https://gymnasium.farama.org/) 環境 [`NineSolsEnv`](NineSolsRL/ninesols/env.py),用 [stable-baselines3](https://stable-baselines3.readthedocs.io/) 的 PPO 加上自訂的 Self-Imitation Learning 進行訓練。

---

## 目錄

1. [需求總覽](#1-需求總覽)
2. [遊戲端:安裝 BepInEx 與 mod](#2-遊戲端安裝-bepinex-與-mod)
3. [Python 端:安裝依賴](#3-python-端安裝依賴)
4. [先做連線測試](#4-先做連線測試-強烈建議)
5. [開始訓練](#5-開始訓練)
6. [訓練產物與監看](#6-訓練產物與監看)
7. [動作 / 觀測空間](#7-動作--觀測空間)
8. [常見問題](#8-常見問題-troubleshooting)
9. [專案結構](#9-專案結構)

---

## 1. 需求總覽

| 類別 | 需求 |
|------|------|
| 作業系統 | Windows 10 64bit |
| 記憶體 | 8G minimum |
| 遊戲 | [Nine Sols](https://store.steampowered.com/app/1809540/Nine_Sols/) |
| 遊戲 mod 框架 | [BepInEx](https://github.com/BepInEx/BepInEx)(5.4.23.5) |
| 編譯 mod | [.NET SDK](https://dotnet.microsoft.com/download)(`dotnet build`,9.0 recommanded) |
| Python | **3.12**(見 [.python-version](.python-version);最低 3.10) |
| 套件管理 | Using [uv](https://docs.astral.sh/uv/)(專案用 `uv.lock` 鎖定版本) |
| GPU(Optional) | NVIDIA GPU + CUDA 12.8 driver(torch version `cu128`) |

> 沒有 GPU 也能跑，只是慢很多。

---

## 2. 遊戲端:安裝 BepInEx 與 mod

### 2.1 安裝 BepInEx

1. 找到 Nine Sols 的安裝資料夾,例如:
   `E:\SteamLibrary\steamapps\common\Nine Sols\`
   (Steam → 右鍵 Nine Sols → 管理 → 瀏覽本機檔案)
2. 下載對應版本的 BepInEx,解壓到上面的遊戲根目錄。
3. **先啟動一次遊戲再關掉**,讓 BepInEx 產生 `BepInEx\plugins\`、`BepInEx\config\` 等資料夾。

### 2.2 編譯 mod

mod 的專案檔是 [NineSolsRL.csproj](NineSolsRL/NineSolsRL.csproj),它會參照遊戲安裝目錄裡的 DLL。
**路徑會自動偵測**(從登錄檔的 Steam 安裝位置 + 常見 Steam library 探測),多數情況直接編譯即可:

```powershell
cd NineSolsRL
dotnet build -c Release
```

> ⚠️ **若編譯報「找不到 Nine Sols 安裝路徑」**:代表自動偵測失敗(例如遊戲裝在較少見的磁碟)。
> 用環境變數或參數明確指定即可,**不必改 csproj**:
> ```powershell
> # 方式一:環境變數(這個 shell 內持續有效)
> $env:NINESOLS_DIR = "X:\path\to\Nine Sols"
> dotnet build -c Release
>
> # 方式二:單次參數
> dotnet build -c Release -p:NineSolsDir="X:\path\to\Nine Sols"
> ```
> (`X:\path\to\Nine Sols` 換成你遊戲根目錄,即含 `NineSols_Data\` 與 `BepInEx\` 的那層。)

編譯成功後會在 `NineSolsRL\bin\Release\netstandard2.1\` 下產生 `NineSolsRL.dll`。

### 2.3 安裝 mod

把編好的 `NineSolsRL.dll` 複製到遊戲的 plugins 資料夾:

```
E:\SteamLibrary\steamapps\common\Nine Sols\BepInEx\plugins\NineSolsRL.dll
```

之後遊戲一啟動,mod 就會自動嘗試連線到 `127.0.0.1:19271`(也就是 Python 端)。
**所以正確順序是:先開 Python 訓練程式(server),再開遊戲(client)。**

---

## 3. Python 端:安裝依賴

本專案用 [uv](https://docs.astral.sh/uv/) 管理 Python 環境與套件。

### 3.1 安裝 uv(若還沒有)
可以用以下方法安裝
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
你也可以從 [Pypi](https://pypi.org/project/uv/) 下載
```
# With pip.
pip install uv
```

### 3.2 安裝專案依賴

Python 專案與訓練腳本都在 `NineSolsRL\` 子資料夾,`pyproject.toml` / `uv.lock` 也在那裡:

```powershell
cd NineSolsRL
uv sync
```

`uv sync` 會依 [NineSolsRL/pyproject.toml](NineSolsRL/pyproject.toml) 與 `uv.lock`:

- 建立虛擬環境 `.venv`
- 安裝 `gymnasium` / `numpy` / `stable-baselines3`
- 從 PyTorch 官方 **CUDA 12.8** index(`cu128`)安裝 `torch`(GPU 版)

之後所有 Python 指令都用 `uv run` 執行,會自動套用這個環境,例如 `uv run python train.py`。

---

## 4. 先做連線測試(強烈建議)

正式訓練前,先確認「遊戲 ↔ Python」橋接是通的。

1. **先啟動 Nine Sols**,進入遊戲、把角色帶到一般可操作的場景，推薦回憶大廳。 

2. **開 Python 測試腳本**(它會 listen 在 19271):
   ```powershell
   cd NineSolsRL
   uv run python test_connection.py
   ```
   看到 `等待遊戲連線到 127.0.0.1:19271 ...` 表示 server 已就緒。
3. 連線成功後,測試腳本會印出 `遊戲已連線!`,並開始週期性地對角色送出各種動作,終端機也會持續印出玩家/Boss 狀態。

看到角色真的會自己動、終端機有 state 輸出,就代表 mod 與橋接都正常,可以進入訓練。

> 還有 [test_reset.py](NineSolsRL/test_reset.py)(測 `reset` 流程)與 `test_*.py` 一系列動作測試腳本,需要時可單獨跑來驗證個別功能。

---

## 5. 開始訓練

訓練入口是 [NineSolsRL/train.py](NineSolsRL/train.py)。

### 步驟

1. **先啟動訓練程式**(它會開 TCP server 等遊戲連入):
   ```powershell
   cd NineSolsRL
   uv run python train.py
   ```
   啟動時會印出使用 GPU 還是 CPU,以及是否有 checkpoint 可續訓。
2. **啟動 Nine Sols,把角色帶到要訓練的 Boss 場**。
3. mod 連上後,訓練就會自動開始:
   - 每個 episode 開始時,Python 會要求 mod 重置這場 Boss 戰(透過 `Player.Suicide()` 觸發 memory mode 重啟)。
   - agent 持續送動作、收狀態、累積獎勵,直到死亡或達到該難度的步數上限。

### 自動續訓

[train.py](NineSolsRL/train.py) 啟動時會掃描 `checkpoints\`,**自動載入步數最大的那組** checkpoint(模型 `.zip` + `vecnormalize` 統計 `.pkl` + 對應的 `sil_buffer.pkl`)接著訓練,不需要手動指定。想從零開始,把 `checkpoints\` 清空(或改名)即可。

### 可調參數

直接改 [train.py](NineSolsRL/train.py) 開頭的常數:

| 參數 | 說明 | 預設 |
|------|------|------|
| `TOTAL_TIMESTEPS` | 訓練目標總步數(已含先前續訓的步數) | `5_000_000` |
| `GAMMA` | 折扣因子 | `0.98` |
| `CKPT_DIR` | checkpoint 存放資料夾 | `checkpoints` |
| `SIL_CFG` | Self-Imitation Learning 設定 | `DEFAULT_SIL_CFG` |

環境本身的難度與獎勵設計則在 [ninesols/env.py](NineSolsRL/ninesols/env.py)(curriculum、里程碑獎勵)與 [ninesols/rewards.py](NineSolsRL/ninesols/rewards.py)。
其中有一套 **curriculum**:Boss 有效血量從滿血的 15% 起步,最近 20 場勝率達 60% 就自動 +15% 難度,直到 100%。

---

## 6. 訓練產物與監看

| 路徑 | 內容 |
|------|------|
| `checkpoints/ppo_ninesols_<步數>_steps.zip` | 每 40960 步存一次的模型 |
| `checkpoints/ppo_ninesols_vecnormalize_<步數>_steps.pkl` | 對應的觀測/獎勵正規化統計 |
| `checkpoints/sil_buffer_*.pkl` | 對應的 SIL 經驗 buffer |
| `logs/run_<時間戳>/progress.csv` | 每個 PPO iteration 一列(ep_rew_mean、loss、entropy、sil_* 等) |
| `logs/run_<時間戳>/episodes.csv` | 每個 episode 一列(步數、存活、獎勵、parry 次數、各動作使用量…) |

每次啟動訓練都會開一個新的 `logs/run_<時間戳>/` 資料夾,續訓不會蓋掉上一輪的 CSV。
想看學習曲線,直接用 Excel / pandas / matplotlib 開 `progress.csv` 或 `episodes.csv` 即可。

> 這些產物都很大,已在 [.gitignore](.gitignore) 內排除,不會進 git。

---

## 7. 動作 / 觀測空間

### 動作(Action)— `MultiDiscrete([3, 3, 7])`

| 維度 | 值 | 意義 |
|------|----|------|
| move | 0 / 1 / 2 | 停 / 左 / 右 |
| dodge | 0 / 1 / 2 | 無 / 跳 / 衝刺 |
| attack | 0 / 1 / 2 / 3 / 4 / 5 / 6 | 無 / 近戰 / 遠程 / 格檔(parry) / 貼道符 / 喝藥 / 蓄力攻擊 |

### 觀測(Observation)— 56 維 `Box`

包含玩家狀態(相對座標、速度、朝向、是否落地、血量、內傷)、Boss 狀態、Boss 攻擊類別(8 類 one-hot)、per-boss 攻擊 ID(20 維 one-hot),以及護盾、地形危險、最近攻擊等額外資訊。詳細定義見 [ninesols/env.py](NineSolsRL/ninesols/env.py) 的 `_encode_obs`。

---

## 8. 常見問題 (Troubleshooting)

**Q. 遊戲開了但 Python 一直停在「等待遊戲連線」?**
確認 mod DLL 真的在 `BepInEx\plugins\`;確認沒有別的程式佔用 19271 port。

**Q. `dotnet build` 報「找不到 Nine Sols 安裝路徑」/ 找不到 BepInEx、UnityEngine、Assembly-CSharp.dll?**
代表 [NineSolsRL.csproj](NineSolsRL/NineSolsRL.csproj) 的自動偵測沒命中你的安裝位置。設環境變數 `NINESOLS_DIR`,或 `dotnet build -c Release -p:NineSolsDir="X:\path\to\Nine Sols"`(指到含 `NineSols_Data\` 的遊戲根目錄),不用手改 csproj。

**Q. agent 連上了但角色不動 / 不會重置?**
mod 只有在角色「可操作 (controllable)」時才會介入。先用 [test_connection.py](NineSolsRL/test_connection.py) 確認動作有送進去;訓練時要先把角色帶進 Boss 場再讓它接管。

**Q. 訓練很慢?**
確認跑起來時印的是「使用 GPU」。若是「改用 CPU」,代表沒裝到 CUDA 版 torch 或沒抓到顯卡;重跑 `uv sync` 並確認 NVIDIA driver 支援 CUDA 12.8。

**Q. 想從頭重新訓練?**
清空 / 改名 `NineSolsRL\checkpoints\` 資料夾,再跑 `uv run python train.py`。

---

## 9. 專案結構

```
Ninesols_rl/
├─ README.md                  ← 你正在看的檔案
└─ NineSolsRL/
   ├─ Plugin.cs               遊戲端 BepInEx mod 原始碼 (C#)
   ├─ NineSolsRL.csproj       mod 編譯設定 (自動偵測遊戲路徑,可用 NINESOLS_DIR 覆寫)
   ├─ pyproject.toml          Python 依賴定義
   ├─ uv.lock                 鎖定的套件版本
   ├─ train.py                ★ 訓練入口 (PPO + SIL,含自動續訓)
   ├─ test_connection.py      連線/動作煙霧測試
   ├─ test_reset.py           reset 流程測試
   ├─ test_*.py               各種動作 (雙跳、蓄力擊…) 的單項測試
   ├─ ninesols/               Python 套件
   │  ├─ env.py               Gymnasium 環境 NineSolsEnv (obs/reward/curriculum)
   │  ├─ bridge.py            TCP server 橋接層 GameBridge
   │  ├─ rewards.py           獎勵函數
   │  ├─ ppo_sil.py           PPO + Self-Imitation Learning 演算法
   │  ├─ sil_buffer.py        SIL 經驗 buffer
   │  └─ sil_callbacks.py     checkpoint / recorder / CSV 紀錄 callbacks
   ├─ checkpoints/            訓練產出的模型 (git 忽略)
   └─ logs/                   每次訓練的 CSV 紀錄 (git 忽略)
```

---

### TL;DR 最短路徑

```powershell
# 1) 編譯並安裝 mod (改好 csproj 的 HintPath 後)
cd NineSolsRL
dotnet build -c Release
# 把 bin\Release\netstandard2.1\NineSolsRL.dll 複製到 遊戲\BepInEx\plugins\

# 2) 安裝 Python 依賴
uv sync

# 3) 先開訓練 (等遊戲連入)
uv run python train.py
# 4) 再開 Nine Sols,把角色帶到 Boss 場 → 自動開始訓練
```
