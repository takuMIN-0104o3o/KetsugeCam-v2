# ketsugecam 使い方

画面をキャプチャして、そのまま GPU 上の `torch.Tensor` として受け取るライブラリ。
YOLO (TensorRT) 検出、フレームの時刻・遅延の計測、遅延ぶんの位置補正もついている。

返ってくるフレームは常に `torch.Tensor`、形は `[高さ, 幅, 3]`、`uint8`、**BGR**、CUDA 上。

## 準備

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA版PyTorch
pip install comtypes
pip install tensorrt opencv-python numpy   # 検出・表示を使うときだけ
pip install pygame                          # 計測ツール・サンプル表示を使うときだけ
```

```python
import ketsugecam
```

---

## 1. 画面を1枚撮る

```python
cam = ketsugecam.create(region=(0, 0, 640, 640))   # (左, 上, 右, 下)
frame = cam.grab()

if frame is not None:        # 画面に変化が無いと None
    print(frame.shape)       # torch.Size([640, 640, 3])

cam.release()
```

`region` を省略すると画面全体。`with` を使えば `release()` は自動:

```python
with ketsugecam.create() as cam:
    frame = cam.grab()
    frame, info = cam.grab_info()   # 画像 + メタデータ (後述)
```

## 2. 連続で撮る

```python
cam = ketsugecam.create(region=(0, 0, 640, 640))
cam.start()                               # 新フレームが届いた瞬間に取る (既定)

try:
    while True:
        frame = cam.get_latest_frame()    # 新しいフレームが来るまで待つ (最大1秒)
        if frame is None:
            continue
        ...
finally:
    cam.stop()
    cam.release()
```

### キャプチャの方式

| 呼び方 | 動き |
|---|---|
| `cam.start()` (既定) | **到着待ち**。DXGI が新フレームを出した瞬間に取る。ポーリングの待ちがない |
| `cam.start(target_fps=240)` | タイマーでその周期ごとに取りに行く。GPU負荷を抑えたいとき用。最大 1/target_fps 遅れる |

実測 (240Hz モニター): present → 取得 が 到着待ち **0.035ms** / タイマー 240fps 1.86ms。

- 上限はモニターのリフレッシュレート。DXGI は合成済みの画面更新しか出さないので、
  240Hz モニターなら 240Hz 前後が最大
- `cam.capture_fps` … 実際のFPS
- `cam.mode` … `"cuda_interop"` (GPUだけで完結) か `"cpu_fallback"`

### 最新フレーム優先

利用側は常に「その時点の最新」を受け取る (キューに溜めて順番に処理しない)。
推論中に届いたフレームは飛ばされ、`get_stats().consumer_skipped` に数えられる。

### GPUバッファの寿命

interop モードの返り値は使い回しバッファのビュー。出力バッファは `num_buffers` 枚 (既定3) で、
最新と貸し出し中のバッファには書き込まない。

| 取り方 | 有効期間 |
|---|---|
| `get_latest_frame()` (copy=False) | 少なくとも `num_buffers-1` フレーム (既定 2 フレーム) |
| `get_latest_frame(copy=True)` | ずっと (専用のコピー) |
| `with cam.lease_latest_frame() as (frame, info):` | `with` を抜けるまで (コピーなし) |

空きバッファが無いときは新しいフレームのほうを捨てる (`busy_drops`)。
lease を2つ同時に持つ・`num_buffers=2` にする、などをしなければ起きない。

## 3. フレームの時刻・連番 (FrameInfo)

```python
frame, info = cam.get_latest_frame_info(copy=True)
if frame is not None and info.is_new:     # 前回と同じフレームなら is_new=False
    info.frame_id                 # 1 から始まる連番
    info.present_time_raw         # DXGI LastPresentTime (QPC の生値)
    info.capture_done_ns          # キャプチャ完了 (perf_counter_ns)
    info.present_to_capture_ns    # 画面更新 → キャプチャ完了 (ns)
    info.age_ns()                 # キャプチャ完了 → 今 (ns)
```

画像とメタデータは同じロックの下で取り出すので、別フレームの組み合わせにはならない。

| 属性 | 内容 |
|---|---|
| `frame_id` | 連番。重複処理・飛ばしの確認に使う |
| `present_time_raw` | DXGI LastPresentTime (QPC ティック、無変換) |
| `acquire_ns` / `acquire_qpc` | AcquireNextFrame が返った時刻 |
| `capture_done_ns` / `capture_done_qpc` | テンソルへのコピー完了時刻 |
| `accumulated_frames` | 前回取得からの Present 数 (DXGI) |
| `width` / `height` | フレームサイズ |
| `is_new` | `get_latest_frame_info()` が前回返したものと別のフレームか |
| `present_to_capture_ns` | Present → キャプチャ完了 |
| `present_to_acquire_ns` | Present → AcquireNextFrame |
| `copy_ns` | AcquireNextFrame → コピー完了 |
| `skipped_presents` | DXGI が合成して取り逃した Present 数 (ROI外も含む) |
| `clock_consistent` | present ≤ acquire ≤ capture_done が成り立つか |
| `age_ns(now_ns=None)` | キャプチャ完了からの経過 |

### 時計について

- `*_ns` は `time.perf_counter_ns()`、`*_qpc` と `present_time_raw` は QPC (`QueryPerformanceCounter`)
- **QPC と perf_counter_ns を直接引き算しない**。遅延は QPC 同士で引いてから `ketsugecam.qpc_to_ns()` で換算する
  (`present_to_capture_ns` などはそうしてある)
- `ketsugecam.qpc_now()` … 今の QPC、`ketsugecam.QPC_FREQUENCY` … QPC の周波数
- LastPresentTime が QPC と同じ時計であることは `python -m ketsugecam.latency` で検証済み
  (全フレームで draw ≤ LastPresentTime ≤ acquire ≤ capture_done)

## 4. 統計とCSVログ

```python
st = cam.get_stats()            # FrameStats (直近 stats_window フレームの集計)
print(st.capture_fps, st.present_to_capture_ms_mean, st.consumer_skipped)
cam.reset_stats()

cam.start_log("frames.csv")     # キャプチャ・受け取りごとに1行
...
cam.stop_log()
```

主な `FrameStats`:

| 項目 | 内容 |
|---|---|
| `captured` / `capture_fps` | キャプチャした新フレーム数 / FPS |
| `interval_ms_mean/min/max/std` | フレーム間隔 |
| `present_to_capture_ms_mean/max` | 画面更新 → キャプチャ完了 |
| `copy_ms_mean` | 取得 → コピー完了 |
| `age_ms_mean/max` | キャプチャ完了 → 利用側が受け取るまで |
| `dxgi_skipped` | DXGI が取り逃した Present |
| `busy_drops` | 空きバッファが無くて捨てたフレーム |
| `cursor_only` / `timeouts` / `access_lost` | カーソルだけの更新 / タイムアウト / 再接続 |
| `consumed` / `consumer_repeats` / `consumer_skipped` | 受け取り回数 / 同じフレームの再取得 / 受け取らずに上書きされた数 |
| `clock_violations` | 時刻の順序が崩れたフレーム数 (0 が正常) |

CSV はキューに積むだけで、書き込みは別スレッド。キャプチャを遅らせない。

## 5. 画面の真ん中を撮る

```python
w, h = ketsugecam.monitor_size()   # 例: (1920, 1080)。DPIスケーリングの影響なし
size = 416
left, top = (w - size) // 2, (h - size) // 2
cam = ketsugecam.create(region=(left, top, left + size, top + size))
```

## 6. YOLOで検出する

```python
cam   = ketsugecam.create(region=(752, 332, 1168, 748))
model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5)
cam.start()

while True:
    frame, info = cam.get_latest_frame_info(copy=True)
    if frame is None or not info.is_new:
        continue
    for d in model.predict(frame, info):      # info を渡すと処理時間とフレームが対応づく
        print(d.name, d.conf, d.xyxy, d.center)
```

前処理・NMS は中でやるので、`frame` をそのまま渡すだけ。numpy の BGR 画像を渡してもよい。

| 属性 | 内容 |
|---|---|
| `xyxy` | 枠 `(x1, y1, x2, y2)` (ROI内の座標) |
| `xywh` | 枠 `(x, y, 幅, 高さ)` |
| `center` | 枠の中心 `(x, y)` |
| `conf` | 信頼度 |
| `cls` / `name` | クラス番号 / クラス名 |

- `"best.engine"` は同梱モデル。自分のモデル (`.engine` / `.onnx`) のパスも渡せる
- クラス名はモデルから自動で読む (`model.names` で確認)
- `model.conf = 0.5` のように、しきい値は途中で変えられる

### YOLOの処理時間 (DetectTiming)

```python
dets = model.predict(frame, info)
tm = model.last_timing
if not tm.warmup:                               # 初回 (CUDAグラフ準備) は除く
    tm.gpu_infer_ms, tm.gpu_decode_ms           # GPU (CUDAイベント)
    tm.wait_ms, tm.post_ms, tm.total_ms         # CPU から見た実時間
    tm.present_to_result_ns(info)               # 画面更新 → 結果確定 (frame_id が違えば例外)
```

| 項目 | 内容 |
|---|---|
| `frame_id` | `predict(frame, info)` で渡したフレーム |
| `warmup` | この呼び出しで CUDA グラフをキャプチャした (初回) |
| `start_ns` / `start_qpc` | 推論開始 |
| `gpu_done_ns` | GPU の結果が CPU に届いた時刻 |
| `done_ns` / `done_qpc` | 後処理 (NMS) まで終わって結果が確定した時刻 |
| `gpu_infer_ms` | コピー + 前処理 + TensorRT 推論 (GPU上の区間) |
| `gpu_decode_ms` | デコード / しきい値 (途中のホスト同期で GPU が空く時間も含む) |
| `wait_ms` / `post_ms` / `total_ms` | GPU結果待ち / NMS / 合計 (実時間) |

- 検出 0 件でも毎回更新される
- `predict()` が例外で終わったら `last_timing` は `None` (前フレームの値は残らない)
- 1つの Detector を複数スレッドから呼ぶと `last_timing` は最後の呼び出しで上書きされる。
  スレッドごとに Detector を作る

### 結果を描いて表示する

```python
import cv2

dets = model.predict(frame)
img = frame.cpu().numpy()       # 表示用にCPUへ
model.draw(img, dets)           # 枠・信頼度・中心点を描き込む
cv2.imshow("ketsugecam", img)
cv2.waitKey(1)
```

表示用の `.cpu()` は検出スレッドでやらない。毎フレーム待つと検出が遅くなる
(実測で 240 → 135fps)。表示ループ側で表示する分だけ CPU へ落とす。

## 7. 遅延補正 (LatencyCompensator)

検出結果を「そのフレームの画面更新時刻 (present)」から「今」まで、対象の速度で進める。
キャプチャ・推論の待ち・推論・受け渡しの遅延は毎フレームの実測 (QPC) がそのまま入る。

```python
comp = ketsugecam.LatencyCompensator()

# 検出スレッド
comp.update(model.predict(frame, info), info)

# 使う側 (別スレッドでよい)。位置を使う直前に呼ぶ
for b in comp.boxes():
    b.xyxy, b.center            # 補正済み (呼んだ瞬間の画面上の位置)
    b.lead, b.lead_ms           # 補正量 (px) / 何 ms 進めたか
```

| 位置の取り方 | 中身 |
|---|---|
| `model.predict()` の結果 | 補正前 (そのフレームを撮った時点の位置) |
| `comp.boxes()` の結果 | **補正済み** (呼んだ瞬間の画面上の位置)。描画でもそれ以外でも同じ |

### 使い方 (手順)

**1. 作る** — カメラを先に作る (torch の CUDA 初期化より前に interop を用意するため)

```python
import threading, time
import ketsugecam

size = 416
w, h = ketsugecam.monitor_size()
left, top = (w - size) // 2, (h - size) // 2
cam = ketsugecam.create(region=(left, top, left + size, top + size))
cam.start()                                   # 到着待ち (遅延が一番小さい)

model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5)
comp  = ketsugecam.LatencyCompensator()       # 既定のままでよい
```

**2. 検出スレッドで `update()`** — 新しいフレームごとに、検出結果と `info` を渡す

```python
stop = threading.Event()

def detection_thread():
    while not stop.is_set():
        frame, info = cam.get_latest_frame_info(timeout=0.005, copy=True)
        if frame is None or not info.is_new:  # 同じフレームは渡さない
            continue
        comp.update(model.predict(frame, info), info)   # info が時刻の元になる

threading.Thread(target=detection_thread, daemon=True).start()
```

- `info` は必ずそのフレームのものを渡す (present の時刻で速度と補正量を計算する)
- `.cpu()` など重い処理はこのスレッドに入れない (検出の頻度が落ちると速度推定が粗くなる)
- クラスを絞るなら `update()` に渡す前に絞る

**3. 使う側で `boxes()`** — 位置を使う直前に呼ぶ (どのスレッドからでもよい)

```python
def to_screen(x, y):                          # ROI 内 → 画面全体の座標
    return left + x, top + y

while True:
    for b in comp.boxes():                    # 呼んだ瞬間の画面上の位置
        sx, sy = to_screen(*b.center)
        print(b.name, sx, sy, b.lead_ms)
    time.sleep(1 / 60)
```

```python
# NG: 取っておいて後で使う → 取ってからの時間ぶん古くなる
boxes = comp.boxes()
heavy_work()
use(boxes)

# OK: 使う直前に取る
heavy_work()
use(comp.boxes())
```

**4. 終わる**

```python
stop.set()
cam.release()
```

**5. 確かめる**

| 方法 | 見るもの |
|---|---|
| `comp.alignment()` を毎秒表示 | 補正ありのずれが補正なしより小さいか (動く対象があるとき)、`time_bias_ms` が 0 付近か |
| `main.py` で `show_raw_boxes = True` | 補正前の枠 (灰) と補正量の線。動く対象で枠が進行方向に出ていれば効いている |
| `python 新しいフォルダー/test_position.py` | 補正済みの位置・画面座標・補正量の関係を自動で確認 |

静止している対象では補正量は 0 (`min_speed` 未満) なので、効果は動く対象でしか見えない。

### 調整

| したいこと | 変えるもの |
|---|---|
| 表示など `boxes()` より後ろの遅延も含めたい | `extra_ms` を足す (例: 表示が 1 フレーム 16ms 遅れるなら `extra_ms=16`) |
| 枠が出るのを早くしたい | `min_hits=1` (誤検出も出やすくなる) |
| 止まっている対象の揺れを抑えたい | `min_speed` を上げる |
| 加減速への追従を早くしたい | `beta` を上げる (速度のノイズも増える) |
| 枠をなめらかにしたい | `alpha` を下げる (観測への追従は遅くなる) |
| 速い対象を取り違える | `gate_px` を上げる |
| 自動の時間ずれ補正を止めたい | `auto_bias=False` |

実行中に `comp.extra_ms = 5.0` のように書き換えてもよい。`comp.reset()` で追跡と自動補正をリセット。

- `boxes()` は呼んだ瞬間を「今」として計算する。取っておくとその分古くなる
- 座標は ROI 内。画面座標は ROI の左上を足す
- `CompensatedBox` は `Detection` と同じ `xyxy / conf / cls / name / center` を持つ

### 引数

| 引数 | 既定 | 内容 |
|---|---|---|
| `extra_ms` | 0.0 | `boxes()` より後ろの、ここで測れない遅延 (表示など) を足す分 |
| `min_speed` | 30.0 | これより遅い (px/s) 対象は止まっているとみなし、進めない |
| `auto_bias` | True | 答え合わせで測った予測器の時間ずれを自動で打ち消す |
| `bias_step_ms` | 0.02 | 自動補正の1件あたりの刻み (符号だけ見るので中央値に収束) |
| `alpha` / `beta` | 0.5 / 0.1 | 位置 / 速度の追従率 (Tracker に渡る) |
| `gate_px` | 60.0 | 同じ対象とみなす距離 |
| `min_hits` | 3 | この回数続けて検出するまで出さない (誤検出対策) |
| `max_misses` | 5 | この回数続けて見失ったら追跡をやめる |
| `max_lead_ms` | 50.0 | 予測する時間の上限 |

### 実際の画面との答え合わせ

`boxes()` で出した「時刻 t の推定位置」を、後から届いた t の前後のフレームの検出
(補間) と毎回比べる。

```python
al = comp.alignment()
al.comp_p50, al.comp_p95      # 補正ありのずれ (px)
al.raw_p50, al.raw_p95        # 補正なし (最後の検出位置) のずれ
al.time_bias_ms               # 進行方向のずれを時間にしたもの (+ 進みすぎ / - 遅れ)
comp.bias_ms                  # 自動補正で差し引いている時間
```

実測 (ボットが動く場面、10秒): 補正あり p50/p95 **1.24 / 7.32px**、補正なし 1.58 / 8.63px。

### 補正の範囲と限界

```
ゲーム内の状態 → ゲームが描画 → [present] → モニター送出 → 液晶が光る
                                   ↑ 起点
   キャプチャ → 推論 → 受け渡し → boxes() を呼んだ「今」   ← ここまでを補正
```

- present より前 (ゲーム内部) と、モニターが光るまでは画面に写らないので測れない。
  合わせたいときは `extra_ms` で足し引きする
- 等速モデル。急な停止・方向転換の直後は外れる (p95 が大きいのは主にこれ)
- 座標は画面上なので、視点の動きも対象の速度に入る
- p50 約1px は YOLO の検出そのものの揺れ (σ 約0.5〜0.8px) と同程度で、これ以下にはならない

### 速度推定だけ使う (Tracker)

```python
tracker = ketsugecam.Tracker()
tracker.update(dets, info)                 # is_new=False のフレームは無視
for tr in tracker.tracks:
    tr.id, tr.x, tr.y, tr.vx, tr.vy, tr.speed   # 位置 (px) / 速度 (px/s)
for tr, (x, y) in tracker.predict_all(extra_ms=0.0):
    ...                                    # 今 + extra_ms の推定中心
```

α-β フィルタ (等速モデル)。`LatencyCompensator` も内部でこれを使う (スレッドセーフな包み)。

## 8. 計測ツール

どれも `python -m` で動く。結果は同じ QPC の時計なので直接比べられる。

### キャプチャ遅延の実測 (`ketsugecam.latency`)

```bash
python -m ketsugecam.latency --seconds 30 --csv latency.csv     # 到着待ち
python -m ketsugecam.latency --seconds 30 --fps 240             # タイマー方式と比較
```

別プロセスのプローブ窓が描画直前の QPC を白黒マスとして描き、キャプチャ画像から読み戻す。
DXGI の申告に頼らない「描画 → キャプチャ完了」の実測と、LastPresentTime の検証ができる。

| 区間 (p50, 到着待ち) | 実測 |
|---|---|
| draw → present (DXGI申告) | 0.76ms |
| present → acquire | 0.035ms |
| acquire → capture_done | 0.55ms |
| **draw → capture_done** | **1.35ms** |

- プローブ窓がゲームなどで隠れると復号失敗が増える
- 光るまで (スキャンアウト・液晶応答) は測れない。フォトダイオード等が要る

### 検出の揺れとパイプラインの遅れ (`ketsugecam.diag`)

```bash
python -m ketsugecam.diag --seconds 20 --csv diag.csv
python -m ketsugecam.diag --seconds 20 --consumer-hz 60    # 表示ループ相当で受け取る
```

検出スレッド + 利用側スレッドで動かし、

- **A (検出位置のノイズ)**: 静止対象の中心・サイズ・信頼度のばらつき、フレーム間の飛び
- **B (遅れ)**: present → capture_done → infer_start → result_done → consume の p50/p95/p99/max、
  frame_id の重複・逆順・飛ばし

を出し、判断の目安を表示する。

| 判断 | 調べる場所 |
|---|---|
| 静止対象なのに中心の σ が 1px 超 | 検出位置そのもの (A) |
| frame_id の重複・逆順 | 古い結果の混入 (B) |
| 推論時間の裾が長い | GPU の取り合い・スレッド競合 (B) |
| capture_done → infer_start が推論時間を超える | 検出スレッドの詰まり (B) |

capture_done → infer_start は推論中に届いたフレームの待ちで、推論1回分までは正常。

### 入力 → 画面反応 (`ketsugecam.input_latency`)

```bash
python -m ketsugecam.input_latency --trials 40 --dx 100 --csv input.csv
python -m ketsugecam.input_latency --selftest      # 計測器自体の確認
```

SendInput の相対移動 (±dx 交互) を送り、画面中央が変わった最初のフレームを探す。
しきい値は開始前に入力なしで測ったノイズから自動で決める。入力後のフレームを取りこぼした
試行は「不確か」として分ける。

- 測れる: SendInput → ゲームが読む → 描画 → present → キャプチャ完了
- 測れない: マウス本体・USB、スキャンアウト・液晶応答
- `--selftest` はマウス移動ごとに白黒が反転する窓 (低レベルフック) で確かめる。
  ゲームがカーソルを固定していても動く。実行中は前面のアプリにも小さな移動が届く

## 9. サンプル (`新しいフォルダー/`)

| ファイル | 内容 |
|---|---|
| `main.py` | 検出スレッド + pygame 表示。補正した枠を表示し、遅延の内訳と位置合わせを毎秒出す。SPACE で一時停止 |
| `test.py` | 補正した検出を表示するだけの最小構成。Capture / Detect / Display の Hz を表示 |
| `test_position.py` | `current_boxes()` / `to_screen()` が補正済みの位置を返すかの確認 (合成 + 実機) |

`test.py` の位置の取り方:

```python
for b in current_boxes():           # = compensator.boxes()。使う直前に呼ぶ
    sx, sy = to_screen(*b.center)   # 補正済み・画面座標の中心
```

`main.py` の設定:

```python
compensation_extra_ms = 0.0     # ここで測れない後段の遅延を足す分
show_raw_boxes = False          # 確認用: 補正前の枠 (灰) と補正量の線も重ねる
```

## 10. 全部おまかせで動かす

キャプチャ → 検出 → 表示を1行で:

```python
ketsugecam.run(
    capture_size=(416, 416),   # 画面中央から撮る大きさ
    model_path="best.engine",
    detect=True,               # False にするとキャプチャだけ
    show_window=True,
    draw_boxes=True,
    show_fps=True,
)
```

```bash
python -m ketsugecam
```

**INSERT** キー (またはプレビュー上で **ESC**) で終了。設定は `__main__.py` 上の定数を書き換える。

## 11. 複数モニター

```python
print(ketsugecam.list_monitors())          # [(0, 0, 1920, 1080), (1920, 0, 3840, 1080)]

cam0 = ketsugecam.create(output_idx=0)     # 1枚目のモニター
cam1 = ketsugecam.create(output_idx=1)     # 2枚目のモニター
```

GPU が複数あるときは `device_idx` で選ぶ。

---

## 早見表

| 関数 / メソッド | 説明 |
|---|---|
| `ketsugecam.create(output_idx=0, device_idx=0, region=None, cuda=True, num_buffers=3, stats_window=512)` | カメラを作る |
| `cam.grab()` / `cam.grab_info()` | 1枚撮る (変化なしなら `None` / `(None, None)`) |
| `cam.start(target_fps=None, wait_ms=50)` / `cam.stop()` | 連続キャプチャの開始 (既定は到着待ち) / 停止 |
| `cam.get_latest_frame(timeout=1.0, copy=False)` | 最新フレーム |
| `cam.get_latest_frame_info(timeout=1.0, copy=False)` | 最新フレーム + `FrameInfo` |
| `cam.lease_latest_frame(timeout=1.0)` | コピーせずに貸し出し (`with`) |
| `cam.get_stats()` / `cam.reset_stats()` | 統計 / リセット |
| `cam.start_log(path)` / `cam.stop_log()` | CSVログ |
| `cam.release()` | 後片付け |
| `ketsugecam.detector(model_path, conf, iou, max_det, agnostic_nms)` | 検出器を作る |
| `model.predict(frame, info=None)` | 検出して結果のリストを返す |
| `model.last_timing` | 直前の `predict()` の処理時間 |
| `model.draw(img, dets)` | numpy画像に結果を描く |
| `ketsugecam.LatencyCompensator(...)` | 遅延補正 (`update` / `boxes` / `alignment`) |
| `ketsugecam.Tracker(...)` | 位置・速度の推定 |
| `ketsugecam.qpc_now()` / `qpc_to_ns(ticks)` / `QPC_FREQUENCY` | QPC 時計 |
| `ketsugecam.run(...)` | キャプチャ + 検出 + 表示をまとめて起動 |
| `ketsugecam.monitor_size()` / `list_monitors()` | モニターのサイズ / 全モニターの座標 |

| 計測ツール | 内容 |
|---|---|
| `python -m ketsugecam.latency` | 描画 → キャプチャの実測、LastPresentTime の検証 |
| `python -m ketsugecam.diag` | 検出の揺れ (A) とパイプラインの遅れ (B) |
| `python -m ketsugecam.input_latency` | 入力 → 画面反応 (C) |
