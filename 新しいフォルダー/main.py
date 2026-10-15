"""
ketsugecam を自分のループから呼ぶ例 (検出スレッド + pygame表示)

    python main.py

前処理・TensorRT推論・デコード・NMS・クラス名の読み取りは ketsugecam 内蔵。
こちらで書くのは「設定」と「取得したフレーム/検出結果をどう使うか」だけ。

    pip install pygame
"""

import sys
import threading
import time
from collections import deque
from threading import Event

import cv2
import numpy as np
import pygame

import ketsugecam

# 絵文字つきの print がコンソール(cp932)やパイプで落ちないように
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ================= 設定 =================
width_x, height_y = 416, 416

model_path = "best.engine"      # 相対パスは ketsugecam/ 内 -> カレントの順で解決
conf_threshold = 0.35
iou_threshold = 0.45
max_detections = 5
display_fps_cap = 60

# ---- YOLO 遅延補正モード ----
# 検出結果を「そのフレームの画面更新時刻」から「今」まで、対象の速度で進める。
# キャプチャ・推論・受け渡しの遅延は毎フレームの実測 (QPC) が自動で入る。
compensation_extra_ms = 0.0     # ここで測れない後段の遅延 (表示など) を足す分
show_raw_boxes = False          # 確認用: 補正前の枠 (灰) と補正量の線も重ねる

running = True

# ================= カメラ初期化 =================
# ketsugecam を先に作る: torch の CUDA 初期化より前に CUDA interop の
# コンテキストを用意する必要がある (検出器が初期化する)。
screen_w, screen_h = ketsugecam.monitor_size()
left = (screen_w - width_x) // 2
top = (screen_h - height_y) // 2

camera = ketsugecam.create(
    output_idx=0,
    region=(left, top, left + width_x, top + height_y),
    cuda=True,
)
camera.start()   # 到着待ち: 新フレームが届いた瞬間に取る

# ================= YOLOロード =================
print("🔄 YOLOモデルロード中...")
model = ketsugecam.detector(
    model_path,
    conf=conf_threshold,
    iou=iou_threshold,
    max_det=max_detections,
    agnostic_nms=True,
)
print(f"✅ モデルロード完了 | {model.version.name} | imgsz={model.imgsz} | "
      f"classes={model.names}")
# ウォームアップとCUDAグラフのキャプチャは最初の predict() で済む。

compensator = ketsugecam.LatencyCompensator(
    extra_ms=compensation_extra_ms,
)

# ================= グローバル =================
_latest_frame_gpu = None         # 表示用 (GPU のまま。CPU へは表示ループで落とす)
_latest_boxes = []
lock = threading.Lock()
frame_ready = Event()            # 新フレーム到着通知

detection_fps = 0
frame_times = deque(maxlen=100)
present_to_result = deque(maxlen=240)   # 画面更新 → 検出結果確定 (ms)
infer_times = deque(maxlen=240)         # GPU推論 (ms)

_stop = Event()                  # 終了
_pause = Event()                 # 一時停止 (SPACE)


# ================= 検出スレッド =================
def detection_thread():
    global _latest_frame_gpu, _latest_boxes, detection_fps

    frame_count = 0
    last_fps_time = time.time()

    while not _stop.is_set():
        if _pause.is_set():
            time.sleep(0.05)
            continue

        frame_start = time.perf_counter()

        # ketsugecam から GPU テンソルを取得。
        # copy=True: interopの返り値は使い回しバッファのビューなので、
        # 表示用にCPUへ落とすまで中身が書き換わらないようコピーを取る。
        frame_gpu, info = camera.get_latest_frame_info(timeout=0.005, copy=True)
        if frame_gpu is None or not info.is_new:
            continue    # 前回と同じフレーム → 重複推論しない

        try:
            # 推論。permute / float / 正規化 / BGR->RGB / NMS は内蔵。
            boxes = model.predict(frame_gpu, info)
            compensator.update(boxes, info)      # 遅延補正用に時刻つきで追跡
            tm = model.last_timing
            if not tm.warmup:
                present_to_result.append(tm.present_to_result_ns(info) / 1e6)
            infer_times.append(tm.gpu_infer_ms)

            # 表示用の CPU 転送はここではしない: 毎フレーム待つと検出が遅くなる。
            # copy=True で取ったテンソルなので、表示ループが後で読んでも書き換わらない。
            with lock:
                _latest_frame_gpu = frame_gpu
                _latest_boxes = boxes
            frame_ready.set()

            # FPS / レイテンシ計測
            frame_times.append((time.perf_counter() - frame_start) * 1000)
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                detection_fps = frame_count
                frame_count = 0
                last_fps_time = now
                avg_time = sum(frame_times) / len(frame_times)
                st = camera.get_stats()
                al = compensator.alignment()
                print(f"位置合わせ: 補正あり {al.comp_p50:.2f}/{al.comp_p95:.2f}px "
                      f"補正なし {al.raw_p50:.2f}/{al.raw_p95:.2f}px (p50/p95) | "
                      f"時間ずれ {al.time_bias_ms:+.2f}ms (動いている対象 {al.n_moving}件) | "
                      f"自動補正 {compensator.bias_ms:+.2f}ms")
                print(f"Detection FPS: {detection_fps} | "
                      f"Latency: {avg_time:.1f}ms | "
                      f"present→capture: {st.present_to_capture_ms_mean:.2f}ms | "
                      f"infer: {sum(infer_times) / len(infer_times):.2f}ms | "
                      f"present→result: {sorted(present_to_result)[len(present_to_result) // 2]:.2f}ms | "
                      f"age: {st.age_ms_mean:.2f}ms | "
                      f"skipped: {st.consumer_skipped} | "
                      f"dxgi_skipped: {st.dxgi_skipped}")

        except Exception as e:
            print(f"検出エラー: {e}")
            import traceback
            traceback.print_exc()


detect_thread = threading.Thread(target=detection_thread, name="Detection",
                                 daemon=True)
detect_thread.start()
print("✅ 検出スレッド起動")

# ================= Pygame初期化 =================
pygame.init()
screen = pygame.display.set_mode((width_x, height_y))
pygame.display.set_caption("YOLO Detection")
clock = pygame.time.Clock()

_surface_cache = None


# ================= 描画ヘルパ =================
def draw_boxes(frame_bgr, boxes, color=(0, 255, 0)):
    """枠 (太さ1) とクラス名だけを描く。信頼度・中心点は描かない。"""
    if not boxes:
        return
    for d in boxes:
        x1, y1, x2, y2 = [int(v) for v in d.xyxy]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 1)
        name = d.name
        if name:
            cv2.putText(
                frame_bgr, name, (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )


def draw_lead(frame_bgr, cboxes):
    """確認用: 補正量の線 (観測中心 → 推定中心) と何 ms 進めたか。"""
    for b in cboxes:
        cx, cy = b.center
        ox, oy = int(cx - b.lead[0]), int(cy - b.lead[1])
        cv2.line(frame_bgr, (ox, oy), (cx, cy), (0, 255, 255), 1, cv2.LINE_AA)
        x1, _, _, y2 = b.xyxy
        cv2.putText(frame_bgr, f"+{b.lead_ms:.1f}ms", (x1, min(y2 + 12, height_y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)


# ================= メインループ =================
try:
    while running:
        clock.tick(display_fps_cap)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if _pause.is_set():
                        _pause.clear()
                        print("▶ 検出再開")
                    else:
                        _pause.set()
                        print("⏸ 検出一時停止")

        # 新フレームを待機 (最大16ms)
        if not frame_ready.wait(timeout=0.016):
            continue
        frame_ready.clear()

        with lock:
            if _latest_frame_gpu is None:
                continue
            frame_gpu = _latest_frame_gpu
            raw_boxes = _latest_boxes
        frame_bgr = frame_gpu.cpu().numpy()      # 表示する分だけ CPU へ (60fps)

        # 使う枠は常に「今」の画面上の位置に補正した枠 (描画直前に計算)。
        # Detection と同じ xyxy / conf / cls / name / center を持つ。
        boxes = compensator.boxes()

        if show_raw_boxes:
            draw_boxes(frame_bgr, raw_boxes, color=(128, 128, 128))
            draw_lead(frame_bgr, boxes)
        draw_boxes(frame_bgr, boxes)

        # BGR -> RGB して pygame へ (サーフェイスは再利用)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        transposed = frame_rgb.swapaxes(0, 1)
        if _surface_cache is None or _surface_cache.get_size() != (width_x, height_y):
            _surface_cache = pygame.surfarray.make_surface(transposed)
        else:
            pygame.surfarray.blit_array(_surface_cache, transposed)

        screen.blit(_surface_cache, (0, 0))

        pygame.display.set_caption(
            f"YOLO | Detect: {detection_fps} fps | "
            f"Display: {clock.get_fps():.0f} fps | "
            f"Capture: {camera.capture_fps:.0f} fps | "
            f"Detections: {len(boxes)}"
        )
        pygame.display.flip()

finally:
    print("🛑 終了処理中...")
    running = False
    _stop.set()
    detect_thread.join(timeout=3)
    camera.stop()
    camera.release()
    pygame.quit()
    print("✅ 終了")