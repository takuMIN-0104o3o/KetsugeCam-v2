"""
補正した検出を表示するだけの最小構成

    python test.py      (ESC で終了)

位置の扱い:
    model.predict() の結果        … 補正前 (そのフレームを撮った時点の位置)
    current_boxes() / compensator.boxes() … 補正済み (呼んだ瞬間の画面上の位置)

    描画でもそれ以外でも、位置を使うときは current_boxes() から取る。
    呼んだ瞬間を「今」として計算するので、使う直前に呼ぶ (取っておくと古くなる)。
"""

import threading
import time

import cv2
import pygame

import ketsugecam

# ================= 設定 =================
width_x, height_y = 416, 416
model_path = "best.engine"
conf_threshold = 0.35
iou_threshold = 0.45
max_detections = 5
display_fps_cap = 60

# ================= 初期化 =================
# ketsugecam を先に作る (torch の CUDA 初期化より前に interop を用意する)
screen_w, screen_h = ketsugecam.monitor_size()
left = (screen_w - width_x) // 2
top = (screen_h - height_y) // 2
camera = ketsugecam.create(region=(left, top, left + width_x, top + height_y))
camera.start()

model = ketsugecam.detector(model_path, conf=conf_threshold, iou=iou_threshold,
                            max_det=max_detections, agnostic_nms=True)
compensator = ketsugecam.LatencyCompensator()

_latest_frame = None
lock = threading.Lock()
stop = threading.Event()
detect_hz = 0.0                  # 検出 (推論+補正の更新) の実測回数/秒


# ================= 検出スレッド =================
def detection_thread():
    global _latest_frame, detect_hz
    count, t0 = 0, time.perf_counter()
    while not stop.is_set():
        frame, info = camera.get_latest_frame_info(timeout=0.005, copy=True)
        if frame is None or not info.is_new:
            continue
        compensator.update(model.predict(frame, info), info)
        with lock:
            _latest_frame = frame
        count += 1
        dt = time.perf_counter() - t0
        if dt >= 1.0:
            detect_hz = count / dt
            count, t0 = 0, time.perf_counter()


threading.Thread(target=detection_thread, daemon=True).start()


# ================= 補正済みの位置 =================
def current_boxes():
    """補正済みの検出 (呼んだ瞬間の画面上の位置)。描画以外でもこれを使う。

    各要素: b.xyxy (枠) / b.center (中心) / b.conf / b.cls / b.name
            b.lead (補正量 px) / b.lead_ms (何 ms 進めたか)
    座標は ROI 内 (ROI の左上が 0,0)。画面座標は to_screen() で変換する。
    """
    return compensator.boxes()


def to_screen(x, y):
    """ROI 内の座標 → 画面全体の座標。"""
    return left + x, top + y

# ================= 表示 =================
pygame.init()
screen = pygame.display.set_mode((width_x, height_y))
clock = pygame.time.Clock()

try:
    running = True
    last_print = time.perf_counter()
    while running:
        clock.tick(display_fps_cap)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        with lock:
            frame = _latest_frame
        if frame is None:
            continue
        img = frame.cpu().numpy()

        # 補正済みの位置を描画の直前に取る (描画以外で使う場合も同じ)
        for b in current_boxes():
            x1, y1, x2, y2 = b.xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if b.name:
                cv2.putText(img, b.name, (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        screen.blit(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
        pygame.display.flip()

        # Hz 測定: キャプチャ / 検出 / 表示
        now = time.perf_counter()
        if now - last_print >= 1.0:
            last_print = now
            st = camera.get_stats()
            msg = (f"Capture {st.capture_fps:.0f}Hz | Detect {detect_hz:.0f}Hz | "
                   f"Display {clock.get_fps():.0f}Hz")
            pygame.display.set_caption(msg)
            print(msg)
finally:
    stop.set()
    camera.release()
    pygame.quit()
