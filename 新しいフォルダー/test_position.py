"""
current_boxes() / to_screen() が「補正済みの位置」を返しているかの確認

    python test_position.py            # 1 (合成) と 2 (実機) の両方
    python test_position.py --synthetic   # 1 だけ (ゲーム・GPU 不要)

1. 合成データ: 動きが分かっている対象で、補正済みの位置が真の「今」の位置に
   補正前より近いことを確かめる。
2. 実機: test.py と同じ構成で動かし、
   - to_screen(*b.center) が ROI 左上を足した値になっていること
   - b.center = 補正前の中心 + b.lead になっていること
   - 実際の画面との答え合わせで、補正ありのずれが補正なしより小さいこと
   を確かめる。
"""

import sys
import threading
import time

import ketsugecam
from ketsugecam import QPC_FREQUENCY as F
from ketsugecam.detect import Detection
from ketsugecam.ketsugecam import FrameInfo

results = []


def check(ok, msg):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + msg)


# ================= 1. 合成データ =================
def synthetic():
    print("== 1. 合成データ (300px/s で右へ動く対象、240Hz、結果は 5ms 遅れて届く) ==")
    comp = ketsugecam.LatencyCompensator()
    speed, x0, y = 300.0, 100.0, 200.0
    true_x = lambda t: x0 + speed * t / F
    for k in range(240):                         # 1 秒分
        present = k * F // 240
        x = true_x(present)
        d = Detection((round(x) - 10, 180, round(x) + 10, 220), 0.9, 1, "t")
        comp.update([d], FrameInfo(k + 1, 0, 0, 0, 0, present, 1, 416, 416))
    last_present = 239 * F // 240
    now = last_present + 5 * F // 1000           # 最後のフレームから 5ms 後に使う

    b, = comp.boxes(t_qpc=now)
    raw_x = round(true_x(last_present))          # 補正前 = 最後に検出した位置
    err_comp = abs(b.center[0] - true_x(now))
    err_raw = abs(raw_x - true_x(now))
    print(f"   今の真の x {true_x(now):.1f} / 補正済み {b.center[0]} / 補正前 {raw_x} "
          f"/ 補正量 {b.lead[0]:+.2f}px ({b.lead_ms:.1f}ms)")
    check(b.lead_ms > 0 and b.lead[0] > 0, "補正済みの位置は進行方向へ進めてある")
    check(err_comp < err_raw, f"補正済みのほうが今の位置に近い (ずれ {err_comp:.2f}px < {err_raw:.2f}px)")

    later = now + 10 * F // 1000                 # さらに 10ms 後に呼ぶ
    b2, = comp.boxes(t_qpc=later)
    check(b2.center[0] > b.center[0], "後で呼ぶほど先の位置になる (呼んだ瞬間が基準)")

    stop = ketsugecam.LatencyCompensator()       # 止まっている対象は動かさない
    for k in range(60):
        stop.update([Detection((190, 180, 210, 220), 0.9, 1, "t")],
                    FrameInfo(k + 1, 0, 0, 0, 0, k * F // 240, 1, 416, 416))
    s, = stop.boxes(t_qpc=60 * F // 240 + 5 * F // 1000)
    check(s.lead == (0.0, 0.0) and s.center == (200, 200), "止まっている対象は補正量 0")


# ================= 2. 実機 =================
def live(seconds=10.0):
    print(f"\n== 2. 実機 ({seconds:.0f} 秒。画面中央に動く対象を映してください) ==")
    width_x = height_y = 416
    screen_w, screen_h = ketsugecam.monitor_size()
    left, top = (screen_w - width_x) // 2, (screen_h - height_y) // 2
    camera = ketsugecam.create(region=(left, top, left + width_x, top + height_y))
    camera.start()
    model = ketsugecam.detector("best.engine", conf=0.35, iou=0.45, max_det=5,
                                agnostic_nms=True)
    compensator = ketsugecam.LatencyCompensator()

    def current_boxes():                         # test.py と同じ
        return compensator.boxes()

    def to_screen(x, y):
        return left + x, top + y

    stop = threading.Event()

    def detection_thread():
        while not stop.is_set():
            frame, info = camera.get_latest_frame_info(timeout=0.005, copy=True)
            if frame is None or not info.is_new:
                continue
            compensator.update(model.predict(frame, info), info)

    th = threading.Thread(target=detection_thread, daemon=True)
    th.start()

    n = screen_ok = lead_ok = moved = 0
    shown = 0
    t_end = time.perf_counter() + seconds
    try:
        while time.perf_counter() < t_end:
            time.sleep(1 / 60)
            with compensator._lock:              # 確認用: 補正前の中心 (追跡の位置)
                before = {tr.id: (tr.x, tr.y) for tr in compensator._tracker.tracks}
            for b in current_boxes():
                sx, sy = to_screen(*b.center)    # ← 使い方そのもの
                n += 1
                screen_ok += (sx, sy) == (left + b.center[0], top + b.center[1])
                bx, by = before.get(b.track_id, (None, None))
                if bx is not None:
                    # 補正済みの中心 = 補正前の中心 + 補正量 (整数への丸めで ±1px)
                    lead_ok += (abs(b.center[0] - (bx + b.lead[0])) <= 1
                                and abs(b.center[1] - (by + b.lead[1])) <= 1)
                else:
                    lead_ok += 1
                moved += b.lead != (0.0, 0.0)
                if shown < 5 and b.lead != (0.0, 0.0):
                    shown += 1
                    print(f"   {b.name}: 画面座標 ({sx}, {sy}) 補正量 "
                          f"({b.lead[0]:+.1f}, {b.lead[1]:+.1f})px +{b.lead_ms:.1f}ms")
    finally:
        stop.set()
        th.join(timeout=2)
        al = compensator.alignment()
        camera.release()

    if n == 0:
        print("   検出がありませんでした。画面中央に対象を映して再実行してください。")
        results.append(False)
        return
    check(screen_ok == n, f"to_screen は ROI 左上 ({left}, {top}) を足した画面座標 ({screen_ok}/{n})")
    check(lead_ok == n, f"b.center = 補正前の中心 + b.lead ({lead_ok}/{n})")
    print(f"   補正量が 0 でない (動いている対象): {moved}/{n}")
    print(f"   実際の画面とのずれ p50/p95: 補正あり {al.comp_p50:.2f}/{al.comp_p95:.2f}px, "
          f"補正なし {al.raw_p50:.2f}/{al.raw_p95:.2f}px ({al.n}件、うち動いている対象 {al.n_moving}件)")
    # 止まっている対象は補正量 0 なので、補正あり/なしの差は検出の揺れだけ。
    # 比べる意味があるのは動いている対象が十分あるときだけ。
    if al.n_moving >= 50:
        check(al.comp_p95 <= al.raw_p95, "補正ありのほうが実際の画面の位置に近い (p95)")
    else:
        print("   SKIP 補正あり/なしの比較: 動いている対象が少ない。動く対象を映して再実行してください。")


if __name__ == "__main__":
    synthetic()
    if "--synthetic" not in sys.argv:
        live()
    print(f"\n{sum(results)}/{len(results)} PASS")
    sys.exit(0 if all(results) else 1)
