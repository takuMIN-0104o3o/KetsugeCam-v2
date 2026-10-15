"""
ketsugecam/ketsugecam.py

DXGI → CUDA 直接転送。CPUノータッチ。
返り値: torch.Tensor (CUDA, uint8, [H, W, 3], BGR)

フロー (CUDA interop有効時):
    DXGI Texture (GPU VRAM)
        ↓ CopySubresourceRegion  ROIのみ GPU→GPU
    D3D11 DEFAULT Texture (GPU VRAM)
        ↓ cudaGraphicsD3D11RegisterResource
        ↓ cudaGraphicsMapResources
        ↓ cudaMemcpy2DFromArray    GPU→GPU
    torch.Tensor (CUDA, BGRA)
        ↓ [..., :3] スライス (ゼロコピー)
    torch.Tensor (CUDA, BGR)  ← CPUノータッチ

フロー (CPUフォールバック):
    DXGI → Staging(Map) → torch.frombuffer → .cuda()

フレームのメタデータ (FrameInfo):
    frame_id           キャプチャした新フレームの連番 (1始まり)
    acquire_ns         AcquireNextFrame が返った時刻   (time.perf_counter_ns)
    capture_done_ns    テンソルへのコピーが完了した時刻 (time.perf_counter_ns)
    present_time_raw   DXGI の LastPresentTime そのまま (QPCティック)
    present_to_capture_ns
                       present → キャプチャ完了。キャプチャ完了時に QPC も
                       読んでおき、QPC同士で引き算してから ns に直す。
                       perf_counter_ns と LastPresentTime を直接引かない。
"""

import csv
import ctypes
import ctypes.wintypes as wintypes
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Thread, Event, Lock
from typing import Iterator, List, Optional, Tuple

import comtypes
import torch

from .libs import (
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
    D3D11_TEXTURE2D_DESC, D3D11_BOX, DXGI_SAMPLE_DESC,
    D3D11_USAGE_STAGING, D3D11_USAGE_DEFAULT,
    D3D11_CPU_ACCESS_READ,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0,
    DXGI_FORMAT_B8G8R8A8_UNORM,
    IDXGIFactory1, IDXGIAdapter1, IDXGIOutput, IDXGIOutput1,
    IDXGIOutputDuplication, IDXGIResource, IDXGISurface,
    DXGI_OUTPUT_DESC, DXGI_OUTDUPL_FRAME_INFO, DXGI_MAPPED_RECT,
    DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT, DXGI_ERROR_NOT_FOUND,
)

# ─── CUDA Runtime (ctypes直接呼び出し) ───────────────────────────────────────
import os, glob as _glob

_CUDA_AVAILABLE = False
_cudart = None

def _load_cudart():
    # 既知パス優先
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudart64_12.dll",
    ]
    # 自動探索
    for pattern in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\cudart64_*.dll",
        r"C:\Windows\System32\cudart64_*.dll",
    ]:
        candidates += _glob.glob(pattern)

    for path in candidates:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return None

_cudart = _load_cudart()
if _cudart is not None:
    # 使用する関数のシグネチャ設定
    # cudaGraphicsD3D11RegisterResource(resource*, pD3DResource, flags)
    _cudart.cudaGraphicsD3D11RegisterResource.restype  = ctypes.c_int
    _cudart.cudaGraphicsD3D11RegisterResource.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # cudaGraphicsResource_t*
        ctypes.c_void_p,                   # ID3D11Resource*
        ctypes.c_uint,                     # flags
    ]
    # cudaGraphicsMapResources(count, resources*, stream)
    _cudart.cudaGraphicsMapResources.restype  = ctypes.c_int
    _cudart.cudaGraphicsMapResources.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    # cudaGraphicsUnmapResources
    _cudart.cudaGraphicsUnmapResources.restype  = ctypes.c_int
    _cudart.cudaGraphicsUnmapResources.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    # cudaGraphicsResourceGetMappedMipmappedArray(array*, resource)
    _cudart.cudaGraphicsResourceGetMappedMipmappedArray.restype  = ctypes.c_int
    _cudart.cudaGraphicsResourceGetMappedMipmappedArray.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    # cudaGetMipmappedArrayLevel(array*, mipmapped, level)
    _cudart.cudaGetMipmappedArrayLevel.restype  = ctypes.c_int
    _cudart.cudaGetMipmappedArrayLevel.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    # cudaMemcpy2DFromArray(dst, dpitch, src, woff, hoff, width, height, kind)
    _cudart.cudaMemcpy2DFromArray.restype  = ctypes.c_int
    _cudart.cudaMemcpy2DFromArray.argtypes = [
        ctypes.c_void_p,  # dst
        ctypes.c_size_t,  # dpitch
        ctypes.c_void_p,  # src cudaArray_t
        ctypes.c_size_t,  # wOffset
        ctypes.c_size_t,  # hOffset
        ctypes.c_size_t,  # width (bytes)
        ctypes.c_size_t,  # height
        ctypes.c_int,     # kind (3=DeviceToDevice)
    ]
    # cudaGraphicsUnregisterResource
    _cudart.cudaGraphicsUnregisterResource.restype  = ctypes.c_int
    _cudart.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]

    _cudart.cudaSetDevice.restype  = ctypes.c_int
    _cudart.cudaSetDevice.argtypes = [ctypes.c_int]

    _CUDA_AVAILABLE = True

# cudaGraphicsRegisterFlagsReadOnly = 1
_CUDA_GRAPHICS_REGISTER_FLAGS_NONE     = 0  # ← これが正解
_CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

# ─── Windows 高精度タイマー ───────────────────────────────────────────────────
_kernel32 = ctypes.windll.kernel32
_INFINITE  = 0xFFFFFFFF
_WAIT_FAILED = 0xFFFFFFFF
_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x1F0003
_DXGI_MAP_READ = 1


def _create_timer():
    h = _kernel32.CreateWaitableTimerExW(
        None, None, _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, _TIMER_ALL_ACCESS
    )
    if h == 0:
        raise ctypes.WinError()
    return h


def _set_timer(handle, target_fps: float):
    interval = ctypes.c_longlong(int(-10_000_000 / target_fps))
    if not _kernel32.SetWaitableTimer(handle, ctypes.byref(interval), 0, None, None, 0):
        raise ctypes.WinError()


def _wait_timer(handle):
    return _kernel32.WaitForSingleObject(handle, _INFINITE)


def _cancel_timer(handle):
    _kernel32.CancelWaitableTimer(handle)
    _kernel32.CloseHandle(handle)


# ─── QPC (DXGI の LastPresentTime と同じ時計) ─────────────────────────────────
_qpc_freq = ctypes.c_longlong(0)
_kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_freq))
QPC_FREQUENCY: int = _qpc_freq.value


def qpc_now() -> int:
    """QueryPerformanceCounter の生の値 (ティック)。"""
    v = ctypes.c_longlong(0)
    _kernel32.QueryPerformanceCounter(ctypes.byref(v))
    return v.value


def qpc_to_ns(ticks: int) -> int:
    """QPCティック数 (差分) を ns に直す。"""
    return ticks * 1_000_000_000 // QPC_FREQUENCY


# ─── フレームのメタデータ / 統計 ──────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FrameInfo:
    """1フレーム分のメタデータ。画像と同じロックの下で一緒に取り出す。

    時刻は2系統ある:
      - *_ns:            time.perf_counter_ns() の値 (PC側の単調時計)
      - present_time_raw: DXGI の LastPresentTime (QPCティック、無変換)
    両者を直接引き算しないこと。present からの遅延は present_to_capture_ns を使う。
    """
    frame_id: int
    acquire_ns: int             # AcquireNextFrame が返った時刻
    acquire_qpc: int            # 同じ瞬間の QPC
    capture_done_ns: int        # テンソルへのコピー完了時刻
    capture_done_qpc: int       # 同じ瞬間の QPC (present との比較用)
    present_time_raw: int       # DXGI LastPresentTime (QPCティック)
    accumulated_frames: int     # DXGI AccumulatedFrames (前回取得からのPresent数)
    width: int
    height: int
    is_new: bool = True         # get_latest_frame_info() が前回返したものと別か

    @property
    def present_to_capture_ns(self) -> Optional[int]:
        """Present → キャプチャ完了 (ns)。QPC同士で引いてから換算する。"""
        if self.present_time_raw <= 0:
            return None
        return qpc_to_ns(self.capture_done_qpc - self.present_time_raw)

    @property
    def present_to_acquire_ns(self) -> Optional[int]:
        """Present → AcquireNextFrame が返るまで (ns)。DWM + ポーリング待ち。"""
        if self.present_time_raw <= 0:
            return None
        return qpc_to_ns(self.acquire_qpc - self.present_time_raw)

    @property
    def clock_consistent(self) -> bool:
        """present ≤ acquire ≤ capture_done が QPC 上で成り立つか。
        False が出るなら LastPresentTime を QPC として扱う前提が崩れている。"""
        return 0 < self.present_time_raw <= self.acquire_qpc <= self.capture_done_qpc

    @property
    def copy_ns(self) -> int:
        """AcquireNextFrame → コピー完了 (ns)。ketsugecam 内部の処理時間。"""
        return self.capture_done_ns - self.acquire_ns

    @property
    def skipped_presents(self) -> int:
        """このフレームより前に DXGI が合成して見られなかった Present 数。
        出力 (モニター) 全体の Present なので、ROI外の更新も含む。"""
        return max(self.accumulated_frames - 1, 0)

    def age_ns(self, now_ns: Optional[int] = None) -> int:
        """キャプチャ完了から now_ns (既定: 今) までの経過 ns。"""
        if now_ns is None:
            now_ns = time.perf_counter_ns()
        return now_ns - self.capture_done_ns


@dataclass(frozen=True)
class FrameStats:
    """get_stats() のスナップショット。*_ms は直近 window フレームの集計。"""
    captured: int               # キャプチャした新フレーム数
    interval_ms_mean: float     # キャプチャ完了の間隔
    interval_ms_min: float
    interval_ms_max: float
    interval_ms_std: float
    capture_fps: float          # interval_ms_mean から求めたFPS
    present_to_capture_ms_mean: float
    present_to_capture_ms_max: float
    copy_ms_mean: float
    dxgi_skipped: int           # DXGI が合成して取り逃した Present の合計
    busy_drops: int             # 空きバッファが無くて捨てたフレーム
    cursor_only: int            # カーソルだけの更新 (画像は同じなので捨てた)
    timeouts: int               # AcquireNextFrame がタイムアウトした回数
    access_lost: int            # DXGI_ERROR_ACCESS_LOST の回数
    clock_violations: int       # present ≤ acquire ≤ done (QPC) が崩れたフレーム数
    consumed: int               # get_latest_frame* が返した回数
    consumer_repeats: int       # 前回と同じ frame_id を返した回数 (重複処理)
    consumer_skipped: int       # 受け取らずに上書きされたフレーム数 (frame_id の飛び)
    age_ms_mean: float          # キャプチャ完了 → 利用側が受け取るまで
    age_ms_max: float


def _mean_std(xs) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


class _Stats:
    """内部カウンタ。KetsugeCam._lock の下で更新・読み出しする。"""

    def __init__(self, window: int):
        self.window = window
        self.reset()

    def reset(self):
        w = self.window
        self.captured = 0
        self.dxgi_skipped = 0
        self.busy_drops = 0
        self.cursor_only = 0
        self.timeouts = 0
        self.access_lost = 0
        self.clock_violations = 0
        self.consumed = 0
        self.consumer_repeats = 0
        self.consumer_skipped = 0
        self.last_done_ns = 0
        self.last_consumed_id = 0
        self.intervals = deque(maxlen=w)   # ns
        self.p2c = deque(maxlen=w)         # ns
        self.copy = deque(maxlen=w)        # ns
        self.ages = deque(maxlen=w)        # ns

    def snapshot(self) -> FrameStats:
        ms = 1e-6
        iv = [x * ms for x in self.intervals]
        iv_mean, iv_std = _mean_std(iv)
        p2c = [x * ms for x in self.p2c]
        ages = [x * ms for x in self.ages]
        return FrameStats(
            captured=self.captured,
            interval_ms_mean=iv_mean,
            interval_ms_min=min(iv, default=0.0),
            interval_ms_max=max(iv, default=0.0),
            interval_ms_std=iv_std,
            capture_fps=1000.0 / iv_mean if iv_mean > 0 else 0.0,
            present_to_capture_ms_mean=_mean_std(p2c)[0],
            present_to_capture_ms_max=max(p2c, default=0.0),
            copy_ms_mean=_mean_std([x * ms for x in self.copy])[0],
            dxgi_skipped=self.dxgi_skipped,
            busy_drops=self.busy_drops,
            cursor_only=self.cursor_only,
            timeouts=self.timeouts,
            access_lost=self.access_lost,
            clock_violations=self.clock_violations,
            consumed=self.consumed,
            consumer_repeats=self.consumer_repeats,
            consumer_skipped=self.consumer_skipped,
            age_ms_mean=_mean_std(ages)[0],
            age_ms_max=max(ages, default=0.0),
        )


_LOG_HEADER = (
    "event", "frame_id", "t_ns", "capture_done_ns", "present_time_raw",
    "acquire_qpc", "capture_done_qpc",
    "present_to_capture_us", "accumulated_frames", "interval_us",
    "copy_us", "age_us", "is_new",
)


# ─── DXGI/D3D 初期化ヘルパー ──────────────────────────────────────────────────
def _create_dxgi_factory():
    fn = ctypes.windll.dxgi.CreateDXGIFactory1
    fn.argtypes = (comtypes.GUID, ctypes.POINTER(ctypes.c_void_p))
    fn.restype  = ctypes.c_int32
    p = ctypes.c_void_p(0)
    fn(IDXGIFactory1._iid_, ctypes.byref(p))
    return ctypes.POINTER(IDXGIFactory1)(p.value)


def _enum_adapters(factory):
    adapters, i = [], 0
    while True:
        try:
            p = ctypes.POINTER(IDXGIAdapter1)()
            factory.EnumAdapters1(i, ctypes.byref(p))
            adapters.append(p)
            i += 1
        except comtypes.COMError as e:
            if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == e.args[0]:
                break
            raise
    return adapters


def _enum_outputs(adapter):
    outputs, i = [], 0
    while True:
        try:
            p = ctypes.POINTER(IDXGIOutput)()
            adapter.EnumOutputs(i, ctypes.byref(p))
            outputs.append(p.QueryInterface(IDXGIOutput1))
            i += 1
        except comtypes.COMError as e:
            if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == e.args[0]:
                break
            raise
    return outputs


def _create_d3d11_device(adapter):
    fn = ctypes.windll.d3d11.D3D11CreateDevice
    levels = (ctypes.c_uint * 3)(
        D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0
    )
    device  = ctypes.POINTER(ID3D11Device)()
    context = ctypes.POINTER(ID3D11DeviceContext)()
    fn(
        adapter, 0, None,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        ctypes.byref(levels), 3, 7,
        ctypes.byref(device), None, ctypes.byref(context),
    )
    im_ctx = ctypes.POINTER(ID3D11DeviceContext)()
    device.GetImmediateContext(ctypes.byref(im_ctx))
    return device, im_ctx


def _create_staging_texture(device, width, height):
    """CPU読み取り用 Staging テクスチャ (フォールバック用)"""
    desc = D3D11_TEXTURE2D_DESC()
    desc.Width          = width
    desc.Height         = height
    desc.MipLevels      = 1
    desc.ArraySize      = 1
    desc.Format         = DXGI_FORMAT_B8G8R8A8_UNORM
    desc.SampleDesc     = DXGI_SAMPLE_DESC(1, 0)
    desc.Usage          = D3D11_USAGE_STAGING
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ
    desc.BindFlags      = 0
    desc.MiscFlags      = 0
    tex = ctypes.POINTER(ID3D11Texture2D)()
    device.CreateTexture2D(ctypes.byref(desc), None, ctypes.byref(tex))
    surf = tex.QueryInterface(IDXGISurface)
    return tex, surf


def _create_default_texture(device, width, height):
    """CUDA interop用 DEFAULT テクスチャ (GPU VRAM常駐)"""
    D3D11_RESOURCE_MISC_SHARED = 0x2
    desc = D3D11_TEXTURE2D_DESC()
    desc.Width          = width
    desc.Height         = height
    desc.MipLevels      = 1
    desc.ArraySize      = 1
    desc.Format         = DXGI_FORMAT_B8G8R8A8_UNORM
    desc.SampleDesc     = DXGI_SAMPLE_DESC(1, 0)
    desc.Usage          = D3D11_USAGE_DEFAULT
    desc.CPUAccessFlags = 0
    desc.BindFlags      = 0
    desc.MiscFlags      = D3D11_RESOURCE_MISC_SHARED
    tex = ctypes.POINTER(ID3D11Texture2D)()
    device.CreateTexture2D(ctypes.byref(desc), None, ctypes.byref(tex))
    return tex


# ─── モニター情報 (公開API) ───────────────────────────────────────────────────
def list_monitors(device_idx: int = 0) -> list[Tuple[int, int, int, int]]:
    """
    モニターごとのデスクトップ座標 (left, top, right, bottom) を列挙する。

    region を自前で計算するとき用。D3D11デバイスを作らずDXGIだけで引くので、
    KetsugeCam の生成前に呼んでもCUDAコンテキストの初期化順序に影響しない。

    Args:
        device_idx: GPUアダプター番号 (0=メイン)
    """
    factory  = _create_dxgi_factory()
    adapters = _enum_adapters(factory)
    outputs  = _enum_outputs(adapters[device_idx])

    rects = []
    for p_out in outputs:
        desc = DXGI_OUTPUT_DESC()
        p_out.GetDesc(ctypes.byref(desc))
        c = desc.DesktopCoordinates
        rects.append((c.left, c.top, c.right, c.bottom))
    return rects


def monitor_size(output_idx: int = 0, device_idx: int = 0) -> Tuple[int, int]:
    """
    モニターの解像度 (width, height) を返す。

    GetSystemMetrics と違い DXGI が報告する実際の複製面サイズなので、
    DPIスケーリング環境でも region とズレない。
    """
    left, top, right, bottom = list_monitors(device_idx)[output_idx]
    return (right - left, bottom - top)


class CUDA_MEMCPY2D(ctypes.Structure):
    _fields_ = [
        ("srcXInBytes",   ctypes.c_size_t),
        ("srcY",          ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_int),   # CU_MEMORYTYPE_ARRAY=3
        ("srcHost",       ctypes.c_void_p),
        ("srcDevice",     ctypes.c_uint64),
        ("srcArray",      ctypes.c_void_p),
        ("srcPitch",      ctypes.c_size_t),
        ("dstXInBytes",   ctypes.c_size_t),
        ("dstY",          ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_int),   # CU_MEMORYTYPE_DEVICE=2
        ("dstHost",       ctypes.c_void_p),
        ("dstDevice",     ctypes.c_uint64),
        ("dstArray",      ctypes.c_void_p),
        ("dstPitch",      ctypes.c_size_t),
        ("WidthInBytes",  ctypes.c_size_t),
        ("Height",        ctypes.c_size_t),
    ]


# ─── KetsugeCam ───────────────────────────────────────────────────────────────
class KetsugeCam:
    """
    DXGI → CUDA 直接転送キャプチャ。CPUノータッチ。

    cuda-python が使える場合:
        GPU → GPU のみ。CPU転送ゼロ。
    フォールバック:
        GPU → CPU → CUDA (従来方式)
    """

    def __init__(
        self,
        output_idx: int = 0,
        device_idx: int = 0,
        region: Optional[Tuple[int, int, int, int]] = None,
        cuda: bool = True,
        num_buffers: int = 3,
        stats_window: int = 512,
    ):
        """
        Args:
            num_buffers:  interop時のGPU出力バッファ数 (2以上)。最新フレームと
                          貸し出し中のバッファは上書きしないので、copy=False の
                          返り値は少なくとも num_buffers-1 フレームの間は有効。
            stats_window: get_stats() の平均・最大を取る直近フレーム数
        """
        if num_buffers < 2:
            raise ValueError("num_buffers は 2 以上")
        self._num_buffers = num_buffers
        self._output_idx  = output_idx
        self._device_idx  = device_idx

        # ── DXGI/D3D初期化 ── (torchより先にやる)
        factory  = _create_dxgi_factory()
        adapters = _enum_adapters(factory)
        adapter  = adapters[device_idx]
        self._adapter = adapter  # interop初期化で使う
        self._device, self._ctx = _create_d3d11_device(adapter)

        outputs  = _enum_outputs(adapter)
        p_out    = outputs[output_idx]

        out_desc = DXGI_OUTPUT_DESC()
        p_out.GetDesc(ctypes.byref(out_desc))
        self.width  = out_desc.DesktopCoordinates.right  - out_desc.DesktopCoordinates.left
        self.height = out_desc.DesktopCoordinates.bottom - out_desc.DesktopCoordinates.top

        if region is None:
            region = (0, 0, self.width, self.height)
        self._validate_region(region)
        self.region = region

        roi_w = region[2] - region[0]
        roi_h = region[3] - region[1]
        self._roi_w = roi_w
        self._roi_h = roi_h

        # ── Duplicator ──
        self._dupl = ctypes.POINTER(IDXGIOutputDuplication)()
        p_out.DuplicateOutput(self._device, ctypes.byref(self._dupl))

        # ── D3D11_BOX キャッシュ ──
        self._roi_box        = D3D11_BOX()
        self._roi_box.left   = region[0]
        self._roi_box.top    = region[1]
        self._roi_box.front  = 0
        self._roi_box.right  = region[2]
        self._roi_box.bottom = region[3]
        self._roi_box.back   = 1

        self._frame_info = DXGI_OUTDUPL_FRAME_INFO()
        self._rect       = DXGI_MAPPED_RECT()

        # ── CUDA interop 初期化 ──
        # ★ torch.cuda.is_available() より先に呼ぶ
        # PyTorchがCUDAコンテキストを作る前にD3D11と紐付ける必要がある
        self._interop  = False
        self._cuda_res = None
        self._gpu_buf  = None
        self._gpu_ptr  = None
        self._stage_tex  = None
        self._stage_surf = None

        if cuda and _CUDA_AVAILABLE:
            self._init_cuda_interop(roi_w, roi_h)

        # interop後にtorchのCUDA有効確認
        self.cuda = cuda and torch.cuda.is_available()

        # interop失敗 or 非CUDA → Stagingフォールバック
        if not self._interop:
            self._stage_tex, self._stage_surf = _create_staging_texture(
                self._device, roi_w, roi_h
            )

        # ── 連続キャプチャ用 ──
        # _latest / _latest_info / _latest_slot / _leases / _stats は _lock の下で扱う。
        # 画像と時刻を必ず同じロックの下で読み書きし、別フレームの組み合わせを防ぐ。
        self._lock = Lock()
        self._latest: Optional[torch.Tensor] = None
        self._latest_info: Optional[FrameInfo] = None
        self._latest_slot = -1
        self._write_slot  = -1
        self._leases = [0] * num_buffers
        self._next_frame_id = 1
        self._stats = _Stats(stats_window)
        self._frame_event = Event()
        self._stop_event  = Event()
        self._thread: Optional[Thread] = None
        self.is_capturing  = False
        self._capture_fps: float = 0.0

        # ── 計測ログ ──
        self._log_q: Optional[deque] = None
        self._log_thread: Optional[Thread] = None
        self._log_stop = Event()

    def _init_cuda_interop(self, roi_w, roi_h):
        try:
            # ── Driver API (nvcuda.dll) でD3D11と紐付けたCUDAコンテキスト作成 ──
            _cuda_driver = ctypes.CDLL(r'C:\Windows\System32\nvcuda.dll')

            _cuda_driver.cuInit.restype  = ctypes.c_int
            _cuda_driver.cuInit.argtypes = [ctypes.c_uint]
            err = _cuda_driver.cuInit(0)
            if err != 0:
                raise RuntimeError(f"cuInit failed: {err}")

            _cuda_driver.cuD3D11GetDevice.restype  = ctypes.c_int
            _cuda_driver.cuD3D11GetDevice.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
            cu_dev      = ctypes.c_int(-1)
            adapter_ptr = ctypes.cast(self._adapter, ctypes.c_void_p).value
            err = _cuda_driver.cuD3D11GetDevice(ctypes.byref(cu_dev), ctypes.c_void_p(adapter_ptr))
            if err != 0:
                raise RuntimeError(f"cuD3D11GetDevice failed: {err}")
            print(f"[ketsugecam] CUDA device: {cu_dev.value}")

            # D3D11デバイスと紐付けたCUDAコンテキスト作成
            _cuda_driver.cuD3D11CtxCreate_v2.restype  = ctypes.c_int
            _cuda_driver.cuD3D11CtxCreate_v2.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                ctypes.c_uint, ctypes.c_void_p,
            ]
            d3d_ptr = ctypes.cast(self._device, ctypes.c_void_p).value
            cu_ctx  = ctypes.c_void_p(0)
            err = _cuda_driver.cuD3D11CtxCreate_v2(
                ctypes.byref(cu_ctx), cu_dev.value, 0, ctypes.c_void_p(d3d_ptr)
            )
            if err != 0:
                raise RuntimeError(f"cuD3D11CtxCreate_v2 failed: {err}")

            # コンテキストをカレントに設定
            _cuda_driver.cuCtxSetCurrent.restype  = ctypes.c_int
            _cuda_driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            err = _cuda_driver.cuCtxSetCurrent(cu_ctx.value)
            if err != 0:
                raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

            # CUDA interop用テクスチャ作成
            self._interop_tex = _create_default_texture(self._device, roi_w, roi_h)
            # ★ p1方式: ctypes.cast(tex, c_void_p).value が正しいポインタ
            tex_ptr = ctypes.cast(self._interop_tex, ctypes.c_void_p).value
            if not tex_ptr:
                raise RuntimeError("テクスチャ作成失敗 (NULL)")
            print(f"[ketsugecam] tex_ptr: {hex(tex_ptr)}")

            # ★ torch.emptyを先に呼んでPyTorchのCUDAコンテキストを確立
            # これをRegisterResourceより後にやるとコンテキストが変わって400になる
            # 出力バッファはリングで複数持つ: 利用側が読んでいる最中のバッファを
            # 次のキャプチャで上書きしないため。
            self._gpu_bufs = [
                torch.empty((roi_h, roi_w, 4), dtype=torch.uint8, device="cuda")
                for _ in range(self._num_buffers)
            ]
            self._gpu_views = [b[..., :3] for b in self._gpu_bufs]  # BGRA → BGR (ゼロコピー)
            self._gpu_buf = self._gpu_bufs[0]
            self._gpu_ptr = self._gpu_buf.data_ptr()

            # コンテキストをD3D11のものに戻す
            err = _cuda_driver.cuCtxSetCurrent(cu_ctx.value)
            if err != 0:
                raise RuntimeError(f"cuCtxSetCurrent(restore) failed: {err}")

            # ★ Driver APIで登録
            _cuda_driver.cuGraphicsD3D11RegisterResource.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsD3D11RegisterResource.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            self._cuda_res = ctypes.c_void_p(0)
            err = _cuda_driver.cuGraphicsD3D11RegisterResource(
                ctypes.byref(self._cuda_res),
                ctypes.c_void_p(tex_ptr),
                _CUDA_GRAPHICS_REGISTER_FLAGS_NONE,
            )
            print(f"[ketsugecam] cuGraphicsD3D11RegisterResource: err={err} res={self._cuda_res.value}")
            if err != 0:
                raise RuntimeError(f"cuGraphicsD3D11RegisterResource failed: {err}")

            # Driver API関数シグネチャ設定
            _cuda_driver.cuCtxSetCurrent.restype  = ctypes.c_int
            _cuda_driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
            _cuda_driver.cuCtxGetCurrent.restype  = ctypes.c_int
            _cuda_driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            _cuda_driver.cuGraphicsMapResources.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsMapResources.argtypes = [
                ctypes.c_uint,      # count
                ctypes.c_void_p,    # resources* (配列の先頭アドレス)
                ctypes.c_void_p,    # stream
            ]
            _cuda_driver.cuGraphicsUnmapResources.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsUnmapResources.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            _cuda_driver.cuGraphicsResourceGetMappedMipmappedArray.restype  = ctypes.c_int
            _cuda_driver.cuGraphicsResourceGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
            _cuda_driver.cuMipmappedArrayGetLevel.restype  = ctypes.c_int
            _cuda_driver.cuMipmappedArrayGetLevel.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
            _cuda_driver.cuMemcpy2D_v2.restype  = ctypes.c_int
            _cuda_driver.cuMemcpy2D_v2.argtypes = [ctypes.c_void_p]
            _cuda_driver.cuStreamSynchronize.restype  = ctypes.c_int
            _cuda_driver.cuStreamSynchronize.argtypes = [ctypes.c_void_p]

            # 毎フレーム使う引数は作り置きして使い回す (Python側の割り当てを削る)
            self._res_arr  = (ctypes.c_void_p * 1)(self._cuda_res.value)
            self._res_addr = ctypes.addressof(self._res_arr)
            self._mip_arr  = ctypes.c_void_p(0)
            self._cuda_arr = ctypes.c_void_p(0)
            self._prev_ctx = ctypes.c_void_p(0)
            self._memcpys = []
            for buf in self._gpu_bufs:
                cp = CUDA_MEMCPY2D()
                cp.srcMemoryType = 3          # CU_MEMORYTYPE_ARRAY
                cp.dstMemoryType = 2          # CU_MEMORYTYPE_DEVICE
                cp.dstDevice     = buf.data_ptr()
                cp.dstPitch      = roi_w * 4
                cp.WidthInBytes  = roi_w * 4
                cp.Height        = roi_h
                self._memcpys.append((cp, ctypes.byref(cp)))

            self._cu_ctx    = cu_ctx
            self._cu_driver = _cuda_driver
            self._interop  = True
            print("[ketsugecam] モード: CUDA interop (CPUノータッチ)")

        except Exception as e:
            print(f"[ketsugecam] CUDA interop 失敗 → CPUフォールバック: {e}")
            self._interop = False

    # ── 単発キャプチャ ────────────────────────────────────────────────────────
    def grab(self, timeout_ms: int = 1) -> Optional[torch.Tensor]:
        """新フレームを1枚取る。timeout_ms 待っても画面が更新されなければ None。

        新フレームが来た時点ですぐ戻るので、timeout_ms を大きくしても遅延は
        増えない (CPUの空回りが減るだけ)。カーソルが動いただけの更新は
        画像が同じなので None を返す。
        """
        r = self._grab(timeout_ms)
        return None if r is None else r[0]

    def grab_info(self, timeout_ms: int = 1
                  ) -> Tuple[Optional[torch.Tensor], Optional[FrameInfo]]:
        """grab() と同じだが (frame, FrameInfo) を返す。新フレームが無ければ (None, None)。"""
        r = self._grab(timeout_ms)
        return (None, None) if r is None else (r[0], r[1])

    def _pick_slot(self) -> Optional[int]:
        """次に書き込む出力バッファ。最新フレームと貸し出し中のものは避ける。"""
        if not self._interop:
            return -1
        n = self._num_buffers
        with self._lock:
            for k in range(1, n + 1):
                i = (self._write_slot + k) % n
                if i != self._latest_slot and self._leases[i] == 0:
                    self._write_slot = i
                    return i
        return None

    def _grab(self, timeout_ms: int = 1
              ) -> Optional[Tuple[torch.Tensor, FrameInfo, int]]:
        res_ptr = ctypes.POINTER(IDXGIResource)()
        try:
            self._dupl.AcquireNextFrame(
                timeout_ms,
                ctypes.byref(self._frame_info),
                ctypes.byref(res_ptr),
            )
        except comtypes.COMError as e:
            code = ctypes.c_int32(e.args[0]).value
            if code == ctypes.c_int32(DXGI_ERROR_WAIT_TIMEOUT).value:
                with self._lock:
                    self._stats.timeouts += 1
                return None
            if code == ctypes.c_int32(DXGI_ERROR_ACCESS_LOST).value:
                with self._lock:
                    self._stats.access_lost += 1
                self._rebuild_dupl()
                return None
            raise
        acquire_qpc = qpc_now()
        acquire_ns  = time.perf_counter_ns()
        present_raw = int(self._frame_info.LastPresentTime)
        accumulated = int(self._frame_info.AccumulatedFrames)

        # LastPresentTime == 0: カーソルだけの更新で画面の中身は前と同じ。
        # 同じ画像で推論すると、その間に来た本物の新フレームが待たされる。
        if present_raw == 0:
            self._dupl.ReleaseFrame()
            with self._lock:
                self._stats.cursor_only += 1
            return None

        slot = self._pick_slot()
        if slot is None:
            # 全バッファが最新 or 貸し出し中。上書きせずこのフレームを捨てる。
            self._dupl.ReleaseFrame()
            with self._lock:
                self._stats.busy_drops += 1
            return None

        try:
            src_tex = res_ptr.QueryInterface(ID3D11Texture2D)
            dst_tex = self._interop_tex if self._interop else self._stage_tex
            self._ctx.CopySubresourceRegion(
                dst_tex, 0, 0, 0, 0,
                src_tex, 0,
                ctypes.byref(self._roi_box),
            )
        finally:
            self._dupl.ReleaseFrame()

        if self._interop:
            frame = self._to_tensor_cuda(slot)
        else:
            self._stage_surf.Map(ctypes.byref(self._rect), _DXGI_MAP_READ)
            try:
                frame = self._to_tensor_cpu()
            finally:
                self._stage_surf.Unmap()
            if frame.is_cuda:
                # 転送完了まで待ってから「完了時刻」を取る
                torch.cuda.current_stream(frame.device).synchronize()

        done_qpc = qpc_now()
        done_ns  = time.perf_counter_ns()
        info = FrameInfo(
            frame_id=self._next_frame_id,
            acquire_ns=acquire_ns,
            acquire_qpc=acquire_qpc,
            capture_done_ns=done_ns,
            capture_done_qpc=done_qpc,
            present_time_raw=present_raw,
            accumulated_frames=accumulated,
            width=self._roi_w,
            height=self._roi_h,
        )
        self._next_frame_id += 1
        self._record_capture(info)
        return frame, info, slot

    def _record_capture(self, info: FrameInfo):
        p2c = info.present_to_capture_ns
        with self._lock:
            st = self._stats
            interval = info.capture_done_ns - st.last_done_ns if st.last_done_ns else None
            st.last_done_ns = info.capture_done_ns
            st.captured += 1
            st.dxgi_skipped += info.skipped_presents
            if interval is not None:
                st.intervals.append(interval)
            if p2c is not None:
                st.p2c.append(p2c)
            st.copy.append(info.copy_ns)
            if not info.clock_consistent:
                st.clock_violations += 1
            # ログもロック下で積む: stop_log() 後に積まれて消える行を作らない
            if self._log_q is not None:
                self._log_q.append((
                    "capture", info.frame_id, info.capture_done_ns, info.capture_done_ns,
                    info.present_time_raw, info.acquire_qpc, info.capture_done_qpc,
                    "" if p2c is None else p2c / 1000,
                    info.accumulated_frames,
                    "" if interval is None else interval / 1000,
                    info.copy_ns / 1000, "", "",
                ))

    def _to_tensor_cuda(self, slot: int) -> torch.Tensor:
        """GPU → GPU 直接転送。CPUノータッチ。Driver API統一。"""
        cu = self._cu_driver
        # 毎フレーム D3D11コンテキストを確実にカレントに設定する。
        # ただし呼び出し元のコンテキストは退避して最後に戻す:
        # torchが明示的に作ったストリームは作成時のコンテキスト専用で、
        # D3D11側のコンテキストがカレントのまま使うと
        # cudaErrorInvalidResourceHandle になる。grab() の直後に
        # 同じスレッドで推論する使い方 (ketsugecam.detector 等) のため。
        prev_ctx = self._prev_ctx
        prev_ctx.value = 0
        cu.cuCtxGetCurrent(ctypes.byref(prev_ctx))
        cu.cuCtxSetCurrent(self._cu_ctx.value)

        res_addr = self._res_addr
        err = cu.cuGraphicsMapResources(1, res_addr, None)
        if err != 0:
            raise RuntimeError(f"cuGraphicsMapResources failed: {err}")
        try:
            err = cu.cuGraphicsResourceGetMappedMipmappedArray(
                ctypes.byref(self._mip_arr), self._cuda_res
            )
            if err != 0:
                raise RuntimeError(f"GetMappedMipmappedArray failed: {err}")

            err = cu.cuMipmappedArrayGetLevel(
                ctypes.byref(self._cuda_arr), self._mip_arr, 0
            )
            if err != 0:
                raise RuntimeError(f"cuMipmappedArrayGetLevel failed: {err}")

            # cuMemcpy2D: cudaArray → GPU buffer (構造体は作り置き)
            cp, cp_ref = self._memcpys[slot]
            cp.srcArray = self._cuda_arr.value
            err = cu.cuMemcpy2D_v2(cp_ref)
            if err != 0:
                raise RuntimeError(f"cuMemcpy2D_v2 failed: {err}")
            # デバイス間コピーはホストと同期しないことがある。利用側は torch の
            # 別コンテキスト/ストリームで読むので、完了を待ってから公開する。
            err = cu.cuStreamSynchronize(None)
            if err != 0:
                raise RuntimeError(f"cuStreamSynchronize failed: {err}")

        finally:
            cu.cuGraphicsUnmapResources(1, res_addr, None)
            if prev_ctx.value and prev_ctx.value != self._cu_ctx.value:
                cu.cuCtxSetCurrent(prev_ctx)

        # BGRA → BGR スライス (ゼロコピー)
        return self._gpu_views[slot]

    def _to_tensor_cpu(self) -> torch.Tensor:
        """CPUフォールバック: Staging Map → torch → .cuda()"""
        pitch  = int(self._rect.Pitch)
        stride = pitch // 4
        size   = pitch * self._roi_h
        buf = (ctypes.c_uint8 * size).from_address(
            ctypes.addressof(self._rect.pBits.contents)
        )
        t = torch.frombuffer(buf, dtype=torch.uint8).reshape(self._roi_h, stride, 4)
        if stride != self._roi_w:
            t = t[:, :self._roi_w, :]
        t = t[..., :3]
        if self.cuda:
            t = t.to("cuda", non_blocking=True)
        else:
            # Map中のメモリのビューなので、Unmap 後も使えるよう複製する
            t = t.clone()
        return t

    # ── 連続キャプチャ ────────────────────────────────────────────────────────
    def start(self, target_fps: Optional[float] = None, wait_ms: int = 50):
        """連続キャプチャを開始する。

        Args:
            target_fps: None (既定) なら AcquireNextFrame で新フレームの到着を
                        待ち、届いた瞬間に取る (ポーリング待ちなし)。
                        数値を渡すと従来どおりタイマーでその周期ごとに取りに行く
                        (GPU負荷を抑えたい時用。最大 1/target_fps 遅れる)。
            wait_ms:    待ちモードで1回に待つ上限。stop() の応答時間になる。
                        新フレームが来ればすぐ戻るので遅延には影響しない。
        """
        if self.is_capturing:
            return
        self.is_capturing = True
        self._stop_event.clear()
        self._frame_event.clear()
        self._thread = Thread(
            target=self._capture_loop,
            args=(target_fps, wait_ms),
            name="KetsugeCam",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        if not self.is_capturing:
            return
        self._stop_event.set()
        self._frame_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_capturing = False
        with self._lock:
            self._latest = None
            self._latest_info = None
            self._latest_slot = -1
        self._frame_event.clear()
        self._stop_event.clear()

    def get_latest_frame(self, timeout: float = 1.0,
                         copy: bool = False) -> Optional[torch.Tensor]:
        """最新フレームをEvent待機で取得。更新が来なければ直前のフレーム。

        Args:
            timeout: 新フレームを待つ秒数
            copy:    True なら専用バッファへコピーして返す。interop時の返り値は
                     リングバッファのビューなので、num_buffers-1 フレーム後には
                     中身が書き換わりうる。数フレームにわたって保持する / 表示用に
                     CPUへ落とすなど、取得後に時間がかかる用途では True にするか、
                     lease_latest_frame() を使う。
        """
        return self.get_latest_frame_info(timeout=timeout, copy=copy)[0]

    def get_latest_frame_info(self, timeout: float = 1.0, copy: bool = False
                              ) -> Tuple[Optional[torch.Tensor], Optional[FrameInfo]]:
        """最新フレームと FrameInfo を一緒に返す。まだ無ければ (None, None)。

            frame, info = cam.get_latest_frame_info()
            if frame is not None and info.is_new:
                ...

        画像とメタデータは同じロックの下で取り出すので、別フレームの組み合わせに
        ならない。info.is_new が False なら前回と同じフレーム (timeout で戻った)。
        copy の意味は get_latest_frame() と同じ。
        """
        frame, info, slot = self._take_latest(timeout, lease=copy)
        if frame is None:
            return None, None
        if copy:
            try:
                frame = frame.clone()
                if frame.is_cuda:
                    torch.cuda.current_stream(frame.device).synchronize()
            finally:
                self._unlease(slot)
        return frame, self._consume(info)

    @contextmanager
    def lease_latest_frame(self, timeout: float = 1.0
                           ) -> Iterator[Tuple[Optional[torch.Tensor], Optional[FrameInfo]]]:
        """最新フレームをコピーせずに貸し出す。with を抜けるまで上書きされない。

            with cam.lease_latest_frame() as (frame, info):
                if frame is not None and info.is_new:
                    boxes = model.predict(frame)

        抜けるときに現在のCUDAストリームを同期してから返却する (貸し出し中に
        積んだGPU処理がバッファを読み終わる前に上書きされないように)。
        長く持つとバッファが足りず busy_drops が増えるので、その場合は copy=True。
        """
        frame, info, slot = self._take_latest(timeout, lease=True)
        try:
            yield frame, (None if info is None else self._consume(info))
        finally:
            if frame is not None:
                try:
                    if frame.is_cuda:
                        torch.cuda.current_stream(frame.device).synchronize()
                finally:
                    self._unlease(slot)

    def _take_latest(self, timeout: float, lease: bool):
        self._frame_event.wait(timeout=timeout)
        self._frame_event.clear()
        with self._lock:
            frame, info, slot = self._latest, self._latest_info, self._latest_slot
            if frame is None:
                return None, None, -1
            if lease and slot >= 0:
                self._leases[slot] += 1
        return frame, info, slot

    def _unlease(self, slot: int):
        if slot >= 0:
            with self._lock:
                self._leases[slot] -= 1

    def _consume(self, info: FrameInfo) -> FrameInfo:
        """利用側に渡す時の統計・ログ。is_new を付けた FrameInfo を返す。"""
        now_ns = time.perf_counter_ns()
        age = now_ns - info.capture_done_ns
        with self._lock:
            st = self._stats
            last = st.last_consumed_id
            is_new = info.frame_id != last
            st.consumed += 1
            if is_new:
                if last and info.frame_id > last + 1:
                    st.consumer_skipped += info.frame_id - last - 1
                st.last_consumed_id = info.frame_id
                st.ages.append(age)
            else:
                st.consumer_repeats += 1
            if self._log_q is not None:
                self._log_q.append((
                    "consume", info.frame_id, now_ns, info.capture_done_ns,
                    info.present_time_raw, "", "", "", "", "", "",
                    age / 1000, int(is_new),
                ))
        return info if is_new else replace(info, is_new=False)

    # ── フレーム統計 ──────────────────────────────────────────────────────────
    def get_stats(self) -> FrameStats:
        """キャプチャ間隔・欠落数・取得頻度・鮮度などのスナップショット。"""
        with self._lock:
            return self._stats.snapshot()

    def reset_stats(self):
        with self._lock:
            self._stats.reset()

    # ── 計測ログ (CSV) ────────────────────────────────────────────────────────
    def start_log(self, path: str, flush_interval: float = 0.25):
        """キャプチャ/取得のたびに1行ずつCSVへ記録する。

        行の追加はキューに積むだけで、ファイル書き込みは別スレッドで行うので
        キャプチャループを遅らせない。列は _LOG_HEADER を参照:
          event="capture": frame_id, capture_done_ns, present_to_capture_us,
                           accumulated_frames, interval_us, copy_us
          event="consume": t_ns (受け取った時刻), age_us, is_new
        *_ns は perf_counter_ns、present_time_raw は QPC ティック (無変換)。
        """
        self.stop_log()
        f = open(path, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(_LOG_HEADER)
        q: deque = deque()
        self._log_stop.clear()

        def _writer():
            try:
                while True:
                    stopping = self._log_stop.wait(flush_interval)
                    while q:
                        writer.writerow(q.popleft())
                    f.flush()
                    if stopping:
                        break
            finally:
                f.close()

        self._log_thread = Thread(target=_writer, name="KetsugeCamLog", daemon=True)
        self._log_thread.start()
        with self._lock:
            self._log_q = q

    def stop_log(self):
        if getattr(self, "_log_q", None) is None:   # __init__ 途中失敗時の __del__ でも安全に
            return
        with self._lock:      # 以後ログ行は積まれない → 書き込みスレッドが全部吐き切る
            self._log_q = None
        self._log_stop.set()
        if self._log_thread:
            self._log_thread.join(timeout=5)
        self._log_thread = None

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    @property
    def mode(self) -> str:
        return "cuda_interop" if self._interop else "cpu_fallback"

    def _capture_loop(self, target_fps: Optional[float], wait_ms: int):
        if self._interop:
            self._cu_driver.cuCtxSetCurrent(self._cu_ctx.value)

        timer = None
        if target_fps:
            timer = _create_timer()
            _set_timer(timer, target_fps)
        count   = 0
        t_start = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                if timer is not None:
                    if _wait_timer(timer) == _WAIT_FAILED:
                        break
                    _set_timer(timer, target_fps)
                    r = self._grab()
                else:
                    # 新フレームが届いた瞬間に戻る (ctypes は待機中 GIL を手放す)
                    r = self._grab(wait_ms)
                if r is not None:
                    frame, info, slot = r
                    with self._lock:
                        self._latest      = frame
                        self._latest_info = info
                        self._latest_slot = slot
                    self._frame_event.set()
                    count += 1
                elapsed = time.perf_counter() - t_start
                if elapsed >= 0.5:
                    self._capture_fps = count / elapsed
                    count   = 0
                    t_start = time.perf_counter()
        finally:
            if timer is not None:
                _cancel_timer(timer)

    # ── Duplicator 再構築 ─────────────────────────────────────────────────────
    def _rebuild_dupl(self):
        time.sleep(0.1)
        try:
            self._dupl.ReleaseFrame()
        except Exception:
            pass
        try:
            self._dupl.Release()
        except Exception:
            pass
        factory  = _create_dxgi_factory()
        adapters = _enum_adapters(factory)
        outputs  = _enum_outputs(adapters[self._device_idx])
        self._dupl = ctypes.POINTER(IDXGIOutputDuplication)()
        outputs[self._output_idx].DuplicateOutput(self._device, ctypes.byref(self._dupl))

    # ── ユーティリティ ────────────────────────────────────────────────────────
    def _validate_region(self, region):
        l, t, r, b = region
        if not (self.width >= r > l >= 0 and self.height >= b > t >= 0):
            raise ValueError(f"Invalid region {region} for {self.width}x{self.height}")

    def release(self):
        self.stop()
        self.stop_log()
        if self._cuda_res is not None:
            try:
                _cudart.cudaGraphicsUnregisterResource(self._cuda_res)
            except Exception:
                pass
        if self._stage_surf is not None:
            try:
                self._stage_surf.Unmap()
            except Exception:
                pass
        if self._stage_tex is not None:
            try:
                self._stage_tex.Release()
            except Exception:
                pass
        try:
            self._dupl.ReleaseFrame()
            self._dupl.Release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        self.release()

    def __repr__(self):
        return (
            f"<KetsugeCam {self.width}x{self.height} "
            f"region={self.region} mode={self.mode}>"
        )