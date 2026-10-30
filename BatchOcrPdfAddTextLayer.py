# -*- coding: utf-8 -*-
r'''
页级 OCR · 透明文字层（PaddleOCR 3.x）—— 批量给扫描型 PDF 补可搜索文字层。

相对旧版（整本 PDF 一次性喂给 ocr.predict）的核心变化：
  1) 页级任务：逐页判断 page.get_text()，已有文字层的页直接跳过；只 OCR 缺文字层的页。
     失败只重试单页（下次运行时该页仍无文字 -> 自动只重试它），不再整文件重跑。
  2) 页级缓存（SQLite，默认 .ocr_page_cache.db）：识别结果按 文件(路径+mtime+size)+页号+DPI+档位
     缓存；中断/崩溃后续跑时命中页零推理直接复用（写文字层本身很快）。
  3) GPU 优先：启动探测 nvidia-smi + paddle 是否 CUDA 版；可用则 device="gpu"。
     同时把 site-packages/nvidia/*/bin（pip 装的 CUDA/cuDNN DLL）注入 DLL 搜索路径
     （paddlepaddle-gpu win 轮子不带这些 DLL，缺了会报 cudnn64_9.dll error 126）。
     无 GPU 走 CPU 优化：OMP/MKL/OPENBLAS=1、cpu_threads=1~2、enable_mkldnn=True。
  4) DPI 默认 200；页均置信度低于 OCR_SCORE_MIN(默认 0.80) 时该页升 300 DPI + server 模型重试一次
     （二者取优）。空白页（非白像素占比 < OCR_BLANK_MAX）跳过。
  5) 参数：use_textline_orientation=False(=use_angle_cls False)、text_recognition_batch_size=16、
     text_det_limit_side_len=960、enable_mkldnn=True(CPU)；模型优先 PP-OCRv4_mobile，升档 server。
  6) 资源监控日志：每文件输出 CPU/内存/GPU/磁盘 + 页数/缓存命中/升档/失败重试计数。

环境变量（全部可选）：
  OCR_DEVICE=auto|gpu|cpu   OCR_THREADS=1|2   OCR_DPI=200  OCR_DPI_BAD=300
  OCR_SCORE_MIN=0.80        OCR_BLANK_MAX=0.001  OCR_CACHE_DB=<path|0>
  OCR_MAX_PAGES=0(全部)     OCR_DET_LIMIT=960  OCR_REC_BATCH=16
  OCR_DET_MODEL/OCR_REC_MODEL(基础档覆盖)   OCR_CENTER_SCALE=(默认1.0，旧版0.95可显式指定)
  OCR_DISABLE_AUX_MODELS=0(设为0才启用辅助模型)  OCR_ENGINE_KWARGS={json}  优先级最高
  OCR_MONITOR=0(关监控)     OCR_VERBOSE=1(每页一行明细)

兼容保留：check_pdf_ocr_add_text / read_pdf_page0 / scale_point / get_ocr_engine 旧签名，
pipeline 子进程与 wm_ocr_batch_worker 不受影响；本模块也可 CLI 单跑（见文件尾 usage）。
'''

import sys
import os
import json
import time
import shutil
import hashlib
import sqlite3
import subprocess
import warnings
import math
import re
from datetime import datetime

# Windows 控制台默认 GBK：emoji 等字符无法编码会导致 print 抛 UnicodeEncodeError 崩溃整个流程。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)

_NO_WIN = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW：探测子进程不弹黑窗


def _env(name, default=""):
    v = os.environ.get(name)
    return (v if v is not None else default).strip()


def _env_int(name, default, lo=1, hi=10**9):
    try:
        return max(lo, min(hi, int(float(_env(name, str(default))))))
    except (ValueError, TypeError):
        return default


def _env_float(name, default):
    try:
        return float(_env(name, str(default)))
    except (ValueError, TypeError):
        return default


# ==================== 一、GPU 探测与线程钉死（必须在 import paddle 之前执行） ====================
def _probe_nvidia():
    """探测 NVIDIA 显卡（nvidia-smi）。返回 dict；任何失败都不抛异常。"""
    info = {"has_nvidia": False, "smi": "", "name": "", "mem_mb": 0,
            "driver": "", "note": ""}
    smi = shutil.which("nvidia-smi")
    if not smi:
        info["note"] = "未找到 nvidia-smi（无 NVIDIA 驱动/显卡）"
        return info
    try:
        r = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, creationflags=_NO_WIN)
        if r.returncode != 0:
            info["note"] = f"nvidia-smi 返回码 {r.returncode}"
            return info
        line = (r.stdout or "").strip().splitlines()[0]
        name, mem, drv = [p.strip() for p in line.split(",")]
        info.update(has_nvidia=True, smi=smi, name=name,
                    mem_mb=int(float(mem)), driver=drv)
    except Exception as e:
        info["note"] = f"nvidia-smi 探测失败: {e}"
    return info


_GPU_INFO = _probe_nvidia()
_OCR_DEVICE_WANT = (_env("OCR_DEVICE", "auto") or "auto").lower()

# CPU 模式线程钉死：paddle/opencv 读 OMP/MKL/OPENBLAS 决定线程数，必须在 import paddle 前写死。
# 不钉死 = 单进程默认吃满 32 逻辑核，N 个 worker 叠加 = N×32 线程抢 16 物理核（旧版超订根因）。
_pin = None
if _OCR_DEVICE_WANT == "cpu" or not _GPU_INFO["has_nvidia"]:
    _t = _env("OCR_THREADS", "1")
    _pin = _t if _t in ("1", "2") else "1"
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _pin
_CPU_THREADS = int(_pin) if _pin else _env_int("OCR_THREADS", 1, 1, 2)


def _inject_nvidia_dlls():
    """把 site-packages/nvidia/<pkg>/bin 注入 DLL 搜索路径（必须在 import paddle 之前）。
    paddlepaddle-gpu 的 win 轮子不自带 CUDA/cuDNN DLL，需要 nvidia-cublas-cu12 /
    nvidia-cudnn-cu12 / nvidia-cuda-runtime-cu12 等 pip 包提供，否则报
    "cudnn64_9.dll ... not configured correctly (error code 126)"。
    仅当检测到 N 卡且未强制 CPU 时注入；任何失败静默（CPU 路径不受影响）。"""
    if not _GPU_INFO["has_nvidia"] or _OCR_DEVICE_WANT == "cpu":
        return []
    if not hasattr(os, "add_dll_directory"):      # 非 Windows 跳过
        return []
    added = []
    try:
        for base in list(sys.path):
            if not base:
                continue
            root = os.path.join(base, "nvidia")
            if not os.path.isdir(root):
                continue
            for pkg in sorted(os.listdir(root)):
                d = os.path.join(root, pkg, "bin")
                if not os.path.isdir(d):
                    continue
                try:
                    os.add_dll_directory(d)       # 让 LoadLibrary 依赖解析能找到
                except Exception:
                    pass
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                added.append(d)
    except Exception:
        pass
    return added


_NVIDIA_DLL_DIRS = _inject_nvidia_dlls()

import fitz                      # pymupdf
import numpy as np
from paddleocr import PaddleOCR

# 添加当前目录到 Python 路径 + 应用中文路径修复补丁（历史遗留，保留）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    from simple_encoding_fix import create_simple_encoding_fix_patch
    if create_simple_encoding_fix_patch():
        print("[OK] 中文路径修复补丁应用成功")
    else:
        print("[WARN] 中文路径修复补丁应用失败，继续执行")
except ImportError as e:
    print(f"[WARN] 无法导入中文路径修复模块: {e}")


# ==================== 二、运行参数（环境变量可调） ====================
OCR_DPI = _env_int("OCR_DPI", 200, 72, 600)
OCR_DPI_BAD = max(OCR_DPI, _env_int("OCR_DPI_BAD", 300, 72, 600))
OCR_SCORE_MIN = _env_float("OCR_SCORE_MIN", 0.80)
OCR_BLANK_MAX = _env_float("OCR_BLANK_MAX", 0.001)
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 0, 0, 10**9)
_CENTER_SCALE = _env_float("OCR_CENTER_SCALE", 1.0) or 1.0
_VERBOSE = _env("OCR_VERBOSE") in ("1", "true", "yes")
_MONITOR = _env("OCR_MONITOR", "1") not in ("0", "false", "no")
_CACHE_PATH = _env("OCR_CACHE_DB", os.path.join(current_dir, ".ocr_page_cache.db"))
if _CACHE_PATH.lower() in ("0", "off", "false", "none"):
    _CACHE_PATH = ""
_DET_LIMIT = _env_int("OCR_DET_LIMIT", 960, 128, 4096)
_REC_BATCH = _env_int("OCR_REC_BATCH", 16, 1, 128)


# ==================== 二·五、显存护栏（A2 看门狗 + A3 CUDA OOM 熔断） ====================
class _CudaOOM(Exception):
    """CUDA 显存不足自定义异常：在 _ocr_page_on 捕获原始异常后转抛，便于页级主流程区分处理。"""


_VRAM_GUARD = {"ts": 0.0, "total": 0, "used": 0, "has": False}


def vram_guard_tripped():
    """显存看门狗（懒采样 + 20s 缓存，正常跑几乎零开销）：used/total >= 90% 返回 True。
    无 GPU 或关监控（OCR_MONITOR=0）时返回 False。worker 每处理完一个文件查一次，命中即
    跳过本批剩余文件并标记 skip(vram)，交父进程降并发重派（A2 自适应降级）。"""
    if not _MONITOR or not _GPU_INFO.get("has_nvidia"):
        return False
    _now = time.time()
    if _now - _VRAM_GUARD["ts"] > 20:
        try:
            _r = subprocess.run(
                [_GPU_INFO["smi"], "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=6, creationflags=_NO_WIN)
            if _r.returncode == 0:
                _p = [x.strip() for x in (_r.stdout or "").strip().splitlines()[0].split(",")]
                _VRAM_GUARD["used"], _VRAM_GUARD["total"] = int(float(_p[0])), int(float(_p[1]))
                _VRAM_GUARD["has"] = True
        except Exception:
            pass
        _VRAM_GUARD["ts"] = _now
    _g = _VRAM_GUARD
    if _g["has"] and _g["total"] > 0:
        return (_g["used"] / _g["total"]) >= 0.90
    return False



def scale_point(x, y, page_width, page_height, scalex, scaley):
    """绕页面中心点缩放坐标（兼容保留）。"""
    cx = page_width / 2
    cy = page_height / 2
    return (x - cx) * scalex + cx, (y - cy) * scaley + cy


# ==================== 三、设备决策（import paddle 之后才能确认是否 CUDA 版） ====================
_DEVICE_STATE = {"device": None, "reason": ""}


def device_decision():
    """最终设备决策（只算一次）：auto = 有 N 卡 且 paddle 是 CUDA 版 且 device_count>0 -> gpu。"""
    if _DEVICE_STATE["device"]:
        return _DEVICE_STATE
    cuda_compiled, cuda_count = False, 0
    try:
        import paddle
        cuda_compiled = bool(paddle.device.is_compiled_with_cuda())
        if cuda_compiled:
            try:
                cuda_count = int(paddle.device.cuda.device_count())
            except Exception:
                cuda_count = 0
    except Exception:
        pass
    if _OCR_DEVICE_WANT == "cpu":
        dev, why = "cpu", "OCR_DEVICE=cpu 强制"
    elif not _GPU_INFO["has_nvidia"]:
        dev, why = "cpu", _GPU_INFO.get("note") or "无 NVIDIA 显卡"
    elif not cuda_compiled:
        dev, why = "cpu", ("检测到 NVIDIA 显卡但 paddle 是 CPU 版（未启用 GPU；"
                           "安装 paddlepaddle-gpu 后自动走 GPU，见 requirements.txt）")
    elif cuda_count <= 0:
        dev, why = "cpu", "paddle CUDA 版可用但 device_count=0（驱动/显存异常）"
    else:
        dev, why = "gpu", f"{_GPU_INFO['name']}（{_GPU_INFO['mem_mb']}MB，驱动 {_GPU_INFO['driver']}）"
    _DEVICE_STATE.update(device=dev, reason=why)
    return _DEVICE_STATE


# ==================== 四、引擎（mobile 基础档 / server 升档，按档单例） ====================
# 档位 0=基础（PP-OCRv4_mobile），档位 1=升档（PP-OCRv4_server，低置信度页用）。
# 构造失败自动降级：v4 -> v5 -> PaddleOCR 默认参数；GPU 全失败自动回退 CPU。
_TIER_MODELS = {
    0: [("PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec", "v4_mobile"),
        ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec", "v5_mobile"),
        (None, None, "default")],
    1: [("PP-OCRv4_server_det", "PP-OCRv4_server_rec", "v4_server"),
        ("PP-OCRv5_server_det", "PP-OCRv5_server_rec", "v5_server"),
        (None, None, "default")],
}
_ENGINES = {}          # tier -> (engine, label)


def _aux_disabled():
    return _env("OCR_DISABLE_AUX_MODELS", "1") not in ("0", "false", "no")


def _engine_kwargs(det_name, rec_name):
    """组装 PaddleOCR(**) 参数（3.x 命名；use_angle_cls=False 对应 use_textline_orientation=False）。"""
    dev = device_decision()["device"]
    kw = {"device": dev}
    if _aux_disabled():
        kw.update(
            use_doc_orientation_classify=False,   # 文档方向分类
            use_doc_unwarping=False,              # UVDoc 去扭曲（最重）
            use_textline_orientation=False,       # 文本行方向分类（= 旧 use_angle_cls=False）
        )
    kw["text_det_limit_side_len"] = _DET_LIMIT
    kw["text_recognition_batch_size"] = _REC_BATCH
    if det_name and rec_name:
        kw["text_detection_model_name"] = det_name
        kw["text_recognition_model_name"] = rec_name
    if dev == "cpu":
        # CPU 优化：线程预算 + MKL-DNN；GPU 下这两个参数无意义（不传）。
        kw["cpu_threads"] = _CPU_THREADS
        kw["enable_mkldnn"] = True
    extra = _env("OCR_ENGINE_KWARGS")
    if extra:
        try:
            kw.update(json.loads(extra))
        except Exception as e:
            print(f"[WARN] OCR_ENGINE_KWARGS 解析失败，忽略: {e}")
    return kw


def _with_fp16(kw):
    """若请求 OCR_FP16 且为 GPU：根据 PaddleOCR 构造签名注入其支持的半精度参数。
    不同版本 fp16 参数名不一（fp16 / precision），用 inspect 判断存在性，不支持则保持默认精度。
    （不试建引擎，避免重复加载模型/额外占显存。）"""
    if _env("OCR_FP16").lower() not in ("1", "true", "yes"):
        return kw
    if kw.get("device") != "gpu":
        print("[WARN] fp16 未生效，原因：当前为 CPU 设备，半精度仅 GPU 支持，已忽略 OCR_FP16", flush=True)
        return kw
    try:
        import inspect
        _params = set(inspect.signature(PaddleOCR.__init__).parameters)
    except (TypeError, ValueError):
        _params = set()
    if "precision" in _params:
        kw["precision"] = "fp16"
        print("[INFO] OCR_FP16 生效：precision='fp16'", flush=True)
    elif "fp16" in _params:
        kw["fp16"] = True
        print("[INFO] OCR_FP16 生效：fp16=True", flush=True)
    else:
        print("[WARN] fp16 未生效，原因：当前 PaddleOCR 构造不支持 fp16/precision 参数（模型/版本不支持），已忽略 OCR_FP16、保持默认 fp32 精度", flush=True)
    return kw


def get_ocr_engine(tier=0):
    """获取 OCR 引擎（按档位单例）。tier=0 基础档，tier=1 server 升档。兼容旧签名 get_ocr_engine()。"""
    ent = _ENGINES.get(tier)
    if ent is not None:
        return ent[0]
    tries = list(_TIER_MODELS.get(tier) or _TIER_MODELS[0])
    if tier == 0 and (_env("OCR_DET_MODEL") or _env("OCR_REC_MODEL")):
        tries.insert(0, (_env("OCR_DET_MODEL") or None, _env("OCR_REC_MODEL") or None,
                         "env_override"))
    last_err = None
    for det, rec, label in tries:
        try:
            kw = _engine_kwargs(det, rec)
            print(f"[INFO] PaddleOCR 引擎初始化 tier={tier} label={label} "
                  f"device={kw.get('device')} det={det or '默认'} rec={rec or '默认'}",
                  flush=True)
            eng = PaddleOCR(**_with_fp16(kw))
            _ENGINES[tier] = (eng, label)
            _log_gpu_mem_policy()
            return eng
        except Exception as e:
            last_err = e
            print(f"[WARN] 引擎 tier={tier} label={label} 初始化失败: {type(e).__name__}: {e}",
                  flush=True)
    if device_decision()["device"] == "gpu":
        print("[WARN] GPU 引擎全部初始化失败，自动回退 CPU 重试", flush=True)
        _DEVICE_STATE.update(device="cpu", reason="GPU 初始化失败自动回退")
        _ENGINES.pop(tier, None)
        return get_ocr_engine(tier)
    raise RuntimeError(f"PaddleOCR 引擎不可用（tier={tier}）：{last_err}")


_MEM_POLICY_LOGGED = False


def _log_gpu_mem_policy():
    """GPU 引擎初始化后打印一次「本次 Paddle 显存策略」自证：环境变量实际值 + 真实占用。
    FLAGS_fraction_of_gpu_memory_to_use=0 语义=不按比例预占、按需增长；若打印显示
    reserved 初始化即占满整卡显存，则该策略实际无效（应移除设置避免误导）。"""
    global _MEM_POLICY_LOGGED
    if _MEM_POLICY_LOGGED or device_decision().get("device") != "gpu":
        return
    _MEM_POLICY_LOGGED = True
    try:
        _fr = os.environ.get("FLAGS_fraction_of_gpu_memory_to_use", "(未设置)")
        _ac = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(未设置)")
        _alloc = _res = None
        try:
            import paddle
            _alloc = paddle.device.cuda.memory_allocated() / 2 ** 20
            _res = paddle.device.cuda.memory_reserved() / 2 ** 20
        except Exception:
            pass
        _tail = (f"；引擎初始化后实测 allocated≈{_alloc:.0f}MB reserved≈{_res:.0f}MB"
                 f"（按需增长策略生效中，未整卡预占）" if _alloc is not None else "")
        print(f"[mem] 本次 Paddle 显存策略：FLAGS_fraction_of_gpu_memory_to_use={_fr}"
              f"（0=不预占、按需增长），PYTORCH_CUDA_ALLOC_CONF={_ac}{_tail}", flush=True)
    except Exception:
        pass


def engine_label(tier):
    """返回档位实际生效的模型标签（引擎未创建时返回预估值，仅用于缓存键）。"""
    ent = _ENGINES.get(tier)
    if ent is not None:
        return ent[1]
    return (_TIER_MODELS.get(tier) or _TIER_MODELS[0])[0][2]


# ==================== 五、页级缓存（SQLite，多进程 WAL 安全） ====================
_CACHE_CONN = None
_CACHE_HIT = 0


def _cache_conn():
    """缓存连接（单例）。任何异常返回 None -> 调用方静默跳过缓存。"""
    global _CACHE_CONN
    try:
        if _CACHE_CONN is not None:
            return _CACHE_CONN
        if not _CACHE_PATH:
            return None
        conn = sqlite3.connect(_CACHE_PATH, timeout=8)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=8000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS page_ocr("
            " key TEXT PRIMARY KEY, pdf TEXT, page INTEGER, dpi INTEGER,"
            " tier TEXT, n_chars INTEGER, avg_score REAL, txt_json TEXT,"
            " created_at TEXT)")
        conn.commit()
        _CACHE_CONN = conn
        return conn
    except Exception:
        _CACHE_CONN = None
        return None


_FILE_HASH_MEMO = {}     # (abspath, mtime_ns, size) -> sha256(文件字节)：逐页调用只算一次


def _file_content_hash(pdf_file):
    """文件内容 sha256（十六进制）。缓存键用它而非路径，重命名/移动后内容不变 -> 缓存仍命中。
    按 (路径+mtime_ns+size) 记忆，同一文件多页只哈希一次；任何异常回退到路径签名（不崩主流程）。"""
    p = os.path.abspath(pdf_file)
    try:
        stt = os.stat(p)
        sig = (p, stt.st_mtime_ns, stt.st_size)
    except OSError:
        return "nostat:" + os.path.normcase(p)
    h = _FILE_HASH_MEMO.get(sig)
    if h is not None:
        return h
    try:
        hs = hashlib.sha256()
        with open(p, "rb") as f:
            for _chunk in iter(lambda: f.read(1 << 20), b""):
                hs.update(_chunk)
        h = hs.hexdigest()
    except OSError:
        h = "nostat:" + os.path.normcase(p)
    _FILE_HASH_MEMO[sig] = h
    return h


def _cache_key(pdf_file, page_idx, dpi, tier):
    """缓存键 = 文件内容 sha256 + 页号 + DPI + 档位 + 模型环境覆盖。
    用内容哈希而非路径，所以重命名/移动（内容不变）不再使缓存失效；文件真被修改（内容变）自动失效。"""
    raw = "|".join((_file_content_hash(pdf_file), str(page_idx), str(dpi), str(tier),
                    _env("OCR_DET_MODEL"), _env("OCR_REC_MODEL")))
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()


def _cache_get(key):
    """命中返回 (texts, scores, polys, avg)；未命中/异常返回 None。"""
    global _CACHE_HIT
    conn = _cache_conn()
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT txt_json FROM page_ocr WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        _CACHE_HIT += 1
        return (list(d.get("texts") or []), [float(x) for x in (d.get("scores") or [])],
                [list(p) for p in (d.get("polys") or [])], float(d.get("avg") or 0.0))
    except Exception:
        return None


def _cache_put(key, pdf_file, page_idx, dpi, tier, texts, scores, polys, avg):
    """写缓存；失败静默（缓存绝不能影响 OCR 主流程）。"""
    conn = _cache_conn()
    if conn is None:
        return
    try:
        payload = json.dumps({"texts": texts, "scores": scores, "polys": polys, "avg": avg},
                             ensure_ascii=False)
        conn.execute(
            "INSERT OR REPLACE INTO page_ocr(key, pdf, page, dpi, tier, n_chars,"
            " avg_score, txt_json, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (key, os.path.abspath(pdf_file), int(page_idx), int(dpi), str(tier),
             sum(len(str(t)) for t in texts), float(avg), payload,
             datetime.now().isoformat(timespec="seconds")))
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


# ==================== 六、页级渲染 / 判层 / 识别 ====================
_ARTS = re.compile(r"\{#\{.*?\}#\}")   # 矢量水印在文字层留下的伪影标记


def _page_has_text(page):
    """该页是否已有真实文字层：get_text 去伪影后仍有实质字符 -> True。"""
    try:
        t = page.get_text() or ""
    except Exception:
        return False
    t = _ARTS.sub("", t)
    return bool(re.sub(r"\s+", "", t))


def _render_page(page, dpi):
    """渲染成 BGR ndarray（PaddleOCR 3.x 对 ndarray 按 BGR 处理）+ (w,h) 像素尺寸。"""
    z = dpi / 72.0
    pm = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    w, h, n = pm.width, pm.height, pm.n
    buf = np.frombuffer(pm.samples, dtype=np.uint8)
    stride = pm.stride
    if stride != w * n:      # 防 stride 行对齐差异
        buf = buf.reshape(h, stride)[:, :w * n]
    arr = buf.reshape(h, w, n)[:, :, :3]
    bgr = np.ascontiguousarray(arr[:, :, ::-1])   # RGB -> BGR
    return bgr, (w, h)


def _is_blank(bgr):
    """空白页判定：抽稀采样后非白（任一通道 < 235）像素占比 < OCR_BLANK_MAX。"""
    try:
        s = bgr[::2, ::2]
        nonwhite = float((s.min(axis=2) < 235).mean())
        return nonwhite < OCR_BLANK_MAX
    except Exception:
        return False


def _ocr_page_on(engine, bgr):
    """对单页图像跑识别。返回 (texts, scores, polys, avg_score)。
    A3: 捕获 CUDA 显存不足异常，转抛 _CudaOOM，交页级主流程熔断降级（置 st["vram_oom"]）。"""
    try:
        res = engine.predict(bgr)
    except Exception as _e:
        _msg = str(_e).lower()
        import re as _re
        if _re.search(r"out[\s_]*of[\s_]*memory", _msg):
            raise _CudaOOM(str(_e)) from _e
        raise
    r0 = res[0] if isinstance(res, (list, tuple)) and res else res
    texts = [str(t) for t in (r0.get("rec_texts") or [])]
    scores = [float(s) for s in (r0.get("rec_scores") or [])]
    polys = []
    for p in (r0.get("rec_polys") or []):
        try:
            polys.append(np.asarray(p).reshape(-1, 2).tolist())
        except Exception:
            polys.append([])
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return texts, scores, polys, avg


# ==================== 七、文字层写入（阅读序拓扑重构 + 微积分仿射对齐） ====================
_READING_ORDER = _env_int("OCR_READING_ORDER", 1, 0, 1)


def recursive_xy_cut(boxes):
    """Recursive XY-Cut 版面分析与阅读序拓扑重构算法：
    输入 boxes: 每个元素为 dict，包含 'xmin', 'ymin', 'xmax', 'ymax', 'item'
    通过递归切分水平空白缝隙（段落切分）与垂直空白缝隙（双栏/多栏分栏），重构人类自然阅读流。"""
    if len(boxes) <= 1:
        return boxes

    # 1. 尝试水平切分 (Y-cuts: 水平空白带，把页面分成上下独立的段落/块)
    boxes_by_y = sorted(boxes, key=lambda b: (b['ymin'], b['xmin']))
    y_cuts = []
    current_max_y = boxes_by_y[0]['ymax']
    current_group = [boxes_by_y[0]]

    for b in boxes_by_y[1:]:
        # 允许 2.0 像素容差避免 OCR 浮点抖动
        if b['ymin'] >= current_max_y - 2.0:
            y_cuts.append(current_group)
            current_group = [b]
            current_max_y = b['ymax']
        else:
            current_group.append(b)
            if b['ymax'] > current_max_y:
                current_max_y = b['ymax']
    y_cuts.append(current_group)

    if len(y_cuts) > 1:
        result = []
        for grp in y_cuts:
            result.extend(recursive_xy_cut(grp))
        return result

    # 2. 如果无法水平切分，尝试垂直切分 (X-cuts: 垂直空白带，将当前块拆分为左右栏)
    boxes_by_x = sorted(boxes, key=lambda b: (b['xmin'], b['ymin']))
    x_cuts = []
    current_max_x = boxes_by_x[0]['xmax']
    current_group = [boxes_by_x[0]]

    for b in boxes_by_x[1:]:
        if b['xmin'] >= current_max_x - 2.0:
            x_cuts.append(current_group)
            current_group = [b]
            current_max_x = b['xmax']
        else:
            current_group.append(b)
            if b['xmax'] > current_max_x:
                current_max_x = b['xmax']
    x_cuts.append(current_group)

    if len(x_cuts) > 1:
        result = []
        for grp in x_cuts:
            result.extend(recursive_xy_cut(grp))
        return result

    # 3. 既无全局水平间隙又无垂直间隙：按自然阅读行排布（平均行高自适应容差分组）
    avg_h = sum(b['ymax'] - b['ymin'] for b in boxes) / max(1, len(boxes))
    tolerance = max(3.0, avg_h * 0.45)

    sorted_y = sorted(boxes, key=lambda b: b['ymin'])
    lines_g = []
    for b in sorted_y:
        placed = False
        for line in lines_g:
            if abs(b['ymin'] - line[0]['ymin']) < tolerance:
                line.append(b)
                placed = True
                break
        if not placed:
            lines_g.append([b])

    final_ordered = []
    for line in lines_g:
        line.sort(key=lambda b: b['xmin'])
        # 同行多个文本块之间打标 append_space，使复制时单元格/并列项间自然分隔
        for idx_in_line, item_box in enumerate(line):
            if idx_in_line < len(line) - 1:
                item_box['append_space'] = True
        final_ordered.extend(line)

    return final_ordered


def _insert_page_text(page, texts, scores, polys, dpi):
    """把 OCR 文本写成高保真透明文字层（PDF Text Rendering Mode 3，不可见、可搜索、精准划词）。
    1. 阅读序拓扑重构：使用 Recursive XY-Cut 解决试卷双栏/多栏复制左右穿插问题。
    2. 微积分级对齐：基于实际目标宽和字体测量宽度计算水平缩放比率 sx 与仿射变换矩阵 morph，
       彻底消灭暴力缩小字号试错循环，实现字级无缝贴合。
    3. 进阶抗畸变与排版优化：
       - 中西文混合自适应基线下沉（中文字 0.83，西文为主 0.78）；
       - 填空题下划线与大空格拉伸畸变保护；
       - 同行相邻单元格自动添加安全空格分词。"""
    k = 72.0 / float(dpi)
    W, H = page.rect.width, page.rect.height

    # 1. 坐标转换与有效性预处理
    valid_items = []
    for txt, sc, poly in zip(texts, scores, polys):
        if not str(txt).strip() or len(poly) < 4:
            continue
        try:
            pts = [(float(x) * k, float(y) * k) for x, y in poly]
            if abs(_CENTER_SCALE - 1.0) > 1e-6:
                pts = [scale_point(x, y, W, H, _CENTER_SCALE, _CENTER_SCALE)
                       for x, y in pts]
            quad = fitz.Quad(pts[0], pts[1], pts[3], pts[2])
            width, height = quad.width, quad.height
            if width <= 0 or height <= 0:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            valid_items.append({
                'xmin': min(xs), 'xmax': max(xs),
                'ymin': min(ys), 'ymax': max(ys),
                'txt': txt, 'sc': sc, 'pts': pts, 'quad': quad,
                'width': width, 'height': height
            })
        except Exception:
            continue

    if not valid_items:
        return 0, 0

    # 2. 阅读序重构排序
    if _READING_ORDER:
        try:
            ordered_items = recursive_xy_cut(valid_items)
        except Exception:
            ordered_items = valid_items
    else:
        ordered_items = valid_items

    # 3. 高保真仿射写入
    n_ok = n_bad = 0
    for it in ordered_items:
        txt = it['txt']
        quad = it['quad']
        width = it['width']
        height = it['height']

        # 若同行右侧还有相邻块，末尾补单空格保证分词清晰
        if it.get('append_space') and not txt.endswith((' ', '	')):
            txt_to_insert = txt + ' '
        else:
            txt_to_insert = txt

        try:
            # 竖排文字（细高框）特殊处理
            if len(txt) > 1 > width / height:
                rect = fitz.Rect(quad.ur.x, quad.ur.y, quad.ur.x + height, quad.ur.y + width)
                d = quad.ll - quad.ur
                angle_deg = math.degrees(math.atan2(d.y, d.x))
                morph = (rect.tl, fitz.Matrix(1, 1).prerotate(-angle_deg))
                fontsize = max(2.0, width)
                rc = page.insert_textbox(rect, txt_to_insert, fontname="china-ss",
                                         fontsize=fontsize, render_mode=3, morph=morph)
                if rc >= 0:
                    n_ok += 1
                else:
                    n_bad += 1
            else:
                # 水平/倾斜文字：计算旋转角度与仿射水平拉伸矩阵 sx
                d = quad.ur - quad.ul
                angle_deg = math.degrees(math.atan2(d.y, d.x))
                fontsize = max(2.0, height * 0.85)

                # 计算字体度量宽度并求取水平缩放比例 sx
                try:
                    base_len = fitz.get_text_length(txt_to_insert, fontname="china-ss", fontsize=fontsize)
                except Exception:
                    base_len = 0.0

                # base_len 极小（字体无法度量/纯符号）时跳过仿射，走 insert_textbox 兜底
                if base_len < 1.0:
                    raise ValueError("base_len degenerate")
                raw_sx = width / base_len

                # 填空题下划线 / 大空格畸变保护
                has_large_gap = ("____" in txt) or ("    " in txt)
                if has_large_gap:
                    sx = max(0.7, min(1.3, raw_sx))
                else:
                    sx = max(0.1, min(3.5, raw_sx))

                # 自适应中西文基线下沉比例
                ascii_count = sum(1 for c in txt if ord(c) < 128)
                is_mostly_ascii = (ascii_count / max(1, len(txt))) > 0.6
                baseline_ratio = 0.78 if is_mostly_ascii else 0.83

                # 基线原点设在 quad.ul 下方 height * baseline_ratio 处
                origin = fitz.Point(quad.ul.x, quad.ul.y + height * baseline_ratio)
                mat = fitz.Matrix(sx, 1.0).prerotate(-angle_deg)
                morph = (origin, mat)

                # PDF Text Rendering Mode 3: 隐形文本层（无描边无填充、高保真可选、无灰斑）
                rc = page.insert_text(origin, txt_to_insert, fontname="china-ss",
                                      fontsize=fontsize, render_mode=3, morph=morph)
                if rc >= 0:
                    n_ok += 1
                else:
                    # 极端情况兜底
                    rect = fitz.Rect(quad.ul, quad.ul.x + width, quad.ul.y + height)
                    rc = page.insert_textbox(rect, txt_to_insert, fontname="china-ss",
                                             fontsize=fontsize, render_mode=3, morph=morph)
                    if rc >= 0:
                        n_ok += 1
                    else:
                        n_bad += 1
        except Exception:
            n_bad += 1

    return n_ok, n_bad


def _monitor_snapshot(sample_gpu=True):
    """资源快照：CPU%/内存/磁盘/GPU 利用率与显存。任何一项失败都跳过，绝不抛异常。"""
    d = {}
    try:
        import psutil
        d["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        vm = psutil.virtual_memory()
        d["mem_used_gb"] = round((vm.total - vm.available) / 1024**3, 2)
        d["mem_total_gb"] = round(vm.total / 1024**3, 1)
    except Exception:
        pass
    try:
        drive = os.path.splitdrive(os.path.abspath(current_dir))[0] + os.sep
        du = shutil.disk_usage(drive)
        d["disk_free_gb"] = round(du.free / 1024**3, 1)
    except Exception:
        pass
    if sample_gpu and _GPU_INFO.get("has_nvidia"):
        try:
            r = subprocess.run(
                [_GPU_INFO["smi"], "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=6, creationflags=_NO_WIN)
            if r.returncode == 0:
                p = [x.strip() for x in (r.stdout or "").strip().splitlines()[0].split(",")]
                d["gpu_util_pct"] = int(float(p[0]))
                d["gpu_mem_used_mb"] = int(float(p[1]))
                d["gpu_mem_total_mb"] = int(float(p[2]))
        except Exception:
            pass
    return d


def ocr_pdf_with_textlayer(pdf_file):
    """页级 OCR 主流程：逐页跳过(已有文字层/空白)，缓存命中零推理，低置信度页升档(300DPI+server)
    重试一次取优；失败只影响单页（下次续跑自动只重试该页）。返回统计 dict。"""
    t_file0 = time.time()
    dev = device_decision()["device"]
    st = {"file": pdf_file, "status": "fail", "err": None, "device": dev,
          "dpi": OCR_DPI, "pages": 0, "text_pages": 0, "blank_pages": 0,
          "ocr_pages": 0, "cached_pages": 0, "esc_pages": 0, "empty_pages": 0,
          "fail_pages": 0, "fail_list": [], "inserted": 0, "insert_fail": 0,
          "page_sec_avg": 0.0, "page_sec_max": 0.0, "cache_hit_rate": 0.0,
          "engine0": "", "engine1": "", "monitor": {},
          "vram_oom": False, "vram_guard": False}
    t_pages = []
    defer_replace = None
    try:
        doc = fitz.open(pdf_file)
    except Exception as e:
        st["err"] = f"打开失败: {e}"
        st["sec"] = round(time.time() - t_file0, 1)
        print(f"[ERR] 打开失败 {os.path.basename(pdf_file)}: {e}", flush=True)
        return st
    try:
        n = doc.page_count
        limit = min(n, OCR_MAX_PAGES) if OCR_MAX_PAGES > 0 else n
        st["pages"] = n
        for i in range(limit):
            t_p0 = time.perf_counter()
            page = doc[i]
            used_dpi = OCR_DPI
            tier_used = 0
            try:
                if _page_has_text(page):
                    st["text_pages"] += 1
                    continue
                bgr, _wh = _render_page(page, OCR_DPI)
                if _is_blank(bgr):
                    st["blank_pages"] += 1
                    continue
                key = _cache_key(pdf_file, i, OCR_DPI, engine_label(0))
                hit = _cache_get(key)
                if hit is not None:
                    texts, scores, polys, avg = hit
                    st["cached_pages"] += 1
                else:
                    eng0 = get_ocr_engine(0)
                    t_i0 = time.perf_counter()
                    texts, scores, polys, avg = _ocr_page_on(eng0, bgr)
                    dt0 = time.perf_counter() - t_i0
                    st["ocr_pages"] += 1
                    _cache_put(key, pdf_file, i, OCR_DPI, engine_label(0),
                               texts, scores, polys, avg)
                    if _VERBOSE:
                        print(f"    [p{i+1}] OCR {dt0:.2f}s avg={avg:.3f} "
                              f"n={len(texts)} dpi={OCR_DPI}", flush=True)
                    # 低置信度 -> 升档：300 DPI + server 模型重识别一次，二者取优（缓存两档都写）
                    if avg < OCR_SCORE_MIN:
                        try:
                            bgr2, _w2 = _render_page(page, OCR_DPI_BAD)
                            key2 = _cache_key(pdf_file, i, OCR_DPI_BAD, engine_label(1))
                            hit2 = _cache_get(key2)
                            if hit2 is not None:
                                t2 = hit2
                                st["cached_pages"] += 1
                            else:
                                eng1 = get_ocr_engine(1)
                                t2 = _ocr_page_on(eng1, bgr2)
                                _cache_put(key2, pdf_file, i, OCR_DPI_BAD,
                                           engine_label(1), *t2)
                            if t2[3] > avg and any(str(x).strip() for x in t2[0]):
                                texts, scores, polys, avg = t2
                                used_dpi = OCR_DPI_BAD
                                tier_used = 1
                                st["esc_pages"] += 1
                        except Exception as e:
                            print(f"    [WARN] p{i+1} 升档失败（保留原结果）: {e}", flush=True)
                st["engine0"] = engine_label(0)
                if tier_used == 1:
                    st["engine1"] = engine_label(1)
                if not texts or not any(str(x).strip() for x in texts):
                    st["empty_pages"] += 1
                    continue
                n_ok, n_bad = _insert_page_text(page, texts, scores, polys, used_dpi)
                if n_ok:
                    st["inserted"] += 1
                else:
                    st["empty_pages"] += 1
                st["insert_fail"] += n_bad
            except _CudaOOM as e:
                # A3: CUDA 显存不足 —— 置熔断标志，停止本文件剩余页（不落库），交父进程降并发/降批大小重试
                st["vram_oom"] = True
                st["fail_pages"] += 1
                st["fail_list"].append(i + 1)
                print(f"    [VRAM-OOM] {os.path.basename(pdf_file)} p{i+1} CUDA 显存不足，停止本文件: {e}", flush=True)
                break
            except Exception as e:
                st["fail_pages"] += 1
                st["fail_list"].append(i + 1)
                print(f"    [ERR] {os.path.basename(pdf_file)} p{i+1} 处理失败: {e}", flush=True)
            finally:
                t_pages.append(time.perf_counter() - t_p0)
        # 只在真的有插入时才动文件：全命中/全跳过 -> 零写入
        if st["inserted"]:
            try:
                doc.saveIncr()
            except Exception:
                tmp = str(pdf_file) + ".octmp"
                doc.save(tmp, garbage=3, deflate=True)
                defer_replace = tmp
    except Exception as e:
        st["err"] = f"主流程失败: {e}"
        print(f"[ERR] {os.path.basename(pdf_file)} 主流程失败: {e}", flush=True)
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if defer_replace:
        try:
            os.replace(defer_replace, pdf_file)
            print("    [INFO] 增量保存不可用，已全量重写保存", flush=True)
        except OSError as e:
            st["err"] = f"保存失败: {e}"
            print(f"[ERR] 保存失败 {os.path.basename(pdf_file)}: {e}", flush=True)
    if t_pages:
        st["page_sec_avg"] = round(sum(t_pages) / len(t_pages), 2)
        st["page_sec_max"] = round(max(t_pages), 2)
    denom = st["cached_pages"] + st["ocr_pages"]
    st["cache_hit_rate"] = round(100.0 * st["cached_pages"] / denom, 1) if denom else 0.0
    st["sec"] = round(time.time() - t_file0, 1)
    if st["fail_pages"]:
        st["status"] = "fail"
        if not st["err"]:
            tail = "..." if st["fail_pages"] > 8 else ""
            st["err"] = f"{st['fail_pages']} 页处理失败: {st['fail_list'][:8]}{tail}"
    else:
        st["status"] = "ok"
    if _MONITOR:
        st["monitor"] = _monitor_snapshot()
    mo = st.get("monitor") or {}
    mo_s = ""
    if mo:
        mo_s = (f" | MON cpu={mo.get('cpu_pct', '-')}% "
                f"mem={mo.get('mem_used_gb', '-')}/{mo.get('mem_total_gb', '-')}GB "
                f"gpu={mo.get('gpu_util_pct', '-')}%/{mo.get('gpu_mem_used_mb', '-')}MB "
                f"disk_free={mo.get('disk_free_gb', '-')}GB")
    print(f"[OCR] {os.path.basename(pdf_file)} status={st['status']} 页{st['pages']}"
          f"(文字层{st['text_pages']}/空白{st['blank_pages']}/OCR{st['ocr_pages']}/"
          f"缓存{st['cached_pages']}/升档{st['esc_pages']}/空{st['empty_pages']}/"
          f"失败{st['fail_pages']}) 插入{st['inserted']}页 均耗{st['page_sec_avg']}s "
          f"缓存命中{st['cache_hit_rate']}% 设备{st['device']} 总耗时{st['sec']}s{mo_s}",
          flush=True)
    return st


# ==================== 九、兼容接口 + CLI ====================
def check_pdf_ocr_add_text(pdf_file: str):
    """兼容旧接口：页级 OCR（含逐页判层/缓存/升档/监控）。返回统计 dict；旧调用方可忽略返回值。"""
    return ocr_pdf_with_textlayer(pdf_file)


def read_pdf_page0(pdf_file: str):
    """兼容保留：读首页文字（去空行/空白/伪影）；无文字返回 None。"""
    with fitz.open(pdf_file) as pdf:
        page = pdf.load_page(0)
        text = page.get_text()
        if text:
            text = '\n'.join([line for line in text.split('\n') if line.strip() != ''])
            text = re.sub(r'\s+', '', text)
            text = re.sub(r'\r\n+', '', text)
            text = _ARTS.sub('', text)
            return text
    return None


def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser(description="页级 OCR 补透明文字层（CLI）")
    ap.add_argument("--pdf", help="单个 PDF 文件")
    ap.add_argument("--dir", help="目录（递归处理其中全部 PDF）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 个文件（0=全部）")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], help="覆盖 OCR_DEVICE")
    ap.add_argument("--dpi", type=int, help="覆盖 OCR_DPI")
    ap.add_argument("--force", action="store_true", help="清掉这些文件的页级缓存后重跑")
    ap.add_argument("--report", help="统计汇总写入 JSON 文件")
    a = ap.parse_args(argv)
    if a.device:
        os.environ["OCR_DEVICE"] = a.device
        globals()["_OCR_DEVICE_WANT"] = a.device
        if a.device == "cpu":
            t = _env("OCR_THREADS", "1")
            t = t if t in ("1", "2") else "1"
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                os.environ[v] = t
            globals()["_CPU_THREADS"] = int(t)
    if a.dpi:
        globals()["OCR_DPI"] = max(72, min(600, a.dpi))
    files = []
    if a.pdf:
        files = [a.pdf]
    elif a.dir:
        for root, _dirs, names in os.walk(a.dir):
            for nm in names:
                if nm.lower().endswith(".pdf"):
                    files.append(os.path.join(root, nm))
        files.sort()
    else:
        ap.print_help()
        dd0 = device_decision()
        print(f"\n[ENV] device={dd0['device']}（{dd0['reason']}）")
        print(f"[ENV] nvidia-smi: {'有' if _GPU_INFO['has_nvidia'] else '无'} "
              f"{_GPU_INFO['name'] or ''} {_GPU_INFO['note'] or ''}")
        if _NVIDIA_DLL_DIRS:
            print(f"[ENV] nvidia DLL 注入 {len(_NVIDIA_DLL_DIRS)} 个目录（cublas/cudnn/cudart 等）")
        print(f"[ENV] dpi={OCR_DPI} dpi_bad={OCR_DPI_BAD} "
              f"aux={'关' if _aux_disabled() else '开'} cache={_CACHE_PATH or 'off'}")
        return 1
    if a.limit and a.limit > 0:
        files = files[:a.limit]
    dd = device_decision()
    print(f"[ENV] device={dd['device']}（{dd['reason']}）files={len(files)} "
          f"dpi={OCR_DPI} cache={'on' if _CACHE_PATH else 'off'}", flush=True)
    if a.force:
        c = _cache_conn()
        if c is not None:
            for f in files:
                try:
                    c.execute("DELETE FROM page_ocr WHERE pdf=?", (os.path.abspath(f),))
                except Exception:
                    pass
            try:
                c.commit()
            except Exception:
                pass
            print("[INFO] --force：已清空这些文件的页级缓存", flush=True)
    stats = []
    for f in files:
        stats.append(ocr_pdf_with_textlayer(f))
    ok = sum(1 for s in stats if s.get("status") == "ok")
    tot_sec = sum(s.get("sec") or 0 for s in stats)
    n_pages = sum(s.get("ocr_pages") or 0 for s in stats)
    print(f"[SUM] 文件 {ok}/{len(stats)} 成功，OCR 页 {n_pages}，总耗时 {tot_sec:.1f}s")
    if a.report:
        try:
            with open(a.report, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, ensure_ascii=False, indent=1)
            print(f"[SUM] 明细已写入 {a.report}")
        except OSError as e:
            print(f"[WARN] 报告写入失败: {e}")
    return 0 if ok == len(stats) else 1


if __name__ == '__main__':
    raise SystemExit(_cli(sys.argv[1:]))
