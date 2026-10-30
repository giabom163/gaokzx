# -*- coding: utf-8 -*-
"""OCR 批处理子进程：在一个进程里连续处理一整批 PDF。

为什么按批判（而不是一文件一进程）：PaddleOCR 模型初始化要花时间。一文件一进程
会把这份开销付 565 次；一个 worker 只加载一次引擎，批内所有文件共用 —— N 个
worker 就只付 N 次。进程隔离的好处仍然保留：模型/驱动级崩溃只损失这一批。

用法: python wm_ocr_batch_worker.py <json_list_file>
输出: 向 stdout 写 RESULT<json array>（父进程按 "RESULT" 前缀截取）
     同时每完成一个文件，向 <json_list_file>.prog 追加一行 jsonl 实时进度
     （父进程 1.5s 轮询读取，看板进度实时可见，不再长时间停在 0/N）

进度行/结果行字段（页级 OCR 版）：file/status/err/sec + pages/ocr_pages/cached_pages/
esc_pages/blank_pages/text_pages/fail_pages/inserted/page_sec_avg/cache_hit_rate/
device/monitor（CPU% 内存 GPU 磁盘快照）。旧字段 file/status/err/sec 始终保留，
父进程（step_ocr）无需感知新增字段。
"""
import sys, os, json, time, traceback

# ⚠ 关键性能设置：必须在本文件 import paddle/opencv 之前执行。
# PaddleOCR 内部用 OpenMP/MKL 多线程，默认吃满全部逻辑核（本机 32）。
# 若同时跑 N 个 worker 而不限制，就是 N×32 线程去抢 16 个物理核 → 严重超订、
# 互相抖动，实测反而比单进程还慢。父进程通过环境变量给出每进程线程预算，
# 这里在 import 之前把它落到 OpenMP/MKL 真正读取的环境变量上。
# 父进程（step_ocr）已直接写死 OMP/MKL/OPENBLAS 等，这里只做兜底与兼容。
_THREADS = (os.environ.get("OCR_THREADS")
            or os.environ.get("OMP_NUM_THREADS")
            or "").strip()
if _THREADS:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _THREADS          # 强制覆盖，不用 setdefault（防外部污染）

# 关掉三个"辅助模型"：试卷扫描件本来正放、平整，方向分类 / UVDoc 去扭曲 /
# 文本行方向分类都属浪费（A/B 实测这三个合计约占推理时间 8%）。
# 用环境变量传给引擎构造处（BatchOcrPdfAddTextLayer.get_ocr_engine）。
os.environ.setdefault("OCR_DISABLE_AUX_MODELS", "1")


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: wm_ocr_batch_worker.py <list.json>\n")
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as fh:
        files = json.load(fh)

    # 逐文件实时进度：每处理完一个文件就追加一行 jsonl（file/status/err/sec）。
    # 写失败静默忽略——进度上报绝不影响 OCR 主流程。
    prog_path = sys.argv[1] + ".prog"

    def _emit_prog(rec):
        try:
            with open(prog_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    _fallback_xer = r"D:\work\xer-doc-paddle-ocr"
    if os.path.isdir(_fallback_xer) and _fallback_xer not in sys.path:
        sys.path.append(_fallback_xer)
    import pymupdf as fitz

    def has_text(p):
        try:
            with fitz.open(p) as d:
                return bool(d[0].get_text().strip())
        except Exception:
            return False

    try:
        from BatchOcrPdfAddTextLayer import check_pdf_ocr_add_text, vram_guard_tripped
    except Exception:
        # 引擎都 import 不进来：把整批标失败（含一次 import traceback）
        tb = traceback.format_exc()[-700:]
        res = [{"file": f, "status": "fail", "err": "import: " + tb}
               for f in files]
        for r in res:
            _emit_prog(r)
        sys.stdout.write("RESULT" + json.dumps(res))
        sys.stdout.flush()
        return

    def text_state(p):
        """返回 (总页数, 有文字层页数)；打开失败返回 (0, 0)。用于处理后的兜底验证。"""
        try:
            with fitz.open(p) as d:
                n = d.page_count
                k = 0
                for i in range(n):
                    try:
                        if d[i].get_text().strip():
                            k += 1
                    except Exception:
                        pass
                return n, k
        except Exception:
            return 0, 0

    out = []
    _guard_hit = False       # 显存看门狗/OOM 触发后，剩余文件一律标 skip(vram) 交父进程降级
    for i, f in enumerate(files):
        # A2: 显存看门狗命中（used/total>=90%）——跳过本批剩余文件，标记 skip(vram) 由父进程降并发重派
        if not _guard_hit and vram_guard_tripped():
            _guard_hit = True
            print(f"[VRAM-GUARD] 显存占满，跳过本批剩余 {len(files) - i} 个文件", flush=True)
        if _guard_hit:
            rec = {"file": f, "status": "skip", "vram_guard": True, "err": "skip(vram)",
                   "device": None, "monitor": {}}
            rec["sec"] = 0
            out.append(rec)
            _emit_prog(rec)
            continue
        _t0 = time.time()
        try:
            st = check_pdf_ocr_add_text(f)
            if isinstance(st, dict):
                # 页级 OCR 版：直接消费统计字典（含页数/缓存/升档/监控）
                rec = {"file": f, "status": st.get("status") or "fail",
                       "err": st.get("err"),
                       "pages": st.get("pages"), "ocr_pages": st.get("ocr_pages"),
                       "cached_pages": st.get("cached_pages"),
                       "esc_pages": st.get("esc_pages"),
                       "blank_pages": st.get("blank_pages"),
                       "text_pages": st.get("text_pages"),
                       "fail_pages": st.get("fail_pages"),
                       "inserted": st.get("inserted"),
                       "page_sec_avg": st.get("page_sec_avg"),
                       "cache_hit_rate": st.get("cache_hit_rate"),
                       "device": st.get("device"),
                       "monitor": st.get("monitor"),
                       "vram_oom": bool(st.get("vram_oom")),
                       "vram_guard": bool(st.get("vram_guard"))}
                # A3: 本文件 CUDA OOM 熔断 —— 停止本批剩余，避免连续 OOM 抖动，交父进程降并发/降批重试
                if rec["vram_oom"] and not _guard_hit:
                    _guard_hit = True
                    print(f"[VRAM-OOM] {os.path.basename(f)} CUDA OOM，停止本批剩余", flush=True)
                # 兜底验证：模块报 ok 但整本 0 文字层 -> 失败（全空白/纯图无字）
                n_pg, n_txt = text_state(f)
                rec["pages_total"], rec["pages_text"] = n_pg, n_txt
                if rec["status"] == "ok" and n_pg and n_txt == 0:
                    rec["status"] = "fail"
                    rec["err"] = "OCR 后整本仍无文字层（可能全为空白/纯图页）"
            else:
                # 旧版模块（返回 None）：退回页0文字层验证
                check_pdf_ocr_add_text(f)
                ok = has_text(f)
                rec = {"file": f, "status": "ok" if ok else "fail",
                       "err": None if ok else "still no text layer",
                       "vram_oom": False, "vram_guard": False}
        except Exception:
            rec = {"file": f, "status": "fail",
                   "err": traceback.format_exc()[-500:],
                   "vram_oom": False, "vram_guard": False}
        rec["sec"] = round(time.time() - _t0, 1)
        out.append(rec)
        _emit_prog(rec)      # 实时进度（父进程轮询读取）
    # 批汇总：缓存命中/升档/设备（父进程可读取最后一行 RESULT 的解包统计）
    _agg = {"files": len(out),
            "ok": sum(1 for r in out if r.get("status") == "ok"),
            "ocr_pages": sum(r.get("ocr_pages") or 0 for r in out),
            "cached_pages": sum(r.get("cached_pages") or 0 for r in out),
            "esc_pages": sum(r.get("esc_pages") or 0 for r in out),
            "device": next((r.get("device") for r in out if r.get("device")), None)}
    _den = _agg["ocr_pages"] + _agg["cached_pages"]
    _agg["cache_hit_rate"] = round(100.0 * _agg["cached_pages"] / _den, 1) if _den else 0.0
    print("[AGG] " + json.dumps(_agg, ensure_ascii=False), flush=True)
    sys.stdout.write("RESULT" + json.dumps(out))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
