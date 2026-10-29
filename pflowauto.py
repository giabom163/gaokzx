#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""gaokzx 流水线 · 自动编排（pflowauto）—— 2026-10-22 自 pflow.py 拆出。

职责：--auto 无人值守编排（run_auto/_run_auto_impl/verify_run_completeness）、
断点续跑（_detect_resume_stage/_pick_resumable_batch/_revive_manifest_paths 族）、
菜单公共中段（_flow_ctx/_ask_paper_type）。持有 FORCE_MISSING。
依赖：pcore/prename/pwm/pdownload/pocr/ppack/padmin。被 pflow/pipeline/padmin（延迟 import）共享。
"""
import os
import re
import traceback
import pcore
import ppack
from pcore import (
    AUTO_ACCEPT,
    STAGE_ADS,
    STAGE_CLEANED,
    STAGE_DONE,
    STAGE_PENDING,
    STEP_ACCEPT,
    STEP_AD_CLEAN,
    STEP_DOWNLOAD,
    STEP_OCR,
    STEP_PACK,
    STEP_WATERMARK,
    _DASH,
    _SEP,
    _ask,
    _hr,
    _scan,
    _selfcheck,
    ask_continue_or_abort,
    bc,
    db,
    ensure_dirs,
    fatal_hint,
    has_targets,
    ledgerio,
    run_step_safe,
)
from prename import (
    step_rename,
)
from pwm import (
    step_watermark,
)
from pdownload import (
    _new_batch_for_url,
    _print_rc3_banner,
    _report_empty_run,
    sniff_url,
    step_ads,
    step_discover_download,
)
from pocr import (
    ocr_worker_budget,
    step_ocr,
    step_qc,
)
from ppack import (
    _batch_zips_exist,
    _final_zip_check,
    _pack_mode,
    _parse_file_ids,
    _stage_root_of,
    _write_auto_crash_log,
    install_auto_crash_hook,
    reconcile_round,
    step_pack,
)


# --------------------- pcore.DB↔磁盘 一致性（跳过三条件 + 双向核对） ---------------------
FORCE_MISSING = False   # --force-missing：只强制重跑“pcore.DB done 但磁盘缺失”的文件


# ---- 向导流程 ----
def _flow_ctx(no_batch_msg, stage, empty_msg):
    """向导公共中段：当前批次 -> 阶段扫描 -> 空防呆；返回 (bid, pairs)。
    无批次/无文件时打印对应提示并返回 (None, None)。"""
    bid = bc.current_batch_id() if bc is not None else None
    if not bid:
        print(no_batch_msg)
        return None, None
    pairs = _scan(stage, bid) if bc is not None else []
    print(f"  批次 : {bid}（{len(pairs)} 个 PDF，manifest 限定）")
    if not pairs:
        print(empty_msg.format(bid=bid))
        return None, None
    return bid, pairs

def _ask_paper_type(cands, ask=None, label="试卷类型"):
    """交互式选择试卷类型（菜单 [d] 与 auto 暂停共用）。

    cands: collect_paper_candidates 产出的候选原文列表；空列表 -> 不询问直接 (None,'skip')。
    ask:   取词函数 prompt->str|None。默认用 _ask（菜单）；auto 传入直读 stdin 的闭包，
           在 EOF/中断时返回 None 以触发 'abort'（建批前保守中止）。
    返回 (paper_type, action)：
      'skip'  回车/0/3次非法 -> 不筛选，paper_type=None；
      'set'   选中候选或自由关键词 -> paper_type 为逗号拼接字符串；
      'abort' stdin 不可用（仅 auto）-> paper_type=None，调用方 return 1。
    """
    if not cands:
        return None, "skip"
    ask = ask or (lambda p: _ask(p))
    for _ in range(3):
        _hr("-")
        print(f"按{label}筛选（可选，直接回车=不筛选抓全部）：")
        for i, c in enumerate(cands[:30], 1):
            print(f"  {i:2d}. {c}")
        if len(cands) > 30:
            print(f"  （仅显示前 30 / 共 {len(cands)} 项，可直接输入关键词）")
        raw = ask("输入序号（可逗号多选）或关键词: ")
        if raw is None:
            return None, "abort"
        raw = raw.strip()
        if raw in ("", "0", "q"):
            return None, "skip"
        toks = [t for t in re.split(r"[,，、\s]+", raw) if t]
        if toks and all(t.isdigit() for t in toks):
            picked, bad = [], False
            for t in toks:
                idx = int(t)
                if 1 <= idx <= len(cands):
                    picked.append(cands[idx - 1])
                else:
                    bad = True
            if bad:
                print("  [!] 存在越界编号，请重输。")
                continue
            out = []
            for p in picked:
                if p not in out:
                    out.append(p)
            return ",".join(out), "set"
        return raw, "set"
    print("  未选择有效条件，按不筛选继续。")
    return None, "skip"


# 无交互全自动后端（CLI --auto / 看板调度 / 菜单[1] 公共入口）
# 流水线阶段顺序：默认旧版兼容。若设置 GAOKZX_CLEAN_FIRST=1（推荐新架构），
# 则将 ads（去底边广告与删纯广告页）前移至 watermark 后、ocr 前，实现"先净图、后注字"，
# 避免在纯广告页上浪费 OCR 算力，且杜绝注字后又做 redact 导致文字层损坏。
_CLEAN_FIRST = os.environ.get("GAOKZX_CLEAN_FIRST", "0") in ("1", "true", "True")
if _CLEAN_FIRST:
    STAGE_SEQ = ("download", "watermark", "ads", "rename", "ocr", "qc", "pack")
else:
    STAGE_SEQ = ("download", "watermark", "rename", "ocr", "qc", "ads", "pack")
_STAGE_ALIAS = {
    "download": "download", "discover": "download", "1": "download",
    "下载": "download", "抓取": "download", "采集": "download",
    "watermark": "watermark", "wm": "watermark", "2": "watermark",
    "去水印": "watermark", "水印": "watermark",
    "rename": "rename", "3": "rename", "重命名": "rename",
    "ocr": "ocr", "4": "ocr", "补层": "ocr", "补文字层": "ocr",
    "qc": "qc", "5": "qc", "质控": "qc",
    "ads": "ads", "ad_clean": "ads", "adclean": "ads", "6": "ads",
    "清广告": "ads", "隐形水印": "ads",
    "pack": "pack", "7": "pack", "打包": "pack",
}
# 起点阶段 -> (找回目录（按优先级）, 视为"已在输入位置"的 tag 集合)；打回重跑时按此把 manifest 路径回退到起点输入。
_REVIVE_PLAN = {
    "watermark": (("pending",), ("pending",)),
    "rename": (("cleaned",), ("cleaned",)),
    "ocr": (("cleaned",), ("cleaned",)),
    "qc": (("cleaned", "done"), ("cleaned", "done")),
    "ads": (("cleaned", "done"), ("cleaned", "done")),
    "pack": (("ads",), ("ads",)),
}

def normalize_stage(s):
    """把 --start-stage 的值规范成 STAGE_SEQ 的英文阶段名；无法识别返回 None。"""
    key = str(s or "").strip().lower()
    return _STAGE_ALIAS.get(key) if key else None

def _detect_resume_stage(bid):
    """菜单[1] 回车续跑的断点探测：按 manifest 读 file_steps，从"最晚的未完成环节"续跑（全部已到
    「已清广告」-> 直接 pack；全部补完文字层 -> 从 qc；其余 -> 最落后步骤）。返回 (stage, note)；
    探测失败返回 (None, "")。"""
    try:
        if bc is None or not bid: return None, ""
        mf = bc.load_manifest(bid) or {}
        fids = [str(f.get("file_id") or "") for f in mf.get("files", [])]
        fids = [x for x in fids if x]
        if not fids: return None, ""
        con = db()
        try:
            rows = con.execute(
                "SELECT file_id, step, status FROM file_steps WHERE batch_id=?",
                (str(bid),)).fetchall()
        finally: con.close()
        st = {}
        for fid, step, status in rows: st.setdefault(str(fid), {})[int(step)] = str(status or "")
        _ok = ("ok", "skip")
        def _all_done(step):
            return all(st.get(f, {}).get(step) in _ok for f in fids)
        n = len(fids)
        if _all_done(STEP_PACK): return "pack", f"{n} 个文件均已打包过，将重新执行打包 zip"
        if _all_done(STEP_AD_CLEAN): return "pack", f"{n} 个文件已全部到「已清广告」，直接执行打包 zip"
        if _all_done(STEP_OCR): return "qc", f"{n} 个文件文字层已补齐，从「质控 QC」续跑直达打包"
        if _all_done(STEP_WATERMARK): return "ocr", f"从「OCR 补层」续跑直达打包（{n} 个文件）"
        return "watermark", f"从「去水印」续跑直达打包（{n} 个文件）"
    except Exception: return None, ""

def _pick_resumable_batch():
    """菜单[1] 回车续跑的批次选择：优先 current；current 指向空批次（如刚探测过 0 PDF 的网址）
    时按 manifest 时间倒序回溯到最近真正有文件的批次，避免空跑。失败返回 None（调用方回退提示）。"""
    if bc is None: return None
    try:
        cur = bc.current_batch_id()
        if cur and (bc.load_manifest(cur) or {}).get("files"): return cur
        for p in sorted(bc.BATCH_DIR.glob("batch_*.json"), reverse=True):
            b = p.stem[len("batch_"):]
            if b == cur: continue
            if (bc.load_manifest(b) or {}).get("files"): return b
        return cur
    except Exception: return None

def _find_in_stage(root, name):
    """在阶段根目录树里按文件名（不区分大小写）找回文件；找不到返回 None。"""
    if not root or not root.is_dir() or not name: return None
    want = str(name).lower()
    for dirpath, _dirs, names in os.walk(str(root)):
        for n in names:
            if n.lower() == want: return os.path.join(dirpath, n)
    return None

def _revive_manifest_paths(bid, fids, stage):
    """打回重跑 / 中间续跑：把 manifest 路径回退到起点阶段的输入目录。背景：文件流走后 manifest.path 指向产物
    （去水印后/已清广告/…），从更早的 --start-stage 重跑时 scan_pdfs 在源目录找不到文件而空跑；本函数按「起点
    阶段需要的输入目录」找回文件并 update_file_path。返回 (回退成功数, [无法找回的 file_id])。"""
    if bc is None: return 0, []
    plan = _REVIVE_PLAN.get(stage)
    if not plan: return 0, []
    want_dirs, skip_tags = plan
    roots = {"pending": STAGE_PENDING, "cleaned": STAGE_CLEANED, "done": STAGE_DONE, "ads": STAGE_ADS}
    mf = bc.load_manifest(bid) or {}
    revived, missing = 0, []
    for f in mf.get("files", []):
        fid = str(f.get("file_id") or "")
        if not fid: continue
        if fids is not None and fid.upper() not in fids: continue
        p = str(f.get("path") or "")
        tag, _root = _stage_root_of(p) if p else (None, None)
        if tag in skip_tags and p and os.path.isfile(p):
            continue          # 已在输入目录（或文件仍在原处）—— 无需回退
        name = os.path.basename(p) or str(f.get("name") or "")
        src_p = None
        for w in want_dirs:
            src_p = _find_in_stage(roots.get(w), name)
            if src_p: break
        if src_p:
            if bc.update_file_path(bid, fid, src_p):
                print(f"  [revive] {fid} 路径回退 -> {src_p}")
                revived += 1
        else: missing.append(fid)
    return revived, missing

def _run_auto_stage(con, stage, bid, force, wm_workers=6, ocr_workers=None, ocr_threads=None,
                    tag="", only_unpacked=False):
    """run_auto 的单阶段分发。返回 None=通过（含"无事可做"）；非 None=失败码。
    tag/only_unpacked：仅透传给 step_pack（分批流水线：zip 名带轮次后缀、每轮只打包新文件）。"""
    if stage == "watermark":
        step_watermark(con, limit=None, dry=False, workers=wm_workers, force=force, batch_id=bid)
    elif stage == "rename":
        step_rename(con, limit=None, dry=False, use_ocr=False,
                    log_path=str(STAGE_CLEANED / "_报告" / "rename_map.json"),
                    batch_id=bid)
    elif stage == "ocr":
        _w, _t, _note = ocr_worker_budget(ocr_workers or 0, ocr_threads or 0)
        if _note: print(f"  [!] {_note}")
        step_ocr(con, None, False, None, workers=_w, batch=4,
                 threads_per_worker=_t, force=force, batch_id=bid)
    elif stage == "qc": step_qc(con, False, None, batch_id=bid)
    elif stage == "ads":
        _out = STAGE_CLEANED if _CLEAN_FIRST else None
        rc = step_ads(dry=False, out=_out, batch_id=bid, force=force)
        if rc == 3:
            print(">>> [auto] 清广告报告校验失败（缺失/过旧/批次不匹配），中止。")
            return 3
        if rc not in (0, None): return rc or 1
    elif stage == "pack":
        # 分批流水线 + topic：中间轮不单独出包（避一专题碎成 N 个 zip）；打包延后到 run 末统一进行。
        # only_unpacked=True 仅分批轮次会传（单轮/中段续跑为 False，照常就地打包）。
        if _pack_mode() == "topic" and only_unpacked:
            print(">>> [pack] 分批流水线(topic)：本批不单独出包，延后到本次运行结束按专题统一打包。")
            return None
        zip_p = step_pack(con, None, None, dry=False, batch_id=bid,
                          tag=tag, only_unpacked=only_unpacked)
        if zip_p == "ALREADY":
            print(">>> [pack] 本批次文件均已打包过（分轮去重），跳过本轮打包。")
            return None
        if not zip_p: return 1
    return None

def verify_run_completeness(run_id, url_ident):
    """机制1（--force-full-run 才调用）："跑不完不许结束"的完整性断言。纯只读校验，绝不改库/改状态。
    核验三件事，任一不满足即 ok=False（调用方据此拒绝自动验收、保留 pending_accept、退出码 1）：
      1) run 下每个批次：本轮"下载成功(step1=ok)"的文件都到达验收步(step8 ∈ {pending_accept, accepted, skip})；
      2) zip 内 PDF 数 == 待/已验收数（batch 模式逐批；topic 模式整 run 合并为少数专题 zip，改按 run 去重 zip 内 PDF 总数 == 各批终态之和）；
      3) 该下载 URL 的续跑游标 discover_cursor == total_target（专题确已遍历完）。
    返回 (ok, reasons)。无台账后端(bc=None)时返回 (True, []) —— 无从校验则保持旧行为、绝不误伤。
    reasons 汇总全部可判定信号（如 cursor=24 < total=46），供人工介入定位。"""
    reasons = []
    if bc is None:
        return True, []
    bids = []
    if run_id:
        try:
            bids = list(bc.db_batches_of_run(run_id) or [])
        except Exception:
            bids = []
    _topic = _pack_mode() == "topic"
    import zipfile as _zip
    _sum_term = 0
    for _b in bids:
        try:
            cnt = bc.db_step_counts(_b) or {}
        except Exception:
            cnt = {}
        _dl = int((cnt.get(STEP_DOWNLOAD) or {}).get(bc.ST_OK, 0))
        s8 = cnt.get(STEP_ACCEPT) or {}
        _pa = int(s8.get(bc.ST_PENDING_ACCEPT, 0))
        _ac = int(s8.get(bc.ST_ACCEPTED, 0))
        _sk = int(s8.get(bc.ST_SKIP, 0))
        _term = _pa + _ac
        _reached = _term + _sk
        if _dl and _reached < _dl:
            reasons.append(f"batch {_b}: 下载成功 {_dl} 个，仅 {_reached} 个到达验收步(step8)——有文件卡在打包/验收前")
        _sum_term += _term
        if not _topic:
            # batch 模式：逐批 zip 内 PDF 数 == 该批待/已验收数
            zips = _batch_zips_exist(_b)
            if zips and _term:
                _zn, _zok = 0, True
                for _z in zips:
                    try:
                        with _zip.ZipFile(_z) as _zf:
                            _zn += sum(1 for n in _zf.namelist() if n.lower().endswith('.pdf'))
                    except Exception:
                        _zok = False
                        reasons.append(f"batch {_b}: 无法读取 zip 校验文件数（{_z}）")
                        break
                if _zok and _zn != _term:
                    reasons.append(f"batch {_b}: zip 内 PDF {_zn} 个 != 待/已验收 {_term} 个")
    if _topic and bids and _sum_term:
        # topic 模式：整 run 合并成少数专题 zip，逐批计数无意义 -> run 级：去重 zip 内 PDF 总数 == 各批待/已验收之和
        _rz = []
        for _b in bids:
            for _z in _batch_zips_exist(_b):
                if _z not in _rz:
                    _rz.append(_z)
        if _rz:
            _zn, _zok = 0, True
            for _z in _rz:
                try:
                    with _zip.ZipFile(_z) as _zf:
                        _zn += sum(1 for n in _zf.namelist() if n.lower().endswith('.pdf'))
                except Exception:
                    _zok = False
                    reasons.append(f"run {run_id}: 无法读取 zip 校验文件数（{_z}）")
                    break
            if _zok and _zn != _sum_term:
                reasons.append(f"run {run_id}: 专题包内 PDF 共 {_zn} 个 != 各批待/已验收之和 {_sum_term}")
    if url_ident:
        try:
            _rid, _cur, _tot, _fp = bc.db_url_cursor(url_ident)
        except Exception:
            _rid, _cur, _tot = (None, 0, None)
        if _tot is not None:
            try:
                _cur_i, _tot_i = int(_cur or 0), int(_tot)
                if _cur_i < _tot_i:
                    reasons.append(f"cursor={_cur_i} < total={_tot_i}（URL 未跑到末尾：后面可能还有目标未采集）")
                elif _cur_i > _tot_i:
                    reasons.append(f"cursor={_cur_i} > total={_tot_i}（游标超过总量，异常）")
            except (TypeError, ValueError):
                reasons.append(f"cursor/total 非法：cursor={_cur} total={_tot}")
    return (len(reasons) == 0), reasons


def run_auto(args):
    """公共后端薄封装：调用 _run_auto_impl 后统一打印 [自检] EXIT=<code>（不改内部逻辑/返回码语义）。
    并作为 KeyboardInterrupt 兵底：中断时保留 pending_accept、不误标 accepted、退出码 1。"""
    try:
        _rc = _run_auto_impl(args)
    except KeyboardInterrupt:
        print("\n[!] [auto] 已中断（保留 pending_accept，未误标 accepted；状态已落库可续跑）。")
        _selfcheck("exit", "EXIT=1 (KeyboardInterrupt)")
        return 1
    _selfcheck("exit", f"EXIT={_rc}")
    return _rc


def _run_auto_impl(args):
    """无交互全自动（CLI --auto / 看板调度 / 菜单[1] 公共后端）。从 --start-stage（默认 download）开始，按
    download -> watermark -> rename -> ocr -> qc -> ads -> pack 顺序无人值守跑完；打包成功后批次状态 ->
    pending_accept（待验收）。--file-id 只处理指定文件（看板调度）；--batch-id 显式批次。退出码：0=打包成功；1=失败；2=参数不满足。
    --chunk>0（默认 24）且从 download 起跑、非单文件调度时进入分批流水线：每批最多 N 个目标，下载完一批立即
    接力跑阶段 2~7（pack 按轮增量打包、zip 名带 _rN），下载器空结果即结束；--chunk 0 = 旧行为（全量下载后统一跑）。
    下载游标按"URL"隔离断点续跑（换 URL 不继承旧 URL 的 offset）。退出码：0=打包成功或本批无新 PDF（正常空转）；
    1=阶段失败；2=参数不满足。未捕获崩溃会写完整 traceback 到 logs/auto_crash_*.log。"""
    global FORCE_MISSING
    ppack._LAST_PACK_ZIP = None          # 修「假成功」前置：重置本次产出标志，杜绝跨运行残留导致误判"已产出/已验收"
    ppack._LAST_PACK_ZIPS = []           # 同上：topic 多 zip 伴生列表一并重置
    _run_bids = []                 # 本次运行创建的各轮 batch_id（收尾统一标待验收）；函数开头初始化，保证各 start 路径均有定义
    install_auto_crash_hook()      # 任何逃逸到顶层的崩溃都先落盘 traceback（闪退专项：绝不"无日志闪退"）
    raw_stage = getattr(args, "start_stage", None)
    if getattr(args, "force_missing", False):
        # --auto --force-missing：每轮 reconcile 额外回滚“产物缺失/哈希不符”的 ok 行，
        # 回滚后 is_step_ok=False，后续阶段必然重处理（不会被 _skip_ok 二次拦截）。
        FORCE_MISSING = True
        print(">>> [auto] --force-missing 已启用：每轮核对后强制重跑产物缺失/内容变更文件")
    if raw_stage and normalize_stage(raw_stage) is None:
        print(f">>> [auto] 无法识别的 --start-stage：{raw_stage}")
        print(f"            可用：{', '.join(STAGE_SEQ)}（或中文别名：去水印/重命名/OCR/…）")
        return 2
    start = normalize_stage(raw_stage) or "download"
    fids = _parse_file_ids(getattr(args, "file_id", None))
    force = bool(getattr(args, "force", False))
    dry = bool(getattr(args, "dry_run", False))
    try:
        chunk = int(getattr(args, "chunk", 24))
    except (TypeError, ValueError):
        chunk = 24
    chunk = max(0, chunk)        # 0 = 关闭分批流水线（旧行为：全量下载后统一跑后续阶段）
    pcore.AUTO_MODE = True
    pcore.ONLY_FILE_IDS = fids
    _staged_multi = False        # 分批流水线标志：True 时阶段 2~7 已在下载循环内逐批跑完
    _run_id = None               # 分批块内赋值；收尾兜底（机制1 verify 用；非分批路径保持 None 由 db_run_of_batch 回推）
    _url_ident = None            # 同上：本次下载 URL 的游标标识（机制1 verify 用）
    _new_total = 0               # 本次运行累计"新下载"文件数（收尾判"本批无新 PDF"正常结束）；默认 0 兼容所有路径
    print(_SEP)
    print("一键全自动（无人值守）" + ("  [dry-run]" if dry else ""))
    print(_SEP)
    print(f"  起点阶段 : {start}（{STAGE_SEQ.index(start) + 1}/{len(STAGE_SEQ)}）")
    print(f"  文件范围 : {', '.join(sorted(fids)) if fids else '整批'}")
    print(f"  重跑模式 : {'强制重跑 --force' if force else '断点续传（已完成自动跳过）'}")
    if start == "download" and chunk > 0 and not fids and not dry:
        print(f"  分批流水线 : 每批最多 {chunk} 个（下载一批 -> 立即续跑后续阶段 -> 下一批）")
    print(_SEP)
    try:
        _dbw = (os.access(str(pcore.DB), os.W_OK) if pcore.DB.exists() else 'new')
    except Exception:
        _dbw = '?'
    try:
        _ow = os.access(str(STAGE_CLEANED), os.W_OK)
    except Exception:
        _ow = '?'
    _selfcheck("entry", f"start={start} chunk={chunk} dry={dry} force={force} "
                        f"url={getattr(args, 'url', None) or '-'} db_writable={_dbw} out_writable={_ow}")
    try:
        if bc is None:
            print(">>> [auto] batchcore 不可用（无法批次化与落库），中止。")
            return 2
        ledgerio.configure(str(pcore.DB))
        # ---- 批次解析 ----
        bid = (getattr(args, "batch_id", None) or "").strip() or None
        if bid and bc.load_manifest(bid) is None:
            print(f">>> [auto] 批次 {bid} 不存在（manifest 缺失），中止。")
            return 2
        if not bid: bid = bc.current_batch_id()
        if start != "download" and not bid:
            print(">>> [auto] 没有当前批次（也未见 --batch-id），中止。")
            print("        提示：从中间阶段续跑需要先有一次下载（步骤1）。")
            return 2
        # ---- [机制2c 默认关] 检测到上次未完成（同 URL 游标<total 或批次非终态）时询问是否续跑；abort->退出1，skip->继续原流程。----
        if getattr(args, "ask_on_resume", False):
            _ar_url = getattr(args, "url", None) or f"topic::{getattr(args, 'topic', None) or 'all'}"
            try:
                _arun, _aoff, _atot, _afp = bc.db_url_cursor(_ar_url)
            except Exception:
                _arun, _aoff, _atot = (None, 0, None)
            _aoff = _aoff or 0
            _incomplete = (_atot is not None and _aoff < int(_atot))
            if not _incomplete and bid:
                try:
                    _bst0 = (bc.load_manifest(bid) or {}).get("status")
                except Exception:
                    _bst0 = None
                _incomplete = bool(_bst0) and _bst0 not in ("accepted", "done")
            if _incomplete:
                _ch = ask_continue_or_abort(f"检测到上次未完成（游标 {_aoff}/{_atot or '?'} 或批次 {bid} 非终态）：是否续跑？", ["abort", "skip"])
                if _ch == "abort":
                    print(">>> [auto] 用户选择中止续跑（保留 pending_accept）。")
                    return 1
                print(">>> [auto] 用户选择续跑（沿用断点）。")
        # ---- file_id 校验（目标必须在本批 manifest 内）----
        if fids and start != "download":
            mf = bc.load_manifest(bid) or {}
            have = {str(f.get("file_id") or "").upper() for f in mf.get("files", [])}
            miss = sorted(fids - have)
            if miss:
                print(f">>> [auto] 批次 {bid} 中找不到：{', '.join(miss)}，中止。")
                return 2
        # ---- 阶段 2~7 执行闭包：分批模式每轮下载后立即调用；_rnd=0 表示单轮全量（旧行为） ----
        idx0 = STAGE_SEQ.index(start)
        wm_w = getattr(args, "wm_workers", None) or 6
        ocr_w = getattr(args, "ocr_workers", None)
        ocr_t = getattr(args, "ocr_threads", None)
        def _run_stages_auto(_con, _bid, _rnd=0):
            """顺序执行 STAGE_SEQ[idx0:]（跳过 download）。_rnd>0（分批第 N 轮）：pack 只打包
            未打包过的文件（only_unpacked），zip 名带 _rN 后缀。返回 None=通过；1=某阶段失败。"""
            _tag = f"_r{_rnd}" if _rnd else ""
            # 阶段入口前置校验（修问题 3）：空批次（manifest 不存在 / files=0）绝不带进 wm/rename/ocr/qc/ads/pack，
            # 直接优雅跳过（返回 None=通过），防"空 manifest 进后续阶段"导致的 None/崩溃。
            if not has_targets(_bid):
                print(f">>> [skip] 空批次（无 manifest 或 files=0），跳过全部后续阶段：{_bid}")
                return None
            # 每轮开始：对当前批次 manifest 做 pcore.DB↔磁盘双向核对（done但产物缺→回滚重处理；有产物无DB→补记）。
            # 修“已跳过 N 但产物目录 MISSING”：把一致性纠正放在所有阶段之前，后续跳过判定基于纠正后的状态。
            if not dry:
                try:
                    reconcile_round(_bid, verbose=True, force_missing=FORCE_MISSING)
                except Exception as _re:
                    print(f">>> [reconcile] 核对异常（忽略，不阻断）：{type(_re).__name__}: {_re}")
            for stage in STAGE_SEQ[idx0:]:
                if stage == "download": continue
                if dry:
                    print(f"[dry-run] 将执行阶段：{stage}")
                    continue
                print("\n" + _DASH)
                print(f"[auto] 阶段：{stage}" + (f"（分批第 {_rnd} 轮）" if _rnd else ""))
                print(_DASH)
                rc = _run_auto_stage(_con, stage, _bid, force, wm_workers=wm_w,
                                     ocr_workers=ocr_w, ocr_threads=ocr_t,
                                     tag=_tag, only_unpacked=bool(_rnd))
                if rc is not None:
                    # [机制2b 默认关] 阶段失败时询问：skip->跳过该失败阶段继续后续，abort/其它->中止返回1。
                    if getattr(args, "ask_on_stage_failure", False):
                        _ch = ask_continue_or_abort(f"阶段 {stage} 失败（rc={rc}）：中止还是跳过该阶段继续后续？", ["abort", "skip"])
                        if _ch == "skip":
                            print(f">>> [auto] 用户选择跳过失败阶段 {stage}，继续后续阶段。")
                            continue
                        print(f">>> [auto] 用户选择中止（阶段 {stage} 失败，保留 pending_accept）。")
                        return 1
                    print(f">>> [auto] 阶段 {stage} 未通过（rc={rc}），中止后续阶段。")
                    return 1
            return None
        def _dl_action(rc_dl, off, dres, bid, staged, url, topic):
            """下载返回码分流（分批/单轮共用）。返回 ("continue"|"break"|"return", code_or_None)：
            3=疑似解析失效硬中止；2=真·空（落盘空跑报告后：分批 break / 单轮 return 0）；
            False=用户跳过；其它非 0=下载未成功。"""
            if rc_dl == 3: return ("return", 3)
            if rc_dl == 2:
                _report_empty_run(url, topic, off, dres, bid)
                return ("break", None) if staged else ("return", 0)
            # rc_dl is False 特指 run_step_safe 的"用户跳过/中断"返回值（输入 s 或 KeyboardInterrupt），
            # 非下载器 rc=0；且 False==0 会被 not in (0,...) 漏判，故必须先于该判断拦截。
            if rc_dl is False:
                print(">>> [auto] 下载被用户跳过，中止。")
                return ("return", 1)
            if rc_dl not in (0, True, None):
                print(f">>> [auto] 下载未成功（rc={rc_dl}），中止。")
                return ("return", 1)
            return ("continue", None)
        # ---- 阶段1：下载（需要 --url / --topic）----
        if start == "download":
            url = getattr(args, "url", None)
            topic = (getattr(args, "topic", None) or "all")
            if not url and topic not in ("1", "2", "all"):
                print(">>> [auto] 下载起点需要 --url 或 --topic（1|2|all），中止。")
                return 2
            if dry: print(f"[dry-run] 将下载：{'url=' + url if url else 'topic=' + topic}")
            else:
                ensure_dirs(verbose=False)
                _pt = getattr(args, "paper_type", None)
                if url and not _pt:
                    # 未显式 --paper-type：进入下载前暂停一次（仅问一次，结果贯穿所有轮）。
                    # 候选取自本页表格「试卷类型」列（collect_paper_candidates）；非交互 EOF/中断 -> 保守中止 return 1
                    # （建批次之前，不动游标/manifest/file_steps）。回车/选"不筛选" -> 与原全量路径一致。
                    try:
                        _cands = getattr(sniff_url(url), "paper_cands", []) or []
                    except Exception:
                        _cands = []
                    if _cands:
                        def _ask_auto(_p):
                            try:
                                return input(_p)
                            except (EOFError, KeyboardInterrupt):
                                return None
                        _pt, _act = _ask_paper_type(_cands, ask=_ask_auto)
                        if _act == "abort":
                            print(">>> [auto] 已取消（建批次前中止，未动游标/批次）。")
                            return 1
                _yt = getattr(args, "year_type", None)
                if url and not _yt:
                    # 未显式 --year-type：进入下载前暂停一次问年份（与试卷类型平行，仅问一次贯穿所有轮）。
                    # 候选取自本页 col 年份列（collect_year_candidates）；EOF/中断 -> 保守中止 return 1。
                    try:
                        _ycands = getattr(sniff_url(url), "year_cands", []) or []
                    except Exception:
                        _ycands = []
                    if _ycands:
                        def _ask_auto_y(_p):
                            try:
                                return input(_p)
                            except (EOFError, KeyboardInterrupt):
                                return None
                        _yt, _yact = _ask_paper_type(_ycands, ask=_ask_auto_y, label="年份")
                        if _yact == "abort":
                            print(">>> [auto] 已取消（建批次前中止，未动游标/批次）。")
                            return 1
                _dl_delay = getattr(args, "delay", 0.5)
                if _dl_delay is None: _dl_delay = 0.5
                _dl_workers = getattr(args, "workers", None) or 6
                _cap = getattr(args, "limit", None)
                if chunk > 0 and not fids:
                    # ---- 分批流水线：每批最多 chunk 个目标，每轮独立 batch_id + 独立 manifest（不跨轮累积），
                    #      全部挂在一个 run_id 下；下载完一批立即接力跑阶段 2~7。----
                    _staged_multi = True
                    # 断点续跑游标：按"下载 URL"隔离（修：旧逻辑按 run_id 维护，换 URL 仍继承旧 offset
                    #   -> 新专题页前 N 个目标被误跳过，甚至 offset>该页 total 直接 0 PDF 空跑）。
                    # 同一 URL 续跑 -> 沿用该 URL 自己的 offset 与 run 分组；换 URL -> 天然 offset=0、新 run。
                    _url_ident = (url or f"topic::{topic}")
                    _prev_run, off, _ptotal, _pfp = (bc.db_url_cursor(_url_ident)
                                                     if bc is not None else (None, 0, None, None))
                    off = off or 0
                    _run_id = _prev_run or (bc.new_batch_id("run") if bc is not None else None)
                    # 明确提示"换 URL 不继承旧游标"：旧 current 指针批次若属另一 URL，本 URL 从 0 起。
                    if off == 0 and bid:
                        try:
                            _old_url = (bc.load_manifest(bid) or {}).get("url")
                        except Exception:
                            _old_url = None
                        if _old_url and _old_url != _url_ident:
                            print(f">>> [auto] 检测到 URL 变化，重置 offset=0（旧 url={_old_url}, 新 url={_url_ident}）")
                    _new_total = 0          # 本次运行累计"新下载"文件数（判定"本批无新 PDF"正常收尾 vs 有产出）
                    print(f">>> [auto] 分批流水线启动：每批最多 {chunk} 个目标；总量"
                          f"{'不限（下载器空结果即结束）' if not _cap else '上限 ' + str(_cap)}；run_id={_run_id}")
                    if off:
                        print(f">>> [auto] 续跑同一 URL：run_id={_run_id}，从该 URL 游标 offset={off} 继续"
                              f"（本 URL 已累计推进 {off} 个目标页）。")
                    r_idx = 0
                    while r_idx < 200:
                        if _cap is not None and off >= _cap:
                            print(f">>> [auto] 已达总量上限 {_cap}，分批下载结束。")
                            break
                        per = chunk if _cap is None else min(chunk, _cap - off)
                        r_idx += 1
                        # 每轮独立 batch + 独立 manifest（仅含本轮目标，绝不跨轮累积），归入同一 run_id。
                        _bid = _new_batch_for_url(url=url, name=getattr(args, "name", None),
                                                  topic=topic, run_id=_run_id)
                        if not _bid:
                            print(">>> [auto] 无法创建本轮批次（manifest），中止。")
                            return 2
                        bid = _bid
                        _run_bids.append(_bid)
                        print("\n" + _SEP)
                        print(f"[auto] 分批第 {r_idx} 轮（批次 {_bid}）：目标第 {off + 1}~{off + per} 个")
                        print(_SEP)
                        _dl = run_step_safe(
                            f"抓取下载(第{r_idx}批)", step_discover_download, topic, per,
                            _dl_workers, _dl_delay, url=url,
                            name=getattr(args, "name", None), batch_id=_bid, offset=off,
                            expected_fp=_pfp, paper_type=_pt, year_type=_yt)
                        rc_dl = _dl[0] if isinstance(_dl, (tuple, list)) and _dl else _dl
                        _dres = _dl[1] if isinstance(_dl, (tuple, list)) and len(_dl) > 1 else None
                        # rc=3（疑似解析失效）：硬中止，绝不落入 rc==2「正常空结束」分支。
                        if rc_dl == 3:
                            _print_rc3_banner("（有目标页但 0 PDF）：中止本次运行；未走「正常空结束」；"
                                              "未推进游标、未标 accepted（详见 logs/suspect_parse_*.json）")
                            return 3
                        # ---- 下载器内部触发的 offset 重置（历史偏移越界 / 目标集合指纹变化）：
                        # 同步本地 off，按 reason 判定是否需换 run_id（指纹变化 -> 新 run，避免跨专题集混同一 run）。
                        if isinstance(_dres, dict) and _dres.get("offset_reset_reason"):
                            _rsn = _dres.get("offset_reset_reason")
                            _ob = _dres.get("offset_before") if _dres.get("offset_before") is not None else off
                            _tt_now = _dres.get("total_targets")
                            off = int(_dres.get("offset") or 0)
                            print(f">>> [auto] 检测到游标越界/变化，已自动重置："
                                  f"reason={_rsn} offset_before={_ob} total_targets={_tt_now} offset_after={off}")
                            if _rsn == "fingerprint_changed" and bc is not None:
                                try:
                                    _run_id = bc.new_batch_id("run")
                                    # 将本轮刚建的新批次重新归入新 run（下一轮亦沿用）。
                                    bc.db_batch(_bid, run_id=_run_id)
                                    print(f">>> [auto] 目标集合指纹变化 -> 启用新 run_id={_run_id}，批次 {_bid} 已归入新 run")
                                except Exception as _e:
                                    print(f">>> [auto] [!] 新 run_id 切换异常（不阻断）：{type(_e).__name__}: {_e}")
                            # 本轮后以新 fp 为参考（避免下一轮重复触发同一重置）。
                            _pfp = _dres.get("target_fp") or _pfp
                        if rc_dl == 2:
                            # 下载器空结果（0 目标页 / 0 待下载 PDF）：这是"本批无新 PDF"的正常信号，不是失败。
                            # 停止分批下载循环，交由收尾统一按"正常结束（无需打包）"处理（退出码 0）；
                            # 绝不回 2、绝不再进后续阶段、绝不写"全流程未真正完成"。
                            # 区分三种情形：
                            #   a) offset_at_end：off > 0 且 off 已达本次 total_targets（自然抵尾）——不催重跑。
                            #   b) no_targets：本 URL 真的 0 目标（首轮且游标 0）。
                            #   c) 其余（中途 rc=2）：目标已全部覆盖，同样自然结束。
                            _tt_now2 = (_dres or {}).get("total_targets") if isinstance(_dres, dict) else None
                            try:
                                _tt_i2 = int(_tt_now2) if _tt_now2 is not None else None
                            except (TypeError, ValueError):
                                _tt_i2 = None
                            if off and _tt_i2 and off >= _tt_i2:
                                _empty_reason = "offset_at_end"
                                print(f">>> [auto] 第 {r_idx} 批：游标 offset={off} 已抵达末尾"
                                      f"（total_targets={_tt_i2}），分批下载正常结束。")
                            elif r_idx == 1 and off == 0:
                                _empty_reason = "no_targets"
                                print(">>> [auto] 第 1 批即为空（该 URL 未发现可下载 PDF）。")
                            else:
                                _empty_reason = "offset_at_end" if (off and (not _tt_i2 or off >= (_tt_i2 or 0))) else "no_targets"
                                print(f">>> [auto] 第 {r_idx} 批无新目标（游标 offset={off} 已到末尾 / 目标已全部覆盖），分批下载结束。")
                            # [机制2a 默认关] 下载返回空但游标未达 total（疑似未跑完）时强制询问；abort->保留 pending_accept 退出1，skip->维持原 break。
                            if getattr(args, "ask_on_empty", False):
                                _tt = (_dres or {}).get("total_targets") if isinstance(_dres, dict) else None
                                if _tt is None and bc is not None:
                                    try: _tt = bc.db_url_cursor(_url_ident)[2]
                                    except Exception: _tt = None
                                if _tt is not None and off < int(_tt):
                                    _ch = ask_continue_or_abort(f"下载返回空但游标 {off}/{_tt} 未达 total，疑似专题未跑完：中止还是按空结束继续？", ["abort", "skip"])
                                    if _ch == "abort":
                                        print(">>> [auto] 用户选择中止，保留 pending_accept（未误标 accepted）。")
                                        return 1
                                    print(">>> [auto] 用户选择按空结果继续（走原分批结束路径）。")
                            # 真·空收尾：落盘空跑报告 + break（交给收尾按"正常结束"处理）。
                            _report_empty_run(url, topic, off, _dres, _bid,
                                              reason=_empty_reason,
                                              suggested_action=("none" if _empty_reason == "offset_at_end" else "rerun"))
                            # 同步一次游标的 total_target/target_fp（自然抵尾/目标变化后下次读到新鲜值）。
                            if bc is not None:
                                try:
                                    bc.db_url_set_cursor(_url_ident, off, run_id=_run_id,
                                                         total_target=_tt_i2,
                                                         target_fp=(_dres or {}).get("target_fp") if isinstance(_dres, dict) else None)
                                except Exception:
                                    pass
                            break
                        # rc 分流（False=用户跳过 / 非 0=未成功 / 0=继续）；rc2/3 已在上方处理。
                        _act, _code = _dl_action(rc_dl, off, _dres, _bid, True, url, topic)
                        if _act == "return": return _code
                        _total = (_dres or {}).get("total_targets") if isinstance(_dres, dict) else None
                        off += per
                        # 游标夹紧：目标总数已知时累计游标不得越过 total（防 168>165 过冲；未知则维持原推进量，行为不变）。
                        if _total is not None:
                            try: off = min(off, int(_total))
                            except (TypeError, ValueError): pass
                        # 推进并持久化游标（按 URL，含目标总数 + 目标集合 fp）：本批后续阶段接力前即落库，中途崩溃也可续跑。
                        if bc is not None:
                            try: bc.db_url_set_cursor(_url_ident, off, run_id=_run_id, total_target=_total,
                                                      target_fp=(_dres or {}).get("target_fp") if isinstance(_dres, dict) else None)
                            except Exception: pass
                        _newn = (_dres or {}).get("downloaded") if isinstance(_dres, dict) else 0
                        try: _new_total += int(_newn or 0)
                        except (TypeError, ValueError): pass
                        _round_files = (_dres or {}).get("pdf_count") if isinstance(_dres, dict) else None
                        # 三数字：本轮新增（新下载）/ 历史累积（游标覆盖目标数）/ 实际待处理（本轮 manifest 带入后续阶段的文件数）。
                        print(f">>> [auto] 第 {r_idx} 轮统计：本轮新增 {_newn} / 历史累积 {off} / "
                              f"实际待处理 {_round_files if _round_files is not None else '本批'} 个文件（批次 {_bid}）")
                        _selfcheck(f"post-download r{r_idx}",
                                   f"target_pages={(_dres or {}).get('target_pages') if isinstance(_dres, dict) else '-'} "
                                   f"pdf_count={_round_files} downloaded={_newn} "
                                   f"failed={(_dres or {}).get('failed') if isinstance(_dres, dict) else '-'} total_targets={_total}")
                        # 空跑保护：本轮新下载为 0 => 本批无新文件，跳过后续 wm/rename/ocr/qc/ads/pack（不再空扫），仅推进游标探测下一批。
                        # [阶段3 默认关] --force-rerun-failed：新下载 0 但本批有失败文件 -> 不跳过，继续重跑后续阶段。
                        _fail = (_dres or {}).get("failed") if isinstance(_dres, dict) else 0
                        _rerun_fail = bool(getattr(args, "force_rerun_failed", False)) and _fail
                        if _newn == 0 and not _rerun_fail:
                            print(f">>> [auto] 第 {r_idx} 轮新下载 0（已存在为主，本批无新文件），跳过本轮后续阶段"
                                  + (f"（失败 {_fail}）" if _fail else "") + "，继续探测下一批。")
                        else:
                            if _newn == 0 and _rerun_fail:
                                print(f">>> [auto] 第 {r_idx} 轮新下载 0，但 --force-rerun-failed 启用且本批失败 {_fail} -> 继续重跑后续阶段。")
                            _con = db()
                            try: _rc = _run_stages_auto(_con, _bid, r_idx)
                            finally: _con.close()
                            if _rc is not None:
                                print(f">>> [auto] 分批第 {r_idx} 轮后续阶段未通过，中止。")
                                return 1
                            _selfcheck("post-pack", f"batch={_bid} zip_exists={_batch_zips_exist(_bid)}")
                        # 结束条件：已知目标总数且游标已覆盖全部（offset>=total）-> 正常收尾，不再空转下一批。
                        if _total and off >= int(_total):
                            print(f">>> [auto] 游标 {off} 已覆盖全部 {_total} 个目标，分批下载结束。")
                            break
                    else:
                        print(">>> [auto] [!] 分批轮数达 200 上限，停止（防死循环）。")
                else:
                    # 旧行为（单轮全量）：一次性下载后统一跑阶段 2~7
                    _dl = run_step_safe(
                        "抓取下载", step_discover_download, topic,
                        getattr(args, "limit", None),
                        _dl_workers,
                        _dl_delay, url=url, name=getattr(args, "name", None),
                        paper_type=_pt, year_type=_yt)
                    rc_dl = _dl[0] if isinstance(_dl, (tuple, list)) and _dl else _dl
                    # rc=3（疑似解析失效）：硬中止，绝不落入 rc==2「正常空结束」分支。
                    if rc_dl == 3:
                        _print_rc3_banner("（有目标页但 0 PDF）：中止本次运行；未走「正常空结束」；"
                                          "未推进游标、未标 accepted（详见 logs/suspect_parse_*.json）")
                        return 3
                    # rc 分流与分批路径共用（False=用户跳过 / 非 0=未成功 / 2=落盘空跑报告后 return 0，
                    # 空结果为正常信号：不进后续阶段、不报"未完成"，退出码 0）。
                    _act, _code = _dl_action(rc_dl, None, None, bid, False, url, topic)
                    if _act == "return": return _code
                    bid = bc.current_batch_id() or bid
                    if not bid:
                        print(">>> [auto] 下载后未找到批次，中止。")
                        return 2
                    print(f">>> [auto] 下载完成，批次 {bid}")
        # ---- 路径回退（打回重跑：manifest 路径退回起点输入目录）----
        if start != "download" and not dry:
            _rev, _miss = _revive_manifest_paths(bid, fids, start)
            if _rev: print(f">>> [auto] 已回退 {_rev} 个文件的路径到起点阶段输入目录")
            if _miss:
                print(f">>> [auto] [!] {len(_miss)} 个文件在输入目录找不到可处理的文件：")
                for _m in _miss[:8]: print(f"        - {_m}")
                print("        中止（文件可能已被移动/删除；可改用更晚的 --start-stage）。")
                return 2
        # ---- 打回重跑标记（--force 且点名了文件）----
        if force and fids and start != "download":
            for _fid in sorted(fids):
                bc.db_step(bid, _fid, STEP_ACCEPT, bc.ST_REJECTED, error=f"打回重跑（起点 {start}）")
            print(f">>> [auto] 已复位 {len(fids)} 个文件的验收状态 -> rejected")
        # ---- 阶段 2~7：分批模式已在下载循环内逐批跑完；否则此处统一跑一轮（含 dry-run 预告） ----
        if not _staged_multi:
            con = db()
            try: _rc = _run_stages_auto(con, bid, 0)
            finally: con.close()
            if _rc is not None:
                return 1
        # ---- 分批 + topic：各轮 wm..ads 已在下载循环内逐批跑完、打包已延后；此处按 run 统一整专题打包一次 ----
        #      （dry 仅预览；_staged_multi=False 时已在上面就地打包，不重复）。----
        if _staged_multi and _pack_mode() == "topic":
            _con = db()
            try:
                _rp = step_pack(_con, None, None, dry=dry, run_id=_run_id)
            finally:
                try: _con.close()
                except Exception: pass
            if not dry and _new_total > 0 and not _rp:
                print(">>> [auto] [!] 整专题打包未产出 zip。")
                return 1
        if dry:
            print("\n>>> [auto] (dry-run) 以上为将执行的阶段，未实际改动。")
            return 0
        # ---- 有效批次集合（分批=本次各轮 / 单轮=[bid]）----
        if _staged_multi and _run_bids:
            _all_bids = list(_run_bids)
        else:
            _all_bids = [bid] if bid else []
        _selfcheck("pre-accept", f"bids={len(_all_bids)} force_full_run={getattr(args, 'force_full_run', False)} (verify 默认关->不调)")
        # ---- 机制1（--force-full-run，默认关）：跑不完不许结束。核验 游标到 total + 每批文件达验收步 + zip 数==验收数；
        #      任一不满足 -> 不标 accepted（保留 pending_accept 供人工核查）+ 打印全部 reasons + 退出码 1。
        #      前移至"零新增 return 0"之前，使 --force-full-run 也能拦"零新增但游标未达 total"的未跑完场景；
        #      默认关时本块恒不执行，零新增 return 0 与所有既有路径行为完全不变。----
        if getattr(args, "force_full_run", False):
            _vrun = _run_id if (_staged_multi and _run_id) else (bc.db_run_of_batch(bid) if (bc and bid) else None)
            _vurl = _url_ident if _staged_multi else ((bc.load_manifest(bid) or {}).get("url") if (bc and bid) else None)
            _vc_ok, _vc_reasons = verify_run_completeness(_vrun, _vurl)
            if not _vc_ok:
                print(_SEP)
                print(">>> [完整性] --force-full-run 校验未通过：判「专题未跑完」，拒绝自动验收（保留 pending_accept）。")
                for _rs in _vc_reasons:
                    print(f"    - {_rs}")
                print(">>> [完整性] 请人工确认后处理：去掉 --force-full-run 以现有产出结束，或续跑补齐剩余目标后重跑。")
                print(_SEP)
                return 1
        # ---- 空结果正常结束：本次既无"新下载"、本轮也未产出任何 zip -> 视为"本批无新 PDF"，退出码 0。
        #      绝不进后续打包/验收、绝不写"全流程未真正完成"（修问题 2；旧 pending_accept 不回滚也不新置）。----
        if _new_total == 0 and not ppack._LAST_PACK_ZIP and not ppack._LAST_PACK_ZIPS:
            print(">>> [auto] 本批无新 PDF，正常结束（无需打包）。")
            print(">>> [auto] 本次零新增，未产出本批次 zip，不触发自动验收（批次保持原状态）。")
            print(_SEP)
            return 0
        # ---- 收尾：最终 zip 确认 + 有效批次（manifest 确有文件者）标「待验收」；空批次只记录不推进（修问题 3/5）----
        if not _final_zip_check(): return 1
        _valid_bids = [b for b in _all_bids if b and has_targets(b)]
        for _b in _all_bids:
            if _b and not has_targets(_b):
                print(f">>> [skip] 空批次（manifest 无文件），不标待验收/不验收：{_b}")
        for _b in _valid_bids:
            try: bc.set_batch_status(_b, "pending_accept", note="全自动流程完成，等待人工验收")
            except Exception: pass
        # ---- 自动验收（AUTO_ACCEPT，默认开）：pack 成功且确认 zip 真实存在才 accept；无 zip 只告警 ----
        _auto_ok = 0
        if AUTO_ACCEPT:
            for _b in _valid_bids:
                if not _batch_zips_exist(_b):
                    # 没有 zip 就不验收（防对未打包/空批次误置 accepted）；保留 pending_accept 供人工核查
                    print(f">>> [auto] [!] 批次 {_b} 未见已生成的 zip，跳过自动验收（保留 pending_accept）")
                    continue
                try:
                    _ar = bc.batch_accept(batch_ids=[_b], operator="auto")
                except Exception as _ae:
                    _ar = {"ok": False, "accepted": 0,
                           "errors": [f"{type(_ae).__name__}: {_ae}"]}
                if _ar.get("ok"):
                    _auto_ok += int(_ar.get("accepted") or 0)
                    print(f">>> [auto] 自动验收完成：{_ar.get('accepted', 0)} 个文件 -> accepted，批次 {_b}"
                          + (f"（跳过 {_ar.get('skipped')}）" if _ar.get("skipped") else ""))
                else:
                    # 自动验收失败：绝不删 zip、绝不回滚打包；保留 pending_accept 供手动批量验收
                    print(f">>> [auto] [!] 批次 {_b} 自动验收失败（保留 pending_accept，可在看板或菜单 [e] 手动批量验收）："
                          f"{'; '.join(_ar.get('errors') or [])}")
        print(_SEP)
        if AUTO_ACCEPT and _auto_ok:
            print(f">>> 全流程跑完并已自动验收：{_auto_ok} 个文件 -> accepted（zip 见「文库资源_打包文件」）")
        elif AUTO_ACCEPT:
            print(">>> 全流程跑完（本轮无待验收文件可自动验收；如需人工归档请到「文件看板」）")
        else:
            print(">>> 全流程跑完，请去「文件看板」进行人工验收")
        if _staged_multi:
            print(f">>> 本次运行 run_id={_run_id} 共 {len(_valid_bids)} 个有效轮次批次"
                  + (f"已自动验收（accepted）" if AUTO_ACCEPT and _auto_ok else "已更新为「待验收」（pending_accept）"))
        else:
            print(f">>> 批次 {bid} " + (f"已自动验收（accepted）" if AUTO_ACCEPT and _auto_ok else "状态「待验收」（pending_accept）"))
        print(_SEP)
        return 0
    except KeyboardInterrupt:
        print("\n[!] [auto] 已中断（状态已落库，可再次运行续传）。")
        return 1
    except SystemExit as e:
        print(f"\n[!] [auto] 步骤触发退出（code={getattr(e, 'code', '?')}）。")
        return 1
    except Exception as e:
        print(f"\n[ERR] [auto] 执行失败：{type(e).__name__}: {e}")
        print(fatal_hint("自动流程", e))
        import traceback
        # 全量 traceback 落盘（闪退专项排查：每次崩溃都要留完整堆栈供事后定位，不凭猜）
        _logp = _write_auto_crash_log(exc_obj=e, tag="run_auto")
        if not _logp:
            # 落盘失败兜底：至少控制台输出完整 traceback（不只最后 8 行）
            for line in traceback.format_exc().strip().splitlines(): print("      " + line)
        else:
            for line in traceback.format_exc().strip().splitlines()[-8:]: print("      " + line)
        return 1
    finally: pcore.AUTO_MODE = False
