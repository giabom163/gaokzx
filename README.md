# gaokzx 文库采集与高保真双层 PDF 引擎（v2.1）

> **系统定位**：高考真题与教学试卷全流程工业级流水线，内置自研**双层高保真可检索 PDF 引擎**，
> 彻底攻克开源 PDF 注字工具普遍存在的文本框缩放漂移、试卷多栏乱序穿插与灰斑污染等硬伤。

### 🌟 双层高保真可检索 PDF 引擎核心特性（v2.1）
- **Recursive XY-Cut 阅读序拓扑重构**：自适应双栏、多栏试卷自动分检，消除跨栏划词/复制穿插。
- **微积分级仿射对齐**：基于实际文本度量进行 `sx` 仿射拉伸注入，消除字号试错循环，单页耗时从 350~500ms 降至 **70ms**（提速 5 倍）。
- **PDF Text Rendering Mode 3 真隐形**：标准无描边无填充隐形文字层，彻底消除假透明导致的底色灰斑。
- **中西文自适应基线 & 填空题防畸变**：西文 0.78、中文 0.83 动态基线，填空题下划线畸变保护，同行相邻块自动分词。

全流程**八阶段流水线**：采集 → 去水印 → 重命名 → OCR 补文字层 → 质控 → 清广告 → 打包 → 验收。
（支持 `GAOKZX_CLEAN_FIRST=1` 先净图后注字双模式新架构）

## 快速开始

```bat
python pipeline.py                          :: 无参数 = 交互菜单（推荐入口）
python pipeline.py --step check-env         :: 环境自检（缺依赖会提示一键安装）
python pipeline.py --auto --url "<专题页URL>"  :: 一键全自动（无人值守跑完到待验收）
python pipeline.py --step watermark,rename,ocr,qc,pack   :: 单步组合（可 --dry-run 预览）
```

看板服务：`D:\work\dashboard\run_dashboard.cmd`（127.0.0.1:8765，文件表视图 + 一键重跑/验收）。

## 架构（v2：8 拆 1 留，2026-10-22 定型）

pipeline.py 已从 6662 行单文件拆为「一个入口 + 八个领域模块」，主控仅剩 444 行：

| 模块 | 行数 | 职责 |
|---|---|---|
| **pipeline.py** | 444 | 只留：解释器守卫 + import 区 + CLI 参数注册 + main 分发 |
| pcore.py | 853 | 公共核心：初始化/STAGE 常量/db()/mark/log/锁族/文件枚举/交互(_ask 族)/依赖自检/步骤安全包装 |
| pdownload.py | 566 | 网址嗅探（sniff_url 族）+ 下载调度（step_discover_download 族）+ 隐形水印清理调度（step_ads） |
| pwm.py | 808 | 去水印算法族（图层/像素双模式）+ step_watermark |
| prename.py | 768 | 文件名解析/改名/回滚/冲突报告 |
| pocr.py | 905 | OCR 内存/显存护栏 + step_ocr(474) + step_clean_ocr_dirty + step_qc |
| ppack.py | 748 | 打包双模式（step_pack_legacy/topic + step_pack 分派）+ step_accept + reconcile_round + step_fix_dst |
| padmin.py | 468 | 数据维护：step_prune_db/step_cleanup/step_migrate_done/step_export_excel/show_status |
| pflow.py | 678 | 交互菜单（MENU/run_wizard/flow_* 族）+ 步骤安全包装 |
| pflowauto.py | 923 | 自动编排：run_auto/_run_auto_impl/verify_run_completeness + 断点续跑族 |
| batchcore.py | 936 | 批次 manifest + 状态表（50 函数，4 文件共享） |
| bcaccept.py | 207 | 批量验收（batch_accept，pending_accept→accepted 唯一合法迁移） |
| gaokzx_download.py | 923 | 采集下载子进程入口（CLI 子命令，被 pdownload 拉起） |
| gaokzx_parse.py | 491 | 试卷名解析（parse_pdf_name/build_norm_name） |
| clean_bottom_ads.py / scan_bottom_ads.py | 762/692 | 广告页清理/扫描（独立 CLI，报告-流水关系） |
| wm_worker.py / wm_ocr_batch_worker.py | 40/169 | 去水印/OCR 批处理子进程入口（引擎初始化一次/批） |

依赖分层（无循环）：

```
batchcore ─→ pcore ─→ pdownload / pocr / pwm / prename / ppack / padmin ─→ pflowauto ─→ pflow ─→ pipeline.py
                     └→ bcaccept（batchcore 末尾 re-export 保持 `import batchcore as bc` 兼容）
```

**跨模块可变全局纪律**（拆分的教训，2026-10-22 立）：定义方 = 读写方所在模块，
跨模块写一律经模块属性访问，**禁止 from-import 快照**：
`DB/ONLY_FILE_IDS/ONLY_SUBDIR/AUTO_MODE → pcore`；`_LAST_PACK_ZIP(S) → ppack`；`FORCE_MISSING → pflowauto`。

看板（D:\work\dashboard\，独立进程，未并入本仓库版本管理）：`dashdata.py`(784 数据层) +
`dashtask.py`(331 任务队列) + `dashboard.py`(435 HTTP 层)。

## 流水线八阶段

| # | 阶段 | 输出目录 | 说明 |
|---|---|---|---|
| 1 | download | 文库资源 | 专题页→详情页→PDF；空结果 rc=2 立即中止 |
| 2 | watermark | 文库资源_去水印后 | 图层水印无损删除 / 像素水印局部块修补（保文字层） |
| 3 | rename | 文库资源_去水印后 | 规范重命名；重名自动 _1/_2，**默认 dry-run 只出清单** |
| 4 | ocr | 文库资源_去水印后 | PaddleOCR 补透明文字层（GPU 优先/CPU 物理核预算） |
| 5 | qc | 文库资源_去水印后 | 零字节/损坏/无文字层 |
| 6 | ads | 文库资源_已清广告 | 隐形水印清理（重扫描生成 reports/bottom_ads_<bid>.json） |
| 7 | pack | 文库资源_打包文件 | topic 模式一专题一 zip（默认）；batch 模式每批一 zip |
| 8 | accept | 文库资源_completed | 验收归档（DB 位 + 物理归档两步） |

## 测试（改动前必跑）

```bat
python test_smoke_readonly.py    :: 42 断言：CLI 参数哨兵/模块链/符号完整性/pcore 纯函数+
                                    可变全局过滤/parse_pdf_name 样本/batchcore 隔离库全流程
python test_topic_pack.py        :: 18 断言：打包双模式 + _batch_zips_exist run 级回退 + 分派路由
```

纪律：**不联网、不写生产库**——smoke 进程启动即设 `GAOKZX_TEST_ROOT` 把 DB/_批次/logs
全重定向临时目录；batchcore 操作全部在隔离库上跑。

## 状态机与续跑

- per-file/per-step 状态记 `pipeline.db`；`_skip_ok` 三条件（DB ok + 产物存在 + sha 一致）跳过
- `reconcile_round` 每轮 DB↔磁盘双向核对（done 但产物缺→回滚重处理）
- `--force-missing` 只重跑“DB=done 但磁盘缺失”；`--force-scope new` 保护旧库
- 看板单文件调度：`--file-id F0001`（pcore.ONLY_FILE_IDS 收窄 scan_pdfs）
- 新专题保护：`--url` 自动限定 `--only-subdir`（pcore.ONLY_SUBDIR 收窄 iter_pdfs）

## v2 变更史（2026-10-21 ~ 10-23）

| 日期 | 事项 |
|---|---|
| 10-21 | 8 拆 1 留执行完毕（prename/pwm → pdownload/pocr/ppack/padmin/pflowauto）；修 `--file-id`/`--only-subdir` 拆分回归（from-import 快照失效，改模块属性访问） |
| 10-22 | 补只读冒烟测试网（42 断言）；dashboard 三拆（dashdata/dashtask/dashboard，服务快照 diff 8/8 逐字节一致）；删 5 项死代码（-78 行） |
| 10-23 | 总量精简终审：相似度矩阵/SQL 去重/常量双定义/5-gram/单调用点全查，确认代码无水分（11879 行 = 代码 79% + 语义注释 15% + 空行 7%） |

## 已知遗留（都是有意保留或待议，非缺陷）

- `gaokzx_download.py` 923 行：过程式主循环（run 273 + main 93），子进程入口内聚，不拆
- **AD_SIG/FOOTER_PROMO_KEYWORDS 签名表漂移**：pwm 版（3/2 项）与 scan 版（4/4 项）不一致，
  scan 是超集。改广告签名时**两处都要改**（未来考虑统一到 pcore）
- `reports/ads_split_by_year.json`（20KB）用途不明，待定去留
- `batchcore.py`(936)/`dashdata.py`(784) 密度高，无重复，不建议再拆

## 备份命名约定

pipeline.db 备份统一格式：`pipeline.db.bak_<用途>_<YYYYMMDD_HHMMSS>`
示例：`pipeline.db.bak_offsetfix_20260926_225524`
用途标识：offsetfix / cleanfix / migration / packmerge / layout ...

## 最简全流程测试（离线，分钟级）

命令（--batch-id 换成本地任一历史批次）：

```bat
python pipeline.py --auto --start-stage pack --batch-id <bid> --test-root D:\work\_test_min --chunk 0
```

前置：_test_min 内需有 pipeline.db 副本、_批次\（manifest）与打包源阶段目录的对应专题子目录。
用途：不联网、不跑 OCR、GPU 零占用，验证整专题打包链路（pack → accept）。
回退：`GAOKZX_PACK_MODE=batch` 即旧行为（每批一 zip）。

## git 提交规范

- Conventional Commits（`feat/fix/refactor/chore/test/docs: 范围 说明`），一个逻辑改动一个提交
- `.gitignore` 已排除：`*.db`（pipeline.db 含 6462 条生产记录，**绝不入账**）、`reports/`、`logs/`、`__pycache__/`、`*.bak_*`
- 提交前必跑两个测试；提交信息用 ASCII 英文（历史提交有中文乱码教训：aae0aab 等）
