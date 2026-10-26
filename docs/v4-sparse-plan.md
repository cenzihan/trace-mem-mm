# V4-Sparse：完整来源库上的三视图检索

2026-10-25，在读取本轮 Hard31 问题进行检索之前确定。该版本是本地研究候选，未替换默认 Add/Search 系统。

## 为什么做这一版

现有 Mem-Gallery 的 C 在两个场景的 Recall@20 略高于 A，但第一场景前排排序退化，第二场景 reader 仅 10 题且 C=5/10、A=6/10。当前没有稳定的端到端收益证据。ATM N0 证明完整来源库上的 BM25 可以运行，却没有 dense 或 reader 结果。下一步先检验表示粒度与来源聚合是否值得继续投入。

V-Mem 的细粒度匹配与原生来源聚合是设计参考；本实现为独立编写的 sparse 适配，不使用其 MiniLM/SigLIP，也不声称复现原版。flat 与细视图包含重叠内容，三通道不是三份独立证据。

## 冻结配置

- 来源固定为 ATM release `78e826dc07e97466b2f54443831ef9a83ab8b27c` 的已有 SGM/邮件文本，完整 11,034 条原始来源，保留全部干扰项。
- flat：逐字复用 N0 formatter 的完整文本，构成对照与一个检索通道。
- summary：图片/视频 `short_caption`，为空则 `caption`；邮件 `short_summary`。
- detail：图片/视频 `caption`、OCR、tags、timestamp、location_name；邮件 detail、timestamp。
- 空字段不补造文字。没有细视图的来源仍保留在 flat 中。detail 按既有 tokenizer 的 512 个词项窗口、384 步长分块，保留来源映射，不丢长文。
- 每通道 BM25 `k1=1.2, b=0.75`。先计算全部匹配块，再按 native parent 取最高分，然后各取最多 100 个不同来源。完全相同的 parent/lane/text 只索引一次。
- 三通道等权 RRF，常数 60；最终最多 100 个不同 native ID，同分按 ID 排序。无查询生成、图扩展、reranker、dense、模型调用或动态选权。

## 一次性对照

完整来源和源码 hash 冻结后，再解析 Hard31 的 question-only 投影。每题均运行 flat 与 V4；全部排名写完并固定 hash 后，独立 scorer 才读取 gold。主指标是 Fractional Recall@20，报告 @1/5/10/20/50/100、Any-hit、全证据覆盖、配对差值、胜平负和 bootstrap 区间，不只挑有利指标。

Hard31 先前已由程序做 schema 和重叠检查，但未发现性能运行或据此改版的记录；它不是从未读取的盲测。其 172 个不同 gold 来源中有 59 个也出现在 regular gold；四份 NIAH 使用相同 Hard 问题。此轮应称“同库新题首次性能对照”，不称跨库独立泛化。

不重复 N0 regular1013 来选择参数。看到 Hard 结果后，不调整本版字段、权重、分块或融合再重跑；如失败，保留失败结论。结果只说明来源检索，不是原图理解、reader 正确率或官方榜单分数。

## 可追溯性和资源

`runtime/atm-v4-hard/FROZEN_CONFIG.json` 固定数据和源码；原始排名及逐题评分仅保存在忽略的 runtime。报告只给聚合结果。来源、旧结果和真实预算/Smoke账本运行前后校验 hash。预算为 0 provider 请求、0 元、CPU 本地处理，不新建付费 guard。

代码：`agent_memory/parent_sparse.py`；运行：`scripts/run_atm_v4_hard_local.py`；评分：`scripts/score_atm_v4_hard_isolated.py`。两个入口都拒绝覆盖已有运行。没有数据的导出仓库只运行合成单元测试，不能伪造基准复现。

## 接下来如何决策

若 V4 未显示稳定收益，保留 flat 作为可靠控制，不在这 31 题上继续调参。下一阶段优先补充共享 text-embedding-v4 的 dense-only 与固定 hybrid 对照，独立分析排序和证据选择；全库源向量只构建一次。模型付费和来源向 provider 的传输属于另一个明确、有上限的实验范围。

若后续做严格从原媒体构建，则另建 gpt-4o-mini 视觉表示与来源标识；不能把 Qwen SGM 缓存改名为严格产物。真正跨场景验证需先登记未用于方法选择的完整 Mem-Gallery 场景，并保持统一预算。
