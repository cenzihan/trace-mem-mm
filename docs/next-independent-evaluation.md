# 下一次独立评估计划（未执行、未新增付费授权）

本轮只准备计划，不发 provider 请求，也不重开已经 complete 的 84 次 guard。
当前剩余共享预算 483.711685 元。以下预占为保守上界，不是实际账单报价。

## 数据与先后顺序

采用固定 ATM regular 数据 revision`78e826dc07e97466b2f54443831ef9a83ab8b27c`，
不复用已分析的 Mem-Gallery 题追分。预期完整 corpus：3759 图、533 视频、6742 邮件，
共 11034 个原生 artifact。图/视频数量已核清单；邮件条目数来自数据卡，payload 尚未核实。
不造对话/session，不只索引 gold 相关记录，不用 frame/view 数增加 native 证据分母。

| 阶段 | 记忆库与题目 | provider 请求上界 | 累计预占上界 | 当前状态 |
|---|---|---:|---:|---|
| N0 本地 SGM/原生 loader 及 BM25 检查 | 完整 11034 来源预期，regular1013 题 | 0 | 0 元 | 源 payload 尚缺，准备未运行 |
| N1 三组受控 SGM 研究比较 | 同完整库，预先固定 20 题 | 3407 | 188.75 元 | 需隐私/来源/规模核验及单独阶段批准 |
| 可选 N1 原图 reader 对照 | 相同 20 题/三组，最多 120 个图或帧 occurrence | 另 72 | 另 18 元；连 N1 共 206.75 元 | 真实媒体/音轨/解码未就绪，不自动附加 |
| strict-from-raw 默认 6 帧全库 | 3759 图＋3198 视频帧，仅视觉 2319 批 | 至少 2319 | 仅视觉 579.75 元 | 已超过剩余预算，不启动 |

N0 只需作者 SGM 两份 JSON 和邮件 bundle，预计 34,311,246 字节；不是下载全部 3.33GB 原媒体。
这些描述/邮件仍可能含私人信息，先做本地来源/隐私审阅，不能上传、打印原文或进 Git。
本轮没有下载这些 payload，也没有 N0 性能分数。

N1 选择规则已在`reports/next-independent-evaluation-plan.json`登记：
先按原始题序取每个 qtype 首题，再补最早未选题，共 20 题；只读 public question metadata，
不读 answer/evidence_ids 选题。具体入选索引仅保留在原实验项目的私人计划报告中。
regular1013 与 Hard31 不是随机拆分的独立 holdout；四份 NIAH 共用同组 Hard 题。

## 对照条件与共同预算

1. 原生 SGM flat BM25：本地词法参照，保留全库。
2. ATM-RAG text-embedding-v4 适配：joint source view 的 dense 检索。
3. V-Mem-style 适配：细粒度 source views、BM25/dense/RRF、原生 parent 聚合，
   每题最多一个 transient gpt-4o-mini anchor；未实现 V→V SigLIP 通道，不称等价原版。

三组使用相同原生内容和 privacy 条件，全局候选上限 100、返回 20 个 native artifacts、
8192 文本 token/24000 字节、最多 2 图、输出 768token。N1 主条件实际原图数为 0，
是**外部 SGM 研究条件**，不能称真实原图 reader 或学术榜从 raw 全程指定模型条件。
可选原图条件另冻结图/帧选择及共同实际预算，禁止用 gold 挑图。
ATM list_recall 回答需原生 evidence ID，引用别名必须映射回它，不能用内部 unit ID 冒充答案。
768 输出上限为 ATM 原生 ID 列表任务预先设定，未回头修改 Mem-Gallery 结果。

raw source/gpt-4o-mini 表示/text-embedding-v4 才是另外的 strict-from-raw 条件。
预计算 Qwen3-VL-2B SGM 与作者邮件摘要的生成来源必须披露；不能改名掩盖。
所有实际新 LLM 组件仍固定 gpt-4o-mini，实际 embedding 固定 text-embedding-v4、256 维。
Qwen reranker 留待独立消融，当前计划关闭；不训练、不使用工业榜替代模型。

## N1 请求与费用上界

同一 joint view 由 BM25 及 ATM 适配共享，另加最多两个 fine views：
最多 33102 个 source embedding 文本，每 10 条一批，3311 请求。
20 个 query 缓存 2 请求；最多 20 个 anchor 编码 2 请求。
因此 embedding 最多 3315 请求，每次预占 0.05 元，共 165.75 元。
anchor 最多 20、三组 reader60、每批 5 条的代理 judge12，共 92 个 LLM 请求，
每次 0.25 元，共 23 元；总 3407 请求、188.75 元。

上界假设所有原生 source 可在约定的最多三个 view 和单条 8192 字节准入内表示。
任何缺记录、重复 ID、超长或源生成模型不明确，都先阻断并重做计划；不静默截断。
付费前必须用实际 payload 核验计数，不把预期 11034 当已验证事实。

价格沿已保存官方快照：embedding0.5 元/百万输入 token；gpt-4o-mini
USD0.15/0.60 每百万输入/输出 token。预算采用 10 元/USD 保守假设，非市场汇率。
每 embedding 请求最多 10×8192 输入 token，费用上界 0.04096 元＜0.05 元；
每 LLM 请求按 128000 输入＋768 输出计，预算上界 0.196608 元＜0.25 元。
真实运行前重新核实价格/服务，不能用过期报价放行。

## 必须先解决

- 源文件 hash、全量 ID/条目数、模型生成声明及 CC BY-NC/归属核查。
- 完整库的外部编码隐私审阅，尤其邮件、地点/GPS 和人物；不能删掉敏感 distractors 后声称全库。
- 视频真实解码与音轨策略；当前帧适配只 fake-process 测过，有音轨明确阻断。
- source-only parser/parent 预算契约已实现；dense index、anchor、native reader orchestration 仍未接通。
- N1 需要新批准的独立有界 guard；现有类仅支持此前小额 scope，不能绕过上限或复用旧 guard。

评估报告继续分 fractional/any-hit/all-evidence、图片证据、跨来源完整性、拒答、
原生 EM/F1 适用项和明确标注的代理 judge，另列费用/延迟/blocked case。
这份计划不承诺新分数，也不把看到的测试错误当 prompt 调参样本。
