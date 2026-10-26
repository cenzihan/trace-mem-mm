# N0 后的共享缓存与严格指定模型计划（未执行）

本文件只做下一阶段设计，不新增 provider 调用或 guard。
不再按“20 个 reader 问题”创建三套全库向量。先区分全库冷启动与后续查询成本，
把表示构建与缓存共享给所有对照方法；候选库始终保留全部 11,034 个来源。

## 已有与没有

- 已有：官方 SGM/邮件文本、本地 BM25 全库索引、1,013 题检索及隔离评分。
- 没有：任何 ATM text-embedding-v4 向量、gpt-4o-mini 新视觉抽取、raw 媒体解码。
- Mem-Gallery 向量不是 ATM 向量，不能只因维数相同就跨 namespace 复用。
- 旧 Qwen SGM 及其将来的向量不能冒充 strict-from-raw 的产物。

缓存键至少包含 dataset revision、namespace、表示版本、规范文本 hash、模型和维度。
本次实际 flat 文本只需 11,035 个块，text-embedding-v4 每次最多 10 条，冷启动 1104 请求。
这是一份可供多方法与全套问题共享的索引，不是每个 reader、每种视图重复付一次。

## 分层计划

| 条件 | 冷启动/问题范围 | 请求上界 | 累计预占上界 |
|---|---|---:|---:|
| 外部 SGM、共享一份全库文本向量 | 1104 source＋105 query＋62 reader＋13 proxy judge；regular1013＋Hard31 检索，Hard31×2 reader | 1284 | 79.20 元 |
| 有效同条件缓存已存在后的 31 个新问题 | 0 source＋4 query＋62 reader＋13 judge | 79 | 18.95 元 |
| 只在完整 BM25 库的 Top100 内作 late embedding | 20 题、source union 最多 2000；202 embedding＋40 reader＋8 judge | 250 | 22.10 元 |
| 重新生成严格指定模型视觉表示，完整库共享 | 另 870 视觉请求＋上述共享全库文本/查询/reader | 2154 | 296.70 元 |

前三项仍使用作者 Qwen SGM，是外部预处理研究条件，不是严格从 raw 条件。
late embedding 保留完整词法来源库，但不是全局 dense search，受第一阶段召回上限约束，
不能伪称全库密集检索。它是廉价的独立消融选项，不是当前已执行的改分。
Hard31 与 regular 题目不同但共享来源及部分 gold 来源，只能叫新题目复核，不称跨库独立泛化。

## 严格视觉表示如何减少请求

保持 3759 图、533 视频各 6 帧，共 6957 图像输入，**不删视频或干扰来源**。
拟将独立输入以最多 8 图一请求、detail=low 批处理，ceil(6957/8)=870 请求；
而先前 3 图/high 需要 2319 请求。原字节另存，模型使用最长边 512 的派生图，
并保持图/帧/原始 source ID 映射。每帧描述上限 512 UTF-8 字节，视频聚合仍属同一 native ID。
low 不保证细小 OCR 质量，与 high 条件须明确区分，不能把省调用等同于更好质量。

这不是立即可运行的承诺：现有 3 图 analyzer 不会偷偷改成 8 图；
需要新的离线契约、经批准且计入 870 中的小批验证，成功结果缓存不重复。
原邮件采用官方提供的原生文本，不宣称恢复了未公开原始邮箱。
视频音轨范围仍须明确，gpt-4o-mini 不原生接收音频/视频；当前帧适配遇音轨会阻断，
不能静默忽略后称完整影音条件。隐私审查、可信解码工具与新 source 表示上界仍是前置条件。

[官方图像文档](https://developers.openai.com/api/docs/guides/images-vision)
规定 gpt-4o-mini 的 low 按每图 2833 base tokens 计；8 图 22664，
连同 8192 文本、1024 framing 余量、2048 输出约 33928，低于 128k 上下文。
这是预算可行性推导，未用真实请求验证该新批处理器。

## 费用与方法控制

完整比较的请求并非为 31 个 reader“凭空预占几千次”：
1104 次是完整来源库的一次文本编码，105 次是全部 1044 个原生问题的共享 query 缓存；
reader 与 proxy judge 仅 75 次。严格条件再加 870 次源图/视频帧表示构建，随后所有方法共用。
若要 V-Mem-style query anchor，另至多 31 个 LLM＋4 个 embedding 请求，预占另 7.95 元，
仅作查询假设，不写记忆事实。不是原版 MiniLM/SigLIP V-Mem 的等价复现。

对照保持同 100 候选/20 个 native evidence、8192 文本 token、最多 2 图和 768 输出 token。
不同稀疏细粒度视图先用本地 BM25，dense 共用 parent 表示；这一省成本适配必须命名披露，
不假称已复现全局细粒度 dense 通道。检索先跑全题，reader 只在预登记的新题集对照；
不据 N0 的具体错题改 prompt 或关键词。

[百炼文档](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)
确认 text-embedding-v4 最多 10 行/单行 8192token，256 维可用，标准价 0.5 元/百万输入 token。
[gpt-4o-mini 模型页](https://developers.openai.com/api/docs/models/gpt-4o-mini)
标准价 USD0.15/0.60 每百万输入/输出 token。使用 0.05 元/embedding 请求、0.25 元/LLM 请求保守预占，
预算折算 10 元/USD 为保守假设，非市场汇率，不假设额度、折扣或 Batch API 降价。
本计划不是实际账单估价；实际 usage 单独入账，未知结果保留占用且不自动重试。

读取日期 2026-10-25。
[官方弃用表](https://developers.openai.com/api/docs/deprecations)
列 gpt-4o-mini-2024-07-18 于 2027-01-14 停止服务。当前继续保留比赛指定模型，
不按官网推荐自动替换其他模型；真正新运行前再核实模型可用性与价格。

当前预算余 483.711685 元，296.70 元规划在数值上可容纳，但**未获本轮执行授权**。
须先完成原媒体向服务商传输的隐私/许可边界、音轨与解码核查，
以及新独立有界 guard 批准；绝不重开已 complete 的 84 次 guard。
本轮不下载原媒体、不发模型请求、不部署、不运行 GPU、不推 GitHub。
