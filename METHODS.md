# TraceMem-MM 多模态方法映射

本项目借鉴下列已核查实现的设计，不声称原样复现，不复制完整框架。当前能够执行的是原图/来源持久化、原文和卡片检索、指定模型适配与预算保护；结构化区域/OCR、保守语义实体边和最终 reader 尚未完成。新方法需在许可明确的原生数据上冻结后比较，不能追着公开测试结果改 prompt。

## 研究实现与取舍

| 来源与固定版本 | 可移植设计 | 本项目对应点与限制 |
|---|---|---|
| [ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench/tree/ef4e5dff1a47ec71213a06e359f02753defa8fb1)，MIT 项目 | image_processor 的 caption/OCR/entity、来源绑定、原图和 evidence ID 给 reader | 采用“一次指定 LLM 结构输出、多字段索引”的计划，降低多次视觉调用；实体字段必须显式纳入索引。上游预生成 SGM 由 Qwen3-VL-2B 产生，不能混入严格指定模型流水线。数据媒体许可另审。 |
| [V-Mem](https://github.com/Dingyi-Kang/V-Mem/tree/36916b14dc5241e04acbf5cd0f3c40799bc09550)，MIT | Session/Round/Unit、BM25/dense/RRF、round binding、有限时间邻居、图像关键词锚点 | 使用 V-Mem-inspired 的 provenance 层和可关闭扩展；不照搬默认 MiniLM/SigLIP。必须保留所有 image parts，不能仅取首图。查询锚点只用于检索，不写为记忆事实。 |
| [M3-Agent](https://github.com/ByteDance-Seed/m3-agent/tree/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c)，Apache-2.0 | episodic/semantic 分层、实体边 | 仅借组织方式；不照搬 text-embedding-3-large、额外视觉模型或面部身份匹配。图像检索不能由 face 节点效果推断为通用视觉能力。 |
| [Mem-Gallery 数据](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery/tree/af912daba984e896e253016b7c7e334ef92c2a6f)，MIT 数据卡 | 原生多模态记忆 benchmark/harness | 它是评估基准，不是新架构。MMMemory/UniversalRAG 的 GME embedding 不能作为指定 text-embedding-v4 的合规实现照搬；代码仓库 revision 与 HF 数据 revision 必须分开。 |

## 统一架构与待实现部分

权威层保留原文、原图 blob 与独立 occurrence；同一 blob 在不同会话不是同一事件。来源卡与原始引文分栏，卡片包含生成器、来源、文本证据和不确定性。计划的结构字段为全局视觉描述、可见文字、区域/对象、关系与保守实体候选，均由 gpt-4o-mini 完成，不引入额外 OCR/CLIP 模型。当前只有 summary/evidence_quotes，真实小样本暴露了多数 summary 忽略视觉内容的问题，结构提取仍需开发集验证。

text-embedding-v4 只编码文本桥；图文先由指定 LLM 形成检索描述，不能把该向量称为原生视觉 embedding。原文 BM25 始终保留，卡片并联 BM25+dense/RRF；来源 round 绑定和有限邻居必须控制最大条数。原图候选按相同预算传给最终 reader，以检验文字桥遗漏是否可被证据复核纠正。当前 reader/答题/拒答未实现。

## 冻结的 A/B/C 比较设计

- A：原文 + 全局图像描述的 flat RAG，作为必须的基线。
- B：A 并联结构 OCR/区域/来源卡；不删除原文，不用测试 gold 生成卡片。
- C：B 增加保守实体边与有限时间/round 邻居；缺少明确时间或实体证据时保留不确定，不能靠规则读 gold。

各组固定相同 query 改写、融合参数、reader、候选数、原图数量/像素及文本 token 上限。B-no-image 和 C-no-link 分别诊断视觉与关联增益。保留完整原生 distractors，不能只索引 gold 相关记录。Qwen reranker 是另一个 scoring-only 独立消融开关：官网明确不限制 reranker，无需另设组委会许可；本轮仅为控制变量关闭，不可用于生成或替代指定模型。上述 A/B/C 是下一原生数据里程碑设计，尚无其原生 benchmark 结果。

## 证据与指标

evidence ID 必须指向来源 occurrence，多个卡片不能虚增证据数量。固定 evaluator commit 和 ID 映射，同时报告 `|retrieved∩GT|/|GT|`、any-hit、all-evidence coverage；三者不能混称 Recall@K。上游 ATM 文档和部分代码定义存在差别，因此本项目提供独立、命名明确的 harness-only 函数，未宣称复现上游总分。没有 GT 的问题不获得虚构的完美 recall，需要单独计算支持不足拒答和答题指标。

图像证据命中、跨会话多证据完整覆盖、误关联、拒答、端到端答题准确率、成本和延迟应分开。当前真实诊断只有 6 个文本检索问题的单证据结果；完整多模态 reader、自然时间和复杂实体推理均无性能结论。只读元信息或数据卡许可不等于 AML 已批准的开发 split，也不自动授权将个人媒体发给外部 API。
