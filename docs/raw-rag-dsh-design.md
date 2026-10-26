# RAW-RAG / DSH 核查与 V6 的实际边界

状态日期：2026-10-26。**V6 研究实验已经完成；V6 方法尚未完整接入在线服务；尚未发布正式 V6 release。** 这里的 V6 是冻结实验版本号，不能据此称在线产品已升级或官方评测已通过。已核验成绩见 [V6 结果](v6-results.md)。

| 层面 | 当前状态 | 能说明什么 |
|---|---|---|
| V6 批处理研究实验 | 检索、Reader 响应恢复和新代理盲评完成 | 固定条件下的真实开发集实验已有证据和结果 |
| 在线 Add / Search 服务 | 尚未整合完整 V6 检索、重排与 Reader 路径；默认服务仍为离线模式 | 实验脚本可运行，不等于线上请求会执行同一条路径 |
| 正式 release / 官方成绩 | 尚无正式 V6 release；没有本轮官方 QS、Smoke 或 Full 成绩 | GitHub 源码、Docker 构建、实际部署和正式评测是分别需要验证的交付项 |

“离线流水实验”指通过独立脚本冻结输入、批量推理和隔离评分，不经在线服务接口执行；它不表示使用假模型或没有真实推理。V5 已取得真实 `text-embedding-v4` 向量；V6 复用这些向量，进行了真实 GPU 候选重排和 `gpt-4o-mini` Reader / judge 调用。当前使用已评估的公开开发集、外部 Qwen SGM 文本；16 题 Reader pilot 也不是未见测试集或严格原始媒体端到端评测。

## RAW-RAG：已借鉴什么

[RAW-RAG 原论文 v1](https://arxiv.org/html/2610.30755v1)研究一种保留来源文本的基线：不先用 LLM 抽取、摘要或图结构替代原文，而是检索、重排后交给 Reader 推理。“raw” 包括基准自带的图片 caption 文本，并不意味着直接读取原图。其官方代码采用 MIT 许可证，第三方模型和数据仍适用各自许可证。[官方仓库](https://github.com/Litmeb/RAW-RAG)

| 方法特点 | 原论文 / 固定实现 | 本项目已做 / 未做 | 下一轮可归因实验 |
|---|---|---|---|
| 保留原始 history，避免有损摘要替代 | 保留消息文本、说话人和日期，已有 caption 作为文本进入 | 已保留完整原生来源 ID 与 SGM / 邮件文本，没有再用摘要替换它们；但 SGM 本身仍是作者提供的派生表示 | 有对应原始记录后，单独比较原文与派生索引；不能从现有 SGM 还原已丢失的图像信息 |
| 重叠窗口检索 | 默认 256 tokens、32 tokens 重叠，窗口保留到原消息的映射 | 未实现这一配置；V6 沿用 V5 的来源表示和向量缓存 | 在同一来源内部切窗口，保留位置和 native ID；来源级合并后再计候选名额 |
| 充分召回后重排 | cosine Top128 → cross-encoder Top15 | 已实现关键思路：Dense / Hybrid 各自固定 Top100，由同一 reranker 评分，比较有无重排 | 保留当前重排作为固定对照，不同时改 chunk、候选数、权重和 Reader |
| 完整 turn 回填 | 命中窗口可展开为它覆盖的完整 source turns；默认重复 turn 不去重 | 尚未实现；当前 Reader 读取完整原生来源的预算前缀，ATM 原生来源不能被伪称为对话 turn | 只有真实 turn / source 映射存在时才回填；单独比较窗口、完整来源与预算内回填 |
| Reader 证据预算 | 主条件没有额外 evidence-token cap，Top15 不能代表固定 token 成本 | 已固定最多 Top20、8192 text tokens、24000 UTF-8 字节、0 图、768 输出 tokens | 保持统一预算，记录打包后覆盖与实际来源数；再独立检验去重及互补证据选择 |

上述默认参数可核对[固定配置文件](https://github.com/Litmeb/RAW-RAG/blob/95639191e9547219b426710e50fe1a36235e26fc/methods/EverCore/evaluation/config/systems/rag.yaml)。主配置开启显示时 turn 回填并关闭显示时去重，部分基准另有覆盖设置，不能称所有实验都采用完全相同的回填规则。论文附录 B 说明文本表示、检索与回填边界；§4.3 中 128–1024 token 块大小的消融没有显著改变总分，因此 256/32 是候选配置，不是已知适合本项目的最优参数。

本次核查的固定来源身份：

- 仓库 commit：`95639191e9547219b426710e50fe1a36235e26fc`。
- 文件：`methods/EverCore/evaluation/config/systems/rag.yaml`。
- 原始文件 SHA256：`788ffa4c8c33fd6872a03e869790549b030e8486e315fd943a976bfd99f269c6`。
- 原始长度：4478 bytes；于 2026-10-26 从该 commit 的官方 raw URL 读取并计算哈希，未执行外部仓库代码。

V6 因此应描述为 **RAW-RAG 启发的固定候选重排实验**，不能称 RAW-RAG 论文复现。其模型、数据、候选粒度、Reader 预算和评分设置都不同，分数不宜与论文横向排名。

## DSH：候选指代尚未确认

本项目文档及管理 workspace 的 Cycle1 研究记录没有给出 DSH 的定义，用户尚未确认具体仓库或论文。当前最可能的候选是 **DeepSeek Harness**；以下内容仅对应这个候选，不代表已确认用户所指，也不代表本项目已采用其 memory 算法。历史 Cycle1 文档中针对 Cemory 的建议，不自动成为本项目的设计或实现记录。

[DeepSeek Harness 官方 README](https://github.com/deepseek-ai/deepseek-harness)将 `dsh` 定义为插件化智能体框架。它不是单一记忆检索算法；不同社区 memory 插件不能混为一种方法。官方架构的可借鉴点包括可替换的模型 / 工具组件、配置层、追加式 session 事件日志和从日志派生模型历史。[官方架构文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)、[session 日志文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md)

| DSH 候选机制 | 本项目当前状态 | 合理借鉴范围 |
|---|---|---|
| 插件化组件与配置分层 | 没有集成 DSH；现有检索、Reader、provider 适配器分别实现 | 保持接口和实验配置明确，便于替换组件与复现，不要求迁移整套运行时 |
| 追加式 session 日志、派生模型历史 | 已有本项目自己的持久收据、run guard 与恢复机制；不能归功于尚未采用的 DSH | 可研究更完整的事件模型和运行溯源，明确 project / session / run 的关系 |
| 独立 workspace / session | 当前实验已有独立运行目录与冻结身份 | 产品化时保证重启、恢复及用户隔离，避免把新会话和新实验混淆 |
| 长期记忆插件 | 未确认用户具体指哪个插件，也未接入 | 获得明确链接后再审读其存储、检索、模型依赖、许可证和评测，暂不宣称效果 |

DSH 官方引用的底层 [Cordis 论文](https://arxiv.org/abs/2608.25512)研究动态组件组合、依赖和可撤销副作用，不是 ATM 长期记忆得分论文。其官方项目目前标记为开发者预览；发布和 Docker 部署本项目不依赖改用 DSH，也不构成更换本项目指定 embedding / LLM 的理由。

## 下一轮的验证边界

下一轮应从**未用于当前方法选择的独立完整场景**开始，先冻结来源、方法、模型、预算和评分规则。当前 16 题及 Regular / Hard 已有结果全部保留，不换 seed、不按题义重选、不调 prompt / RRF 权重反复追分，也不把它们重新称为 holdout。

按最小消融顺序推进：先固定当前重排和 Reader，比较来源表示与窗口检索；再固定排名比较窗口 / 原生来源回填；最后在同一预算下单独比较去重和互补证据选择。每一步同时报告来源级检索、打包后覆盖、Reader 指标和成本，避免把多个改变的收益归于单一机制。公开开发集上的探索结果与独立场景的确认结果应分别记录。

在线整合及 release 验收则另行验证：Add / Search 路径是否执行冻结配置、重启后数据是否恢复、用户隔离和资源预算是否成立、Docker 内实际接口是否可用。完成这些工程检查也不自动产生官方 benchmark 成绩。
