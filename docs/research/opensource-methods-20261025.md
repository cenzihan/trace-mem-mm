# 开源方法核查与本项目选择

核查日期：2026-10-25。只采用作者论文、官方仓库与源码；本轮没有运行上游框架或复制其实现。

| 方法 | 实现与许可 | 可用思路 | 本项目边界 |
|---|---|---|---|
| ATM-RAG / SGM | 代码 MIT；固定主表入口为 MiniLM + Qwen3-VL-8B-Instruct-FP8，SGM 由 Qwen3-VL-2B 预处理 | 原生 source ID、字段格式和统一证据预算 | 现有 N0/V4 都是外部 SGM 研究条件；替换 reader/embedding 仍不等于严格从 raw 构建 |
| V-Mem | MIT；文本 MiniLM、视觉 SigLIP、查询侧 caption/anchor | 细粒度匹配后聚合回原始 round/parent，避免碎片占满输出 | V4 只借聚合结构，使用本地 BM25；不声称原版复现或新视觉能力 |
| M3-Agent | Apache-2.0；memorization/control 模型、实体图，涉及人脸/音轨/说话人和独立 embedding | 保留事件实体与原始来源，逐步补充跨片段证据 | 当前模型/媒体范围不同，暂不引入整套框架或把其 M3-Bench 成绩比较到 ATM |
| LightMem | MIT；主题分段、分层记忆和离线合并；示例还依赖 LLMLingua-2、MiniLM、Qdrant | 将来源构建和在线查询分开，共享缓存和批处理 | 借成本组织方式；不能直接搬默认模型/压缩器到指定模型主线 |

## 分数参照不能混用

ATM 作者固定版本主表的 ATM-RAG regular Recall@10 为 68.7、QS 为 51.0。当前 N0 Fractional Recall@10 为 68.857%，只是数值接近：检索器、格式及 reader 配置没有全部控制，不能宣称超过 ATM-RAG。N0 没有 QS。

V-Mem README 报告的 0.825 是完整 Mem-Gallery 1,711 题的 LLM-judge 分数，不是 Recall，也不是我们第二场景 10 题 reader 的可比成绩。AML 平台统一 Answer/Eval；本地 proxy 结果均不是正式榜单成绩。

## 当前优先级

1. V4：完整来源、多视图 sparse、native parent 聚合、固定新题对照。
2. 后续：一次构建的共享 parent dense 缓存，独立 dense-only / hybrid 对照；不为每个方法重建整库。
3. 另做证据选择消融：图像 occurrence 与实际返回证据一一对应，并报告完整证据覆盖。现有 reader 文本没有截断，不能把主要问题归因于上下文长度。
4. 查询 anchor 和实体图推迟到更强控制建立以后；查询假设不写入记忆事实。

## 原始依据

- [ATM 主表与数据说明](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/README.md)
- [ATM 主表入口](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/scripts/QA_Agent/MMRAG/run.sh) · [MIT](https://github.com/JingbiaoMei/ATM-Bench/blob/ef4e5dff1a47ec71213a06e359f02753defa8fb1/LICENSE)
- [V-Mem README](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/README.md) · [retrieve.py](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/retrieve.py) · [MIT](https://github.com/Dingyi-Kang/V-Mem/blob/36916b14dc5241e04acbf5cd0f3c40799bc09550/LICENSE)
- [M3-Agent](https://github.com/ByteDance-Seed/m3-agent) · [模型配置](https://github.com/ByteDance-Seed/m3-agent/blob/master/configs/api_config.json) · [Apache-2.0](https://github.com/ByteDance-Seed/m3-agent/blob/master/LICENSE)
- [LightMem 论文](https://arxiv.org/abs/2510.18866) · [运行配置](https://github.com/zjunlp/LightMem/blob/main/experiments/longmemeval/run_lightmem_gpt.py) · [MIT](https://github.com/zjunlp/LightMem/blob/main/LICENSE)
- [AML 官方规则与流程](https://agentmemoryleaderboard.ai/docs) · [比赛说明](https://agentmemoryleaderboard.ai/competition/)

代码许可证不等于数据许可证。ATM 数据此轮沿用已核实的 CC BY-NC 4.0；个人来源文本、QA、gold、索引和模型响应继续仅保存在本地忽略目录，不随代码导出。
