# V6：Raw-RAG 启发的固定候选重排消融

**历史设计与启动记录（2026-10-25）。** 本文保留当时的冻结设计和首批启动状态，文中的“正在执行”“尚未产生结果”等表述仅适用于该历史时点。V6检索现已完成；已核验成绩、Reader故障恢复与补评状态以[最新V6结果](v6-results.md)为准。

当时用户在讨论 Raw-RAG 论文后明确要求“尝试推进”；首批256对完成，完整145,629对开始在4个固定分片执行。以下内容不是当前完成成绩。

## 问题与固定条件

V5 Hard31 的 Dense Recall@100为60.78%，Recall@20为40.72%。V6检验：已经进入候选池的证据能否通过重排前移，以及前移能否改善统一reader的回答。

四组为Dense、Dense+reranker、Hybrid、Hybrid+reranker。每种检索器各自固定V5 Top100，只有组内有无重排改变；不强行让Dense与Hybrid共享同一候选集合。同一问题两组候选的并集只评分一次，共1,044题、145,629对。来源仍是完整11,034个原生来源的作者SGM/邮件格式化文本；不重新分块、不重建embedding、不更改RRF权重。并列分数保留各自原排名。

R@20为主要检索端点，同时记录完整证据覆盖与至少命中比例。重排前后的R@100及相应覆盖必须逐题一致；它们是候选集合不变的检查，不是V6能提高的指标。配对区间是已曝光开发集上的描述性统计，不是未见数据的确认性检验。

## 实际重排

- 模型：`Qwen/Qwen3-VL-Reranker-2B`，固定revision `4bd860ac4f15ad1897a214615cccc700f8f71818`；只进行文本候选相关性打分，不生成、不训练。
- 复用已校验权重及现有GPU1_zhilian环境；四个空闲GPU固定分片，各次执行最多7200秒。已有已提交分数在后续执行中直接复用。
- BF16、SDPA、left padding、batch最多8、padding后最多8192 tokens；单对最多4096 tokens。完整保留原问题与评分模板，仅超长来源取文本前缀，并记录截断数。
- 用最后有效token的yes-minus-no logit作分数，FP32投影，避免sigmoid饱和造成额外并列。
- 源文、问题与score保留在私有本地/自有远端运行目录；不上传Git。gold只在独立评分阶段读取。

首批256对实测完成：90.31秒（含校验/加载）、120,798输入tokens、峰值allocated4493.25MiB、0截断、0付费API调用。吞吐只用于观察运行条件，不据此挑选方法或问题。完整任务尚未产生结果。

## Reader开发集pilot

固定seed20261025，只按dataset和题ID的稳定哈希选择Regular8+Hard8，选择在新结果产生前冻结；不读题义或答案选题，不换seed。四组共64份回答。固定gpt-4o-mini、Top20完整来源前缀、8192输入text tokens、24000 UTF-8字节、0图、768输出tokens。保留原生来源ID，支持列表问题；记录实际打包后来源覆盖，不能只报打包前Top20。

来源/题型分布如实保留：固定样本有10道开放题、4道数字题、2道列表题；Regular子集没有列表题。数字与列表采用固定上游确定性评分；开放题采用独立gpt-4o-mini盲化代理判分，不能称官方QS。拟每道开放题的4个答案在同一请求中打散呈现；最多16次代理judge调用，连同64次reader，总上限80次、累计预留20元，计入既有项目500元预算。

所有HTTP响应在解析前保存私有收据；异常结果不自动重发。旧完成guard不重开，历史账本不重置。reader看不到gold，judge仅在全部四组答案完成之后由隔离脚本准备。

## 复现与证据

离线阶段：`prepare_atm_v6_rerank.py`、`score_atm_v6_rerank.py rank/score`、`prepare_atm_v6_reader.py select/pack`。主运行证据：`reports/atm-v6-rerank-preparation.json`、`runtime/atm-v6-rerank/FROZEN_CONFIG.json`、`reports/atm-v6-live-progress.json`、`runtime/atm-v6-reader/FROZEN_SELECTION.json`。只有`RERANK_RUN.json` complete且全量hash通过才能产生最终新排名。

这不是原论文模型配置的复现。Raw-RAG使用文本历史和已有caption，且部分设置把窗口回填为完整turn；本轮仅借鉴其重排思路，分块、回填和原始媒体路线另行控制。下一独立Mem-Gallery场景尚未下载或评估。

- [Raw-RAG论文v1](https://arxiv.org/html/2610.30755v1)
- [Raw-RAG固定版本配置](https://github.com/Litmeb/RAW-RAG/blob/95639191e9547219b426710e50fe1a36235e26fc/methods/EverCore/evaluation/config/systems/rag.yaml)
- [Qwen固定模型说明](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/4bd860ac4f15ad1897a214615cccc700f8f71818/README.md)
- [指定reader计价](https://developers.openai.com/api/docs/models/gpt-4o-mini)：每百万输入/输出tokens为0.15/0.60美元；预算使用既有10 CNY/USD保守假设，不表示市场汇率或最终账单。

本文到此为历史设计快照。请查看[最新V6结果与Reader恢复状态](v6-results.md)；不要据本页历史运行状态重新启动已完成实验或把检索Recall当成答题正确率。
