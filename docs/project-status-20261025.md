# TraceMem-MM：接管分析与下一阶段

2026-10-25。当前已有两个 Mem-Gallery 场景、本地 ATM regular 全库基线，以及本轮 V4 Hard31 对照。没有正式 AML 榜单成绩。

## Benchmark 和成绩

| 条件 | 来源/问题规模 | 主要结果 | 解释 |
|---|---|---|---|
| Mem-Gallery 第一场景 | 185轮、31图、57题；49题有检索gold | A/B/C Recall@20 = 83.08/79.61/84.95% | C@1为20.09%，A为39.88%，邻居扩展没有稳定改善前排 |
| Mem-Gallery 第二场景 | 177轮、45图、77题；67题有检索gold | A/B/C Recall@20 = 74.92/77.21/78.40% | 同一批10题，proxy reader A6/10、C5/10；无稳定答题提升证据 |
| ATM regular N0 | 完整11,034来源、1,013题 | Recall@10/20/100 = 68.86/73.83/82.99% | 作者Qwen SGM文本＋本地BM25，无reader成绩 |
| ATM Hard31 flat | 同完整来源库、31题 | Recall@20/100 = 14.13/33.54% | 此轮新建立的同题控制，不能与regular分数直接相减当成版本变化 |
| ATM Hard31 V4 | 同完整来源库、同31题 | Recall@20/100 = 17.21/38.59% | 固定三视图BM25＋parent聚合；小幅改善，但不确定性仍大 |

这里的 Recall 是逐题“找回的标注来源数/该题标注来源数”的宏平均，不是答案正确率。Mem-Gallery按round，ATM按native artifact计数，不能混用单位。正式 Search K=100 与本地reader K=20分别报告。

V4主要指标提升3.08个百分点，31题5胜24平2负，配对bootstrap 95%区间[-0.51,+7.38]个百分点。V4未替换默认生产检索，Hard31配置和结果已冻结。详见[完整结果](v4-sparse-results.md)。

## 方法判断

当前最大缺口是更强的语义检索对照、跨来源完整证据覆盖，以及返回图像的选择。第二场景reader没有触发文本截断；不能把错误归结为上下文窗口装不下。合法引用也不代表支持充分。

ATM N0在邮件上Recall@20为91.08%，图像63.50%，视频62.67%；list-recall为55.10%。这说明外部描述的稀疏检索存在模态/题型差异，并不直接证明原图理解失败。Hard31的明显困难进一步说明，继续增加相关词法视图不足以替代语义/多证据能力。

研究了ATM-RAG、V-Mem、M3-Agent、LightMem。当前采用V-Mem式“细粒度匹配→原始来源聚合”结构，先实现零API适配；默认MiniLM/SigLIP等不带入指定模型主线。见[核查与原始链接](opensource-methods-20261025.md)。

## Git与GitHub

接管时，实验源码目录自身Git为没有提交的main；真正的导出Git仓库是相邻的`trace-mem-mm-export`。实验源码与可发布导出是两层，不能把导出commit当作全部实验源码的版本。

实时核验的[GitHub仓库](https://github.com/cenzihan/trace-mem-mm)仍为private、默认main；接管时远端HEAD为`2781867fa8c812f9022a79522f5291d85a062373`，本地导出HEAD为`ca5c04742448774e5c33aaa060557fbc80fe500e`，领先一提交。导出也缺N0、误差分析和新基线代码。没有已通过的GitHub CI证据。

本轮在导出仓库创建本地`codex/v4-sparse-hard31`分支，按清单同步代码、合成测试和聚合结果文档。保持原有导出定制测试；不复制原数据、QA/gold、runtime、原始响应、账本或日志。没有push；远端仍需单独审查后同步。随后用户停止原Dot工作会话，已核实idle并由当前会话接管；其可靠性改动经审查修复后进入V5准备版本。详见[v5缓存与接管说明](v5-cache-preparation.md)。

## 下一阶段顺序

1. 固定本轮版本和所有正负结果；维护来源、配置、源码和排名hash。旧A/B/C、N0、真实预算及Smoke账本保持。
2. 先补共享`text-embedding-v4`向量缓存，同一份来源编码供dense-only和固定hybrid共用。不能把“20或31个reader问题”解释为多次重建全库的理由。
3. 来源向量和query向量分别核算；保留完整11,034来源。当前ATM向量缓存为0，冷启动计划约1,104个source embedding请求，其金额上限须按实际表示重新核验，不能将保守预占当实际费用。
4. 在新的预登记题/完整Mem-Gallery场景上评估，不把本轮Hard31重新称heldout。Hard与regular共享来源，跨库泛化另测。
5. 单独验证证据/选图策略及reader，不把Recall当最终准确率。原始材料重建和外部SGM条件分开；原始多模态传输、视频音轨和解码须有明确范围。

当前继承开发总预算500元；最近已核实账本占用16.288315元、余483.711685元（含保守预占，不是实际账单），累计168次provider请求。本轮新增0。旧84次guard已完成，不能重开。下一次付费调用需要新的、范围清晰的运行安排，避免与Dot并行重复启动。

AML官方要求参赛方提供Add/Search，平台统一Answer/Eval；目前Smoke0、Full0，无部署或报名。本次交付为本地研究版本。[官方文档](https://agentmemoryleaderboard.ai/docs)与[比赛说明](https://agentmemoryleaderboard.ai/competition/)已于今日复核。
