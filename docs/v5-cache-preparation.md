# V5：共享向量缓存与检索准备

2026-10-25。用户已停止原 Dot 工作会话，工具复核其状态为 idle；由当前会话统一推进。此里程碑完成本地准备，没有新的 benchmark 成绩或 provider 调用。

## 实际完成

- 修复继承的视觉恢复代码：凭据在 JSON 键和异常转义响应中的清理、已有 journal 清单损坏时发送前拦截、恢复阶段拒绝非有限数值和重复 JSON 键。恢复测试从15项增至19项，均为本地故障注入。
- 新增 `SourceEmbeddingCache`：scope固定 namespace、数据revision、表示版本、指定模型和维度；按精确UTF-8内容hash复用已成功向量，同内容的多个native来源仍分别保留。
- 原文按最多8192字节无损切块，完整来源manifest先冻结，批次最多10条。缺向量不能publish；不同向量不能覆写已成功缓存。纯空白来源或分块明确拒绝，不trim、不删来源。
- 新增 `SharedDenseIndex`：搜索全部已发布source chunks，按native来源取最大cosine；固定hybrid使用flat BM25和同一份dense缓存、等权RRF60。最终最多100个来源，query及query vector不写盘。
- 新增本地完整库准备入口，N0 source-index以只读方式打开；不读取问题/gold，不创建provider transport、预算账本或真实guard。

## 全库准备结果

| 项目 | 实测状态 |
|---|---:|
| 原始来源 | 11,034 |
| 图像 / 视频 / 邮件来源 | 3,759 / 533 / 6,742 |
| 唯一文本块 | 11,035 |
| 源embedding批次（每批≤10） | 1,104 |
| 已有真实ATM向量 | 0 |
| 已发布dense索引 | 否 |
| 本轮provider请求 / 费用 | 0 / 0元 |

沿用此前每个embedding请求0.05元的保守预占假设，**仅source编码的拟议预占为55.20元**；它不是实际费用、不是本轮授权，也不含未来query/reader请求。已有全部来源仍为作者Qwen SGM条件，不能称严格原媒体端到端。

首次准备记录和hash保留在 `reports/atm-v5-cache-preparation.json`。之后补入了纯空白chunk拒绝规则；只读核验现有11,035块没有这种情况，缓存字节和source manifest均未改变。新代码hash及核验在 `reports/atm-v5-cache-validation.json`；没有重建缓存或重跑任何冻结题。

## 已验证范围

最终源码全套 **291项通过**；整理后导出全套 **256项通过**。缓存15项、dense/hybrid4项、视觉恢复19项专项均通过。合成向量只验证机制，不计为性能成绩。N0、V4、Mem-Gallery完成报告和真实预算/Smoke账本hash保持。

导出包含review后的模块、合成测试、文档；原始来源文本、SQLite缓存、真实响应、gold、逐题排名和账本仍在本地忽略目录。视觉审计测试需要可选依赖 `Pillow==13.0.0`，缓存与dense代码仅依赖标准库。

## 开始真正编码前还需要什么

当前缓存是“已成功向量仓库”，不是完整的付费执行器。`missing_batches()`只说明缺缓存，**不表示可以发送或重试**。后续执行器必须逐批保存响应，绑定run ordinal与budget attempt，并对unknown/inflight保持阻断；不能因为缓存里缺向量就重复收费。导入receipt hash是来源引用，不会自动校验或结算供应商账单。

现有 `DevelopmentRun` 只有50/100请求范围，不能承载1,104次全库source请求。新运行须有单一明确上限，不能拆多个run重置计数，也不能重开旧完成guard。ATM向供应商传输的内容范围仍须明确；本轮只做本地处理。

准备好执行器和范围后，再构建一份共享source编码，并在新的预登记问题/完整场景上比较dense-only和固定hybrid。已经测过的Hard31保留为历史结果，不继续调参追分。reader和选图消融放在证据覆盖验证之后。

相关记录：[V4结果](v4-sparse-results.md)、[原视觉恢复设计](visual-recovery.md)、[ATM准入检查](atm-admission-review.md)。
