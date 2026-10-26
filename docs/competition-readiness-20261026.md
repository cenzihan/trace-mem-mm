# 多模态学术组准入准备与真实模型 Smoke

核对日期：2026-10-26。依据官网实际申请弹窗、Multimodal Smoke 入口、[接入指南](https://agentmemoryleaderboard.ai/api-guide)、[评测文档](https://agentmemoryleaderboard.ai/docs)及[比赛说明](https://agentmemoryleaderboard.ai/competition/)。这是准入准备记录，尚未提交申请，尚未取得 AML Eval Key，官方 Smoke 使用次数仍为 0。

## 已核实的真实模型结果

| 测试 | 执行位置与范围 | 实际结果 |
|---|---|---|
| text-embedding-v4 | 先前隔离 Docker profile 的 Add/Search 与重建后查询 | 3 次 HTTP200，256 维向量，54 输入 tokens；同一份父账本预留已按收据核对 |
| gpt-4o-mini | 本机直接官方 HTTPS，单独连接与语义检查 | 1 次 HTTP200，返回 gpt-4o-mini-2024-07-18；50 输入、11 输出 tokens；正确提取合成来源中的船名 |
| gpt-4o-mini | 当前部署容器 tracemem-api-1 内部，直接官方 HTTPS | 1 次 HTTP200，返回同一模型快照；47 输入、44 输出 tokens；生成摘要与逐字来源引用 |
| Qwen/Qwen3-VL-Reranker-2B | 现有私有 GPU 主机，固定权重、真实 bfloat16 forward | 6 个合成问题—文档对；2 道问题的正确来源均排名第 1；仅打分、不生成答案 |

容器 GPT 的原文引用校验成功，耗时约 5.28 秒。Qwen 固定 revision 为 `4bd860ac4f15ad1897a214615cccc700f8f71818`，权重 SHA256 为 `466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9`；所有固定模型文件均核验。6 对 forward 约 0.94 秒，显存峰值约 4096 MiB；包含加载与校验的整次 GPU 运行约 50 秒。这些数值不是吞吐或容量保证，输入只有合成文本。

本机 GPT 首次结果被复用的 source-card 校验器拒绝，因为测试只要求简单 JSON。原 HTTP200 收据、`invalid_contract` 观察和原答案完整保留；追加审计确认连接及语义检查通过，以 `rejected_response` 关闭该 guard，没有改写原收据或重发请求。随后容器调用使用真实卡片格式并通过校验。

本轮新增 GPT 请求总计 2 次，各自独立一次性 guard、最多 64 输出 tokens、无自动重试，共保留 CNY0.50 预算预留，纳入原项目 CNY500 账本。成功 usage 估计合计 USD0.00004755，账单未核销；预算预留不等于实际花费。之前 3 次 embedding grant 已耗尽，不重开。私有收据与执行程序保留在操作者工作区，不进入仓库。

**能力边界：** 目前常驻 Add/Search 仍报告 Dense 与 reranker 禁用。上述 GPT 容器调用及 Qwen GPU smoke 是真实组件测试；embedding 曾单独验证动态 Add/Search，但三个组件尚未在同一个在线候选版本中完成端到端验收。不能据此申报“在线 Dense＋Qwen 已部署”，也没有官方榜单分数。

## 实际申请表与当前缺口

官网选择 Academic leaderboard 后，表单包含以下字段。联系人具体值只保存在私人草稿及用户的官网表单，不随本仓库发布。

| 表单字段 | 当前情况 |
|---|---|
| Leaderboard track | Academic / Open-source Methods；说明中明确 Multimodal |
| Work email（必填） | 用户已提供，并填入官网草稿 |
| Name、Organization / Team（选填） | 用户已提供，并填入官网草稿 |
| Invitation code（选填） | 留空，不影响准备 |
| System name（必填） | TraceMem-MM 已填入草稿 |
| Version name（必填） | 待在线模型链路完成后固定；研究 V7.1 不自动等于部署版本 |
| Add API URL、Search API URL | 仍为服务器回环服务；平台需要稳定公网地址，当前不填写不可达地址 |
| Authentication | 草稿已选择 Authorization: Bearer，与当前服务一致 |
| Memory System Key（启用鉴权必填） | 服务端已生成；最终仅配置到官方专用安全字段，不放进说明、URL 或仓库 |
| Health URL（选填） | 现有 /health 正常，公网入口完成后填写 |
| Add status URL（选填） | 留空；当前同步 Add，不申报 HTTP202 异步支持 |
| Public GitHub repository URL（必填） | 仓库当前私有；公开发布及许可证仍需确定，公开前审查最终提交历史和构建内容 |
| Submission notes and run instructions（必填） | 按本记录与部署文档整理；最终版本、模型和容量声明必须与运行实例一致 |

版本运行说明应披露：原图与有序文本保留、严格 user_id 隔离、固定模型与 reranker 版本、用户向量索引、候选数、同请求幂等、同步完成语义、持久卷、备份恢复、认证、限流、容量、超时、来源和方法改动。当前一个 Gunicorn worker／四线程、容器 1 GiB 上限和 64 MiB JSON 上限是配置值；没有相应并发负载或 30 分钟 SLA 实测，不能写成承诺。

官网文档明确：Embedding 使用 text-embedding-v4，LLM 使用 gpt-4o-mini，reranker 不限制；Qwen 只用于候选打分，不替代 LLM、embedding 或平台 Answer/Eval。申请表另有笼统的模型提示，提交说明需明确组件职责，并附该现行文档依据。方法来源及 RAW-RAG / DSH 借鉴边界见 [研究说明](raw-rag-dsh-design.md)；不能把未实现的窗口回填或其他上游功能写成已实现。

## Smoke 的实际前置条件

点击 Start a track Smoke → Multimodal 后，官网要求输入已签发的 `ldbd_key`，验证后选择已绑定版本。用户确认尚未收到此 Key，故没有发起官方任务，也没有消耗官方 Smoke 次数。Memory System Key、OpenAI key、百炼 key 和 AML Eval Key 各有不同用途，不能混用。

下一步顺序：完成同一在线候选的模型接入与隔离验收 → 配置稳定的公网 Add/Search/Health → 冻结代码、镜像及运行声明并准备公开仓库 → 复核并提交准入申请 → 取得绑定 Key 后运行一次官方 Multimodal Smoke。保留现有用户授权总上限 5 次；官网标准上限为本届每赛道 30 次、每小时 1 次，两种限制同时遵守。Full 另属正式评测，本次准备不启动 Full。

官网材料提交截止为 **2026-10-31 23:59（UTC+8）**，评测截止为 **2026-11-04 23:59（UTC+8）**。评测数据及派生物应仅用于评测，并在结束后 30 天内清理，包括备份；提交前还需落实对应操作流程。
